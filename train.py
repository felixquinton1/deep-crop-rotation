import torch
import torchnet as tnt
import numpy as np

from sklearn.metrics import confusion_matrix
import os
import json
import pickle as pkl
import argparse
import pprint
from tqdm import tqdm
from models.stclassifier import PseTae, PseLTae, PseGru, PseTempCNN
from learning.loader import get_loaders
from learning.output import prepare_output, checkpoint, save_pred, save_results,\
    overall_performance_by_year, overall_performance
from dataset import PixelSetData, PixelSetData_preloaded

from learning.focal_loss import FocalLoss
from learning.weight_init import weight_init
from learning.metrics import mIou

from learning.temperature_scaling import ModelWithTemperature

def train_epoch(model, optimizer, criterion, data_loader, device, config):
    acc_meter = tnt.meter.ClassErrorMeter(accuracy=True)
    loss_meter = tnt.meter.AverageValueMeter()
    y_true = []
    y_pred = []

    for i, (data, y) in enumerate(data_loader):
        y_true.extend(list(map(int, y)))

        x = recursive_todevice(data, device)
        y = y.to(device)

        optimizer.zero_grad()

        out = model(x)
        loss = criterion(out, y.long())
        loss.backward()
        optimizer.step()

        pred = out.detach()
        y_p = pred.argmax(dim=1).cpu().numpy()
        y_pred.extend(list(y_p))
        acc_meter.add(pred, y)
        loss_meter.add(loss.item())

        if (i + 1) % config['display_step'] == 1:
            print('Step [{}/{}], Loss: {:.4f}, Acc : {:.2f}, IoU : {:.4f}'.format(i + 1, len(data_loader), loss_meter.value()[0],
                                                                    acc_meter.value()[0], mIou(y_true, y_pred, config['num_classes'])))

    epoch_metrics = {'train_loss': loss_meter.value()[0],
                     'train_accuracy': acc_meter.value()[0],
                     'train_IoU': mIou(y_true, y_pred, n_classes=config['num_classes'])}

    return epoch_metrics


def evaluation(model, criterion, loader, device, config, mode='val'):
    y_true = []
    y_pred = []

    acc_meter = tnt.meter.ClassErrorMeter(accuracy=True)
    loss_meter = tnt.meter.AverageValueMeter()
    for (data, y) in tqdm(loader):
        y_true.extend(list(map(int, y)))
        x = recursive_todevice(data, device)
        y = y.to(device)

        with torch.no_grad():

            prediction = model(x)

            if (config['save_pred'] and mode == 'test'):
                soft = torch.nn.Softmax(dim=0)
                for i, id in enumerate(data['pid']):
                    pred = soft(prediction[i]).tolist()
                    save_pred(pred, id, config)
            loss = criterion(prediction, y)

        acc_meter.add(prediction, y)
        loss_meter.add(loss.item())
        y_p = prediction.argmax(dim=1).cpu().numpy()
        y_pred.extend(list(y_p))

    metrics = {'{}_accuracy'.format(mode): acc_meter.value()[0],
               '{}_loss'.format(mode): loss_meter.value()[0],
               '{}_IoU'.format(mode): mIou(y_true, y_pred, config['num_classes'])}

    if mode == 'val':
        return metrics

    if mode == 'test':
        return metrics, confusion_matrix(y_true, y_pred, labels=list(range(config['num_classes'])))



def recursive_todevice(x, device):
    if type(x).__name__ == 'str':
        return x
    elif isinstance(x, torch.Tensor):
        return x.to(device)
    elif isinstance(x, dict):
        return {k: recursive_todevice(v, device) for k, v in x.items()}
    else:
        return [recursive_todevice(c, device) for c in x]




def model_definition(config, dt):

    if config['tae']:
        model_config = dict(input_dim=config['input_dim'], mlp1=config['mlp1'], pooling=config['pooling'],
                            mlp2=config['mlp2'], n_head=config['n_head'], d_k=config['d_k'], mlp3=config['mlp3'],
                            dropout=config['dropout'], T=config['T'], len_max_seq=config['lms'],
                            positions=dt.date_positions if config['positions'] == 'bespoke' else None,
                            mlp4=config['mlp4'], d_model=config['d_model'], with_extra=False, extra_size=None,
                            with_temp_feat=config['tempfeat'])
        if config['geomfeat']:
            model_config.update(with_extra=True, extra_size=4)


        model = PseTae(**model_config)

    elif config['gru']:
        model_config = dict(input_dim=config['input_dim'], mlp1=config['mlp1'], pooling=config['pooling'],
                            mlp2=config['mlp2'], hidden_dim=config['hidden_dim'],
                            positions=dt.date_positions if config['positions'] == 'bespoke' else None,
                            mlp4=config['mlp4'], with_extra=False, extra_size=None,
                            with_temp_feat=config['tempfeat'])
        if config['geomfeat']:
            model_config.update(with_extra=True, extra_size=4)

        model = PseGru(**model_config)

    elif config['tcnn']:
        model_config = dict(input_dim=config['input_dim'], mlp1=config['mlp1'], pooling=config['pooling'],
                            mlp2=config['mlp2'], nker=config['nker'], mlp3=config['mlp3'],
                            positions=dt.date_positions if config['positions'] == 'bespoke' else None,
                            mlp4=config['mlp4'], with_extra=False, extra_size=None,
                            with_temp_feat=config['tempfeat'])
        if config['geomfeat']:
            model_config.update(with_extra=True, extra_size=4)

        model = PseTempCNN(**model_config)
    else:
        model_config = dict(input_dim=config['input_dim'], mlp1=config['mlp1'], pooling=config['pooling'],
                            mlp2=config['mlp2'], n_head=config['n_head'], d_k=config['d_k'], mlp3=config['mlp3'],
                            dropout=config['dropout'], T=config['T'], len_max_seq=config['lms'],
                            mlp4=config['mlp4'], d_model=config['d_model'], with_extra=False, extra_size=None,
                            with_temp_feat=config['tempfeat'])
        if config['geomfeat']:
            model_config.update(with_extra=True, extra_size=4)


        model = PseLTae(**model_config)
    return model, model_config


def run(config, train_loader, val_loader, test_loader, dt, fold, device):

    model, model_config = model_definition(config, dt)
    config['N_params'] = model.param_ratio()
    with open(os.path.join(config['res_dir'], 'conf.json'), 'w') as file:
        file.write(json.dumps(config, indent=4))

    model = model.to(device)
    model.apply(weight_init)
    optimizer = torch.optim.Adam(model.parameters())
    criterion = FocalLoss(config['gamma'])

    trainlog = {}

    best_mIoU = 0
    for epoch in range(1, config['epochs'] + 1):
        print('EPOCH {}/{}'.format(epoch, config['epochs']))

        model.train()
        train_metrics = train_epoch(model, optimizer, criterion, train_loader, device=device, config=config)
        print('Validation . . . ')
        model.eval()
        val_metrics = evaluation(model, criterion, val_loader, device=device, config=config, mode='val')

        print(
            'Loss {:.4f},  Acc {:.2f},  IoU {:.4f}'.format(val_metrics['val_loss'], val_metrics['val_accuracy'],
                                                           val_metrics['val_IoU']))
        trainlog[epoch] = {**train_metrics, **val_metrics}
        checkpoint(fold + 1, trainlog, config)

        if val_metrics['val_IoU'] >= best_mIoU:
            best_mIoU = val_metrics['val_IoU']
            torch.save({'epoch': epoch, 'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict()},
                       os.path.join(config['res_dir'], 'Fold_{}'.format(fold + 1), 'model.pth.tar'))

    print('Testing best epoch . . .')
    path = config['res_dir']
    test_phase(config, path, val_loader, test_loader, dt, fold, device)



def test_phase(config, path, val_loader, test_loader, dt, fold, device):
    model, model_config = model_definition(config, dt)

    config['N_params'] = model.param_ratio()
    with open(os.path.join('conf.json'), 'w') as file:
        file.write(json.dumps(config, indent=4))
    model = model.to(device)
    model.apply(weight_init)

    model.load_state_dict(
        torch.load(os.path.join(path, 'Fold_{}'.format(fold + 1), 'model.pth.tar'))['state_dict'])
    model.eval()
    if config['save_pred']:
        model = ModelWithTemperature(model)
        model.set_temperature(val_loader)
    np.random.seed(config['rdm_seed'])
    torch.manual_seed(config['rdm_seed'])
    for year, loader in enumerate(test_loader):
        print("Année: {} ".format(config['year'][year]))

        criterion = FocalLoss(config['gamma'])
        test_metrics, conf_mat = evaluation(model, criterion, loader, device=device, mode='test', config=config)
        print('Loss {:.4f},  Acc {:.2f},  IoU {:.4f}'.format(test_metrics['test_loss'], test_metrics['test_accuracy'],
                                                             test_metrics['test_IoU']))
        save_results(test_metrics, conf_mat, config, config['year'][year], fold + 1)

def main(config):
    np.random.seed(config['rdm_seed'])
    torch.manual_seed(config['rdm_seed'])
    prepare_output(config)

    mean_std = pkl.load(open(config['dataset_folder'] + '/normvals_tot.pkl', 'rb'))
    extra = 'geomfeat' if config['geomfeat'] else None

    subset = None
    if config['preload']:
        dt = []
        for year in config['year']:
            dt.append(PixelSetData_preloaded(config['dataset_folder'], labels=year, npixel=config['npixel'],
                                             sub_classes=subset,
                                             norm=mean_std,
                                             extra_feature=extra))
    else:
        dt = []
        for year in config['year']:
            dt.append(PixelSetData(config['dataset_folder'], labels=year, npixel=config['npixel'],
                                   sub_classes=subset, norm=mean_std,
                                   extra_feature=extra, extra_feature_temp=config['tempfeat'], year=year,
                                   years_list=config['year'], num_classes=config['num_classes'], lms=config['lms'], return_id=True))

    device = torch.device(config['device'])


    if config['tempfeat'] > 0:
        config['mlp4'][0] = config['mlp4'][0] + config['num_classes']

    loaders, dt = get_loaders(dt, config['kfold'], config)

    if config['test_mode']:
        for fold, (val_loader, test_loader) in enumerate(loaders):
            print("Starting Fold: {} ".format(fold + 1))
            test_phase(config, config['loaded_model'], val_loader, test_loader, dt, fold, device)

    else:
        for fold, (train_loader, val_loader, test_loader) in enumerate(loaders):
            print('Starting Fold {}'.format(fold + 1))
            print('Train {}, Val {}, Test {}'.format(len(train_loader), len(val_loader), len(test_loader[0])))
            run(config, train_loader, val_loader, test_loader, dt, fold, device)

    for year in config['year']:
        overall_performance_by_year(config, year)
    overall_performance(config)





if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Set-up parameters
    parser.add_argument('--dataset_folder', default='path_to_your_folder', type=str,
                        help='Path to the folder where the results are saved.')
    parser.add_argument('--year', default=['2018', '2019', '2020'], type=str,
                        help='The year of the data you want to use')
    parser.add_argument('--res_dir', default='./results/model', help='Path to the folder where the results should be stored')
    parser.add_argument('--num_workers', default=8, type=int, help='Number of data loading workers')
    parser.add_argument('--rdm_seed', default=1, type=int, help='Random seed')
    parser.add_argument('--device', default='cuda', type=str,
                        help='Name of device to use for tensor computations (cuda/cpu)')
    parser.add_argument('--display_step', default=50, type=int,
                        help='Interval in batches between display of training metrics')
    parser.add_argument('--preload', dest='preload', action='store_true',
                        help='If specified, the whole dataset is loaded to RAM at initialization')
    parser.set_defaults(preload=False)
    parser.add_argument('--save_pred', default=False, type=bool,
                        help='Save predictions by parcel during test')
    parser.add_argument('--save_pred_dir', default='path_to_your_folder',
                        help='Path to the folder where the predictions should be stored')

    #Parameters relatives to test
    parser.add_argument('--test_mode', default=False, type=bool,
                        help='Load a pre-trained model and test on the whole data set')
    parser.add_argument('--loaded_model',
                        default='models_saved/model_you_want_to_load',
                        type=str,
                        help='Path to the pre-trained model')



    # Training parameters
    parser.add_argument('--kfold', default=5, type=int, help='Number of folds for cross validation')
    parser.add_argument('--epochs', default=100, type=int, help='Number of epochs per fold')
    parser.add_argument('--batch_size', default=128, type=int, help='Batch size')
    parser.add_argument('--lr', default=0.001, type=float, help='Learning rate')
    parser.add_argument('--gamma', default=1, type=float, help='Gamma parameter of the focal loss')
    parser.add_argument('--npixel', default=64, type=int, help='Number of pixels to sample from the input images')

    # Architecture Hyperparameters
    ## PSE
    parser.add_argument('--input_dim', default=10, type=int, help='Number of channels of input images')
    parser.add_argument('--mlp1', default='[10,32,64]', type=str, help='Number of neurons in the layers of MLP1')
    parser.add_argument('--pooling', default='mean_std', type=str, help='Pixel-embeddings pooling strategy')
    parser.add_argument('--mlp2', default='[132,128]', type=str, help='Number of neurons in the layers of MLP2')
    parser.add_argument('--geomfeat', default=1, type=int,
                        help='If 1 the precomputed geometrical features (f) are used in the PSE.')

    ## L-TAE
    parser.add_argument('--n_head', default=16, type=int, help='Number of attention heads')
    parser.add_argument('--d_k', default=8, type=int, help='Dimension of the key and query vectors')
    parser.add_argument('--mlp3', default='[256,128]', type=str, help='Number of neurons in the layers of MLP3')
    parser.add_argument('--T', default=1000, type=int, help='Maximum period for the positional encoding')
    parser.add_argument('--positions', default='bespoke', type=str,
                        help='Positions to use for the positional encoding (bespoke / order)')
    parser.add_argument('--lms', default=36, type=int,
                        help='Maximum sequence length for positional encoding (only necessary if positions == order)')
    parser.add_argument('--dropout', default=0.2, type=float, help='Dropout probability')
    parser.add_argument('--d_model', default=256, type=int,
                        help="size of the embeddings (E), if input vectors are of a different size, a linear layer is"
                             " used to project them to a d_model-dimensional space"
                        )

    ## Classifier
    parser.add_argument('--tempfeat', default=2, type=int,
                        help='Indicate the number of years of past ground truth you wish to use. If you are testing data '
                             'on a pretrained model, be sure to fill the same number of year.')
    parser.add_argument('--num_classes', default=20, type=int, help='Number of classes')
    parser.add_argument('--mlp4', default='[128, 64, 32, 20]', type=str, help='Number of neurons in the layers of MLP4')

    ## Other methods (use one of the flags tae/gru/tcnn to train respectively a TAE, GRU or TempCNN instead of an L-TAE)
    ## see paper appendix for hyperparameters
    parser.add_argument('--tae', dest='tae', action='store_true',
                        help="Temporal Attention Encoder for temporal encoding")

    parser.add_argument('--gru', dest='gru', action='store_true', help="Gated Recurent Unit for temporal encoding")
    parser.add_argument('--hidden_dim', default=156, type=int, help="Hidden state size")

    parser.add_argument('--tcnn', dest='tcnn', action='store_true', help="Temporal Convolutions for temporal encoding")
    parser.add_argument('--nker', default='[32,32,128]', type=str, help="Number of successive convolutional kernels ")

    parser.set_defaults(gru=False, tcnn=False, tae=False)

    config = parser.parse_args()
    config = vars(config)
    for k, v in config.items():
        if 'mlp' in k or k == 'nker':
            v = v.replace('[', '')
            v = v.replace(']', '')
            config[k] = list(map(int, v.split(',')))

    pprint.pprint(config)
    main(config)
