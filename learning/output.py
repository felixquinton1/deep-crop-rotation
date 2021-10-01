import numpy as np
import os
import json
import pickle as pkl
from learning.metrics import confusion_matrix_analysis


def prepare_output(config):
    os.makedirs(config['res_dir'], exist_ok=True)

    for fold in range(1, config['kfold'] + 1):
        for year in config['year']:
            os.makedirs(os.path.join(config['res_dir'], 'Fold_{}'.format(fold), year), exist_ok=True)
        os.makedirs(os.path.join(config['res_dir'], 'Fold_{}'.format(fold)), exist_ok=True)

    if config['save_pred']:
        for year in config['year']:
            os.makedirs(os.path.join(config['save_pred_dir'], year), exist_ok=True)

    os.makedirs(os.path.join(config['res_dir'], 'overall'), exist_ok=True)


def checkpoint(fold, log, config):
    with open(os.path.join(config['res_dir'], 'Fold_{}'.format(fold), 'trainlog.json'), 'w') as outfile:
        json.dump(log, outfile, indent=4)


def save_pred(pred, key, config):
    np.save(os.path.join(config['save_pred_dir'], key[-4:], key[:-5]), pred)

def save_results(metrics, conf_mat, config, year, fold):
    with open(os.path.join(config['res_dir'], 'Fold_{}'.format(fold), year, 'test_metrics.json'), 'w') \
            as outfile:
        json.dump(metrics, outfile, indent=4)
    pkl.dump(conf_mat, open(os.path.join(config['res_dir'], 'Fold_{}'.format(fold), year, 'conf_mat.pkl'
                                         ), 'wb'))


def overall_performance_by_year(config, year):
    cm = np.zeros((config['num_classes'], config['num_classes']))
    for fold in range(1, config['kfold'] + 1):
        if not config['test_mode']:
            cm += pkl.load(open(os.path.join(config['res_dir'], 'Fold_{}'.format(fold), year, 'conf_mat.pkl'), 'rb'))
        else:
            cm += pkl.load(open(os.path.join(config['res_dir'], 'Fold_{}'.format(fold), year, 'conf_mat.pkl'), 'rb'))

    _, perf = confusion_matrix_analysis(cm)

    print('Overall performance in:' + year)
    print('Acc: {},  IoU: {}'.format(perf['Accuracy'], perf['MACRO_IoU']))

    pkl.dump(cm.astype(int), open(os.path.join(config['res_dir'], 'overall', year + '_conf_mat.pkl'), 'wb'))
    with open(os.path.join(config['res_dir'], 'overall', year + '_overall.json'), 'w') as file:
        file.write(json.dumps(perf, indent=4))


def overall_performance(config):
    cm = np.zeros((config['num_classes'], config['num_classes']))
    for year in config['year']:
        cm += pkl.load(open(os.path.join(config['res_dir'], 'overall', year + '_conf_mat.pkl'), 'rb'))

    _, perf = confusion_matrix_analysis(cm)

    print('Overall performance:')
    print('Acc: {},  IoU: {}'.format(perf['Accuracy'], perf['MACRO_IoU']))

    pkl.dump(cm.astype(int), open(os.path.join(config['res_dir'], 'overall', 'conf_mat.pkl'), 'wb'))
    with open(os.path.join(config['res_dir'], 'overall', 'overall.json'), 'w') as file:
        file.write(json.dumps(perf, indent=4))
