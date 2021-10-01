import numpy as np
import torch
from sklearn.model_selection import KFold
import torch.utils.data as data
from torch.nn import functional as F
import collections.abc
import re


def get_loaders(dt, kfold, config):
    indices = [np.array(range(i * len(dt[i]), (i + 1) * len(dt[i]))) for i in range(len(dt))]
    np.random.shuffle(indices[0])
    if len(indices) > 1:
        for i in range(1, len(indices)):
            indices[i] = np.array(indices[0]) + len(indices[0]) * i

    kf = KFold(n_splits=kfold, shuffle=False)
    indices_seq = [list(kf.split(list(range(len(i))))) for i in dt]
    ntest = len(indices_seq[0][0][1])

    loader_seq = []
    test_loader = [[] for i in range(kfold)]
    validation_indices = [[] for i in range(kfold)]
    train_indices = [[] for i in range(kfold)] #change
    merged_dt = dt[0]

    for i in range(1, len(dt)):
        merged_dt.date_positions.update(dt[i].date_positions)
        merged_dt.pid += dt[i].pid
        merged_dt.target += dt[i].target

    for idx, dataset in enumerate(dt):
        for id_fold, (trainval, test_indices) in enumerate(indices_seq[idx]):
            test_indices = [indices[idx][i] for i in test_indices]
            trainval = [indices[idx][i] for i in trainval]
            test_sampler = data.sampler.SubsetRandomSampler(test_indices)

            test_loader[id_fold].append(data.DataLoader(merged_dt, batch_size=config['batch_size'],
                                                        sampler=test_sampler,
                                                        num_workers=config['num_workers'],
                                                        collate_fn=pad_collate))
            validation_indices[id_fold] += trainval[-ntest:]

            if not config['test_mode']:
                train_indices[id_fold] += trainval[:-ntest]

    for i in range(kfold):
        validation_sampler = data.sampler.SubsetRandomSampler(validation_indices[i])
        validation_loader = data.DataLoader(merged_dt, batch_size=config['batch_size'],
                                            sampler=validation_sampler,
                                            num_workers=config['num_workers'], collate_fn=pad_collate)
        if config['test_mode']:
            loader_seq.append((validation_loader, test_loader[i]))
        else:
            train_sampler = data.sampler.SubsetRandomSampler(train_indices[i])
            train_loader = data.DataLoader(merged_dt, batch_size=config['batch_size'],
                                           sampler=train_sampler,
                                           num_workers=config['num_workers'], collate_fn=pad_collate)
            loader_seq.append((train_loader, validation_loader, test_loader[i]))

    return loader_seq, merged_dt


np_str_obj_array_pattern = re.compile(r'[SaUO]')

def pad_tensor(x, l, pad_value=0):
    padlen = l - x.shape[0]
    pad = [0 for _ in range(2 * len(x.shape[1:]))] + [0, padlen]
    return F.pad(x, pad=pad, value=pad_value)


def pad_collate(batch):
    # modified default_collate from the official pytorch repo
    # https://github.com/pytorch/pytorch/blob/master/torch/utils/data/_utils/collate.py
    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        if len(elem.shape) > 0:
            sizes = [e.shape[0] for e in batch]
            m = max(sizes)
            batch = [pad_tensor(e, m) for e in batch]

        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        return torch.stack(batch, 0, out=out)
    elif elem_type.__name__ == 'str':
        return batch
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        if elem_type.__name__ == 'ndarray' or elem_type.__name__ == 'memmap':
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError('Format not managed : {}'.format(elem.dtype))
            return pad_collate([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, collections.abc.Mapping):
        return {key: pad_collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(pad_collate(samples) for samples in zip(*batch)))
    elif isinstance(elem, collections.abc.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError('each element in list of batch should be of equal size')
        transposed = zip(*batch)
        return [pad_collate(samples) for samples in transposed]

    raise TypeError('Format not managed : {}'.format(elem_type))
