
import os
import numpy as np
import torch
from e3nn import o3
from .dataset import NeighborhoodsDataset
from protein_holography_pytorch.so3 import ZernickeRadialFunctions, RadialSphericalTensor, MultiChannelRadialSphericalTensor

from typing import *
from torch import Tensor


def get_norm_factor(projections: Tensor, data_irreps: o3.Irreps):
    ls_indices = torch.cat([torch.tensor(data_irreps.ls)[torch.tensor(data_irreps.ls) == l].repeat(2*l+1) for l in sorted(list(set(data_irreps.ls)))]).type(torch.float)
    batch_size = 2000
    norm_factors = []
    num_batches = projections.shape[0] // batch_size
    for i in range(num_batches):
        signals = projections[i*batch_size : (i+1)*batch_size]
        batch_norm_factors = torch.sqrt(torch.einsum('bf,bf,f->b', signals, signals, 1.0 / (2*ls_indices + 1)))
        norm_factors.append(batch_norm_factors)
    
    # final batch for the remaining signals
    if (projections.shape[0] % batch_size) > 0:
        signals = projections[(i+1)*batch_size:]
        batch_norm_factors = torch.sqrt(torch.einsum('bf,bf,f->b', signals, signals, 1.0 / (2*ls_indices + 1)))
        norm_factors.append(batch_norm_factors)

    norm_factor = torch.mean(torch.cat(norm_factors, dim=-1)).item()

    return norm_factor


def load_data(hparams, splits=['train', 'valid'], test_data_filepath=None):
    '''
    Returns norm_factor if training split is requested
    '''

    for split in splits:
        assert split in {'train', 'valid', 'test'}

    if 'no_sidechain' in hparams['neigh_kind']:
        data_args = ['rmax', 'lmax', 'n_channels', 'rcut', 'rst_normalization']
    else:
        data_args = ['rmax', 'lmax', 'n_channels', 'rcut', 'rst_normalization', 'get_H', 'get_SASA', 'get_charge']
    
    OnRadialFunctions = ZernickeRadialFunctions(hparams['rcut'], hparams['rmax']+1, hparams['lmax'], complex_sph = False)
    rst = RadialSphericalTensor(hparams['rmax']+1, OnRadialFunctions, hparams['lmax'], 1, 1)
    mul_rst = MultiChannelRadialSphericalTensor(rst, hparams['n_channels'])
    data_irreps = o3.Irreps(str(mul_rst))

    def stringify(data_id):
        return '_'.join(list(map(lambda x: x.decode('utf-8'), list(data_id))))
    
    norm_factor = None
    datasets = {}
    if 'train' in splits:
        train_data_id = '-'.join(list(sorted(['%s=%s' % (arg, hparams[arg]) for arg in data_args] + ['n_neigh=%s' % (hparams['n_train_neigh'])])))
        train_arrays = np.load(os.path.join(hparams['data_dir'], hparams['neigh_kind'], 'all_arrays-train-complex_sph=False-{}.npz'.format(train_data_id)))
        train_data = torch.tensor(train_arrays['projections'])

        if hparams['normalize_input']:
            norm_factor = get_norm_factor(train_data, data_irreps)
            train_data = train_data / norm_factor

        train_labels = torch.tensor(train_arrays['labels'])
        train_ids = train_arrays['data_ids']
        train_ids = np.array(list(map(stringify, train_ids)))
        train_dataset = NeighborhoodsDataset(train_data, data_irreps, train_labels, train_ids)
        datasets['train'] = train_dataset
    
    if 'valid' in splits:
        valid_data_id = '-'.join(list(sorted(['%s=%s' % (arg, hparams[arg]) for arg in data_args] + ['n_neigh=%s' % (hparams['n_valid_neigh'])])))
        valid_arrays = np.load(os.path.join(hparams['data_dir'], hparams['neigh_kind'], 'all_arrays-val-complex_sph=False-{}.npz'.format(valid_data_id)))
        valid_data = torch.tensor(valid_arrays['projections'])

        if hparams['normalize_input'] and norm_factor is not None:
            valid_data = valid_data / norm_factor

        valid_labels = torch.tensor(valid_arrays['labels'])
        valid_ids = valid_arrays['data_ids']
        valid_ids = np.array(list(map(stringify, valid_ids)))
        valid_dataset = NeighborhoodsDataset(valid_data, data_irreps, valid_labels, valid_ids)
        datasets['valid'] = valid_dataset
    
    if 'test' in splits:
        if test_data_filepath is not None: # if you have new test data to evaluate on! Only requrement is that the field 'projections{}'.format(normalize_str) must be present
            test_arrays = np.load(test_data_filepath)
        else:
            test_data_id = '-'.join(list(sorted(['%s=%s' % (arg, hparams[arg]) for arg in data_args] + ['n_neigh=%s' % (hparams['n_test_neigh'])])))
            test_arrays = np.load(os.path.join(hparams['data_dir'], hparams['neigh_kind'], 'all_arrays-test-complex_sph=False-{}.npz'.format(test_data_id)))

        test_data = torch.tensor(test_arrays['projections'])

        if hparams['normalize_input'] and norm_factor is not None:
            test_data = test_data / norm_factor

        test_labels = torch.tensor(test_arrays['labels']) if 'labels' in test_arrays else torch.tensor(np.full(test_data.shape[0], np.nan))
        test_ids = np.array(list(map(stringify, test_arrays['data_ids']))) if 'data_ids' in test_arrays else torch.tensor(np.full(test_data.shape[0], np.nan))
        test_dataset = NeighborhoodsDataset(test_data, data_irreps, test_labels, test_ids)
        datasets['test'] = test_dataset

    return datasets, data_irreps, norm_factor
