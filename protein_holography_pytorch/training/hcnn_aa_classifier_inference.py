
import os, sys
import gzip, pickle
import json
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from protein_holography_pytorch.so3.radial_spherical_tensor import MultiChannelRadialSphericalTensor

import torch
from torch.utils.data import Dataset, DataLoader
from torch import Tensor
import e3nn
from e3nn import o3
from sklearn.metrics import accuracy_score, confusion_matrix

from typing import *

from protein_holography_pytorch.models import CGNet, SO3_ConvNet
from protein_holography_pytorch.so3.functional import put_dict_on_device
from protein_holography_pytorch.cg_coefficients import get_w3j_coefficients
from protein_holography_pytorch.preprocessing import get_neighborhoods, get_structural_info, get_zernikegrams
from protein_holography_pytorch.utils.data import NeighborhoodsDataset
from protein_holography_pytorch.so3 import ZernickeRadialFunctions, RadialSphericalTensor, MultiChannelRadialSphericalTensor


def hcnn_aa_classifier_inference(experiment_dir: str,
                                 output_filepath: Optional[str] = None,
                                 data_filepath: Optional[str] = None,
                                 normalize_input_at_runtime: bool = False,
                                 model_name: str = 'lowest_valid_loss_model'):

    # get hparams from json
    with open(os.path.join(experiment_dir, 'hparams.json'), 'r') as f:
        hparams = json.load(f)

    # seed the random number generator
    rng = torch.Generator().manual_seed(hparams['seed'])

    # setup device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Running on %s.' % (device))


    ########## THE CODE BLOCK BELOW MAY BE CHANGED TO ACCOMODATE A DIFFERENT DATA-LOADING PIPELINE ##########

    if data_filepath is None: # run on default test data
        from protein_holography_pytorch.utils.data import load_data
        output_filepath = os.path.join(experiment_dir, 'test_data_results-{}.npz'.format(model_name))
        datasets, data_irreps, _ = load_data(hparams, splits=['test'])
        test_dataset = datasets['test']
        # test_dataloader = DataLoader(test_dataset, batch_size=hparams['batch_size'], generator=rng, shuffle=False, drop_last=False)

    elif data_filepath[-4:] == '.pdb':
        protein = get_structural_info(data_filepath)[0]
        nbs = get_neighborhoods(protein, remove_central_residue = True, backbone_only = False)
        zgrams_data = get_zernikegrams(nbs, hparams['rcut'], hparams['rmax'], hparams['lmax'], False, hparams['get_H'], hparams['get_SASA'], hparams['get_charge'], hparams['n_channels'], request_frame = False)
        if hparams['normalize_input']:
            normalize_input_at_runtime = True
        else:
            normalize_input_at_runtime = False
        
        OnRadialFunctions = ZernickeRadialFunctions(hparams['rcut'], hparams['rmax']+1, hparams['lmax'], complex_sph = False)
        rst = RadialSphericalTensor(hparams['rmax']+1, OnRadialFunctions, hparams['lmax'], 1, 1)
        mul_rst = MultiChannelRadialSphericalTensor(rst, hparams['n_channels'])
        data_irreps = o3.Irreps(str(mul_rst))

        def stringify(data_id):
            return '_'.join(list(map(lambda x: x.decode('utf-8'), list(data_id))))

        test_dataset = NeighborhoodsDataset(torch.tensor(zgrams_data['projections']), data_irreps, torch.tensor(zgrams_data['labels']), np.array(list(map(stringify, zgrams_data['data_ids']))))
        # test_dataloader = DataLoader(test_dataset, batch_size=hparams['batch_size'], generator=rng, shuffle=False, drop_last=False)

        if output_filepath is None:
            raise TypeError("'output_filepath' cannot be None if input is pdb file.")

    else:
        raise NotImplementedError()
    
    ########## THIS CODE BLOCK ABOVE MAY BE CHANGED TO ACCOMODATE A DIFFERENT DATA-LOADING PIPELINE ##########
    print('Data Irreps: %s' % (str(data_irreps)))
    sys.stdout.flush()


    # load w3j coefficients
    w3j_matrices = get_w3j_coefficients()
    for key in w3j_matrices:
        # if key[0] <= hparams['net_lmax'] and key[1] <= hparams['net_lmax'] and key[2] <= hparams['net_lmax']:
        if device is not None:
            w3j_matrices[key] = torch.tensor(w3j_matrices[key]).float().to(device)
        else:
            w3j_matrices[key] = torch.tensor(w3j_matrices[key]).float()
        w3j_matrices[key].requires_grad = False
    
    # create model and load weights
    if hparams['model_type'] == 'cgnet':
        model = CGNet(data_irreps, w3j_matrices, hparams['model_hparams'], normalize_input_at_runtime=normalize_input_at_runtime).to(device)
    elif hparams['model_type'] == 'so3_convnet':
        model = SO3_ConvNet(data_irreps, w3j_matrices, hparams['model_hparams'], normalize_input_at_runtime=normalize_input_at_runtime).to(device)
    else:
        raise NotImplementedError()
    
    model.load_state_dict(torch.load(os.path.join(experiment_dir, model_name + '.pt'), map_location=torch.device(device)))
    
    num_params = 0
    for param in model.parameters():
        num_params += torch.flatten(param.data).shape[0]
    print('There are %d parameters' % (num_params))
    sys.stdout.flush()

    predictions = model.predict(test_dataset, device=device)

    print(np.mean(predictions['logits']))
    print(np.std(predictions['logits']))
    print(np.max(predictions['logits']))
    print(np.min(predictions['logits']))

    print('Accuracy: %.3f' % (accuracy_score(predictions['targets'], predictions['best_indices'])))

    np.savez_compressed(output_filepath,
                        logits=predictions['logits'],
                        targets=predictions['targets'],
                        data_ids=predictions['data_ids'])

