
import os, sys
import gzip, pickle
import json
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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


def hcnn_aa_classifier_inference(experiment_dir: str,
                                 output_file: str = 'test_data_results.npz',
                                 data_file: Optional[str] = None,
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
    # get data and make dataloaders
    if data_file is None: # run on test data
        from protein_holography_pytorch.utils.data import load_data
        datasets, data_irreps, _ = load_data(hparams, splits=['test'])
        test_dataset = datasets['test']
        test_dataloader = DataLoader(test_dataset, batch_size=hparams['batch_size'], generator=rng, shuffle=False, drop_last=False)
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

    # inference loop!
    y_hat_all_logits = []
    y_hat_all_index = []
    y_all = []
    data_ids_all = []
    for i, (X, X_vec, y, data_ids) in enumerate(test_dataloader):
        X = put_dict_on_device(X, device)
        y = y.to(device)
        model.eval()
        
        y_hat = model(X)
        y_hat_all_logits.append(y_hat.detach().cpu().numpy())
        y_hat_all_index.append(np.argmax(y_hat.detach().cpu().numpy(), axis=1))
        y_all.append(y.detach().cpu().numpy())
        data_ids_all.append(data_ids)

    y_hat_all_logits = np.vstack(y_hat_all_logits)
    y_hat_all_index = np.hstack(y_hat_all_index)
    y_all = np.hstack(y_all)
    data_ids_all = np.hstack(data_ids_all)

    print('Accuracy: %.3f' % (accuracy_score(y_all, y_hat_all_index)))

    np.savez_compressed(os.path.join(experiment_dir, output_file),
                        logits=y_hat_all_logits,
                        y_true=y_all,
                        data_ids=data_ids_all)

