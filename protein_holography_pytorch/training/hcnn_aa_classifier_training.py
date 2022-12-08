'''

General classification training loop using CGNet-derived architectures

Assumes dataset giving data and labels are given
Takes as input a directory, which contains a json file with the model hyperparameters,
and in which the model will save checkpoints and all the good stuff

'''

import os, sys
import gzip, pickle
import json
import time
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader
from torch import Tensor
import e3nn
from e3nn import o3
from sklearn.metrics import accuracy_score

from cgnet_utils import *

from typing import *

from ..models import CGNet, SO3_ConvNet


def train_classifier(experiment_dir, w3j_matrices_path='../cg_coefficients/w3j_matrices-lmax=14-version=0.5.0.pkl'):

    # get hparams from json`
    with open(os.path.join(experiment_dir, 'hparams.json'), 'r') as f:
        hparams = json.load(f)

    # seed the random number generator
    rng = torch.Generator().manual_seed(hparams['seed'])

    # setup device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Running on %s.' % (device))
    sys.stdout.flush()

    # get data and make dataloaders
    print('Loading data...')
    sys.stdout.flush()
    DATA_ARGS = ['rmax', 'lmax', 'n_channels', 'rcut', 'rst_normalization', 'get_H', 'get_SASA', 'get_charge']
    datasets, data_irreps = load_data(hparams, DATA_ARGS, splits=['train', 'valid'])
    train_dataset, valid_dataset = datasets['train'], datasets['valid']
    train_dataloader = DataLoader(train_dataset, batch_size=hparams['batch_size'], generator=rng, shuffle=True, drop_last=False)
    valid_dataloader = DataLoader(valid_dataset, batch_size=hparams['batch_size'], generator=rng, shuffle=True, drop_last=False)
    print('Data Irreps: %s' % (str(data_irreps)))
    sys.stdout.flush()

    # load w3j coefficients
    with gzip.open(w3j_matrices_path, 'rb') as f:
        w3j_matrices = pickle.load(f)
    
    for key in w3j_matrices:
        if key[0] <= hparams['net_lmax'] and key[1] <= hparams['net_lmax'] and key[2] <= hparams['net_lmax']:
            if device is not None:
                w3j_matrices[key] = torch.tensor(w3j_matrices[key]).float().to(device)
            else:
                w3j_matrices[key] = torch.tensor(w3j_matrices[key]).float()
            w3j_matrices[key].requires_grad = False
    
    # create model
    if hparams['model'] == 'cgnet':
        model = CGNet(data_irreps, w3j_matrices, hparams).to(device)
    elif hparams['model'] == 'so3_convnet':
        model = SO3_ConvNet(data_irreps, w3j_matrices, hparams).to(device)
    else:
        raise NotImplementedError()
    
    num_params = 0
    for param in model.parameters():
        num_params += torch.flatten(param.data).shape[0]
    print('There are %d parameters' % (num_params))
    sys.stdout.flush()

    # setup learning algorithm
    optimizer = torch.optim.Adam(model.parameters(), lr=hparams['lr'])

    if hparams['lr_scheduler'] is None:
        lr_scheduler = None
    elif hparams['lr_scheduler'] == 'reduce_lr_on_plateau':
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08, verbose=False)
    else:
        raise NotImplementedError()

    # setup loss function
    loss_fn = torch.nn.CrossEntropyLoss()

    # training loop!
    n_steps_until_report = 1000
    n_times_to_report = len(train_dataloader) // n_steps_until_report
    if len(train_dataloader) % n_steps_until_report != 0:
        n_times_to_report += 1
    train_loss_trace = []
    valid_loss_trace = []
    valid_acc_trace = []
    lowest_valid_loss = np.inf
    highest_valid_acc = 0.0
    for epoch in range(hparams['n_epochs']):
        print('Epoch %d/%d' % (epoch+1, hparams['n_epochs']))
        sys.stdout.flush()
        temp_train_loss_trace = []
        n_steps = 0
        reported_times = 0
        start_time = time.time()
        for train_i, (X_train, y_train, data_ids) in enumerate(train_dataloader):
            X_train = dict_to_device(X_train, device)
            y_train = y_train.to(device)
            model.train()

            optimizer.zero_grad()
            
            y_train_hat = model(X_train)
            loss_train = loss_fn(y_train_hat, y_train)
            temp_train_loss_trace.append(loss_train.item())
            
            loss_train.backward()
            optimizer.step()

            n_steps += 1

            # record train and validation loss
            if n_steps == n_steps_until_report or train_i == len(train_dataloader)-1:
                reported_times += 1
                temp_valid_loss_trace = []
                y_valid_hat_all = []
                y_valid_all = []
                for valid_i, (X_valid, y_valid, data_ids) in enumerate(valid_dataloader):
                    X_valid = dict_to_device(X_valid, device)
                    y_valid = y_valid.to(device)
                    model.eval()
                    
                    y_valid_hat = model(X_valid)
                    loss_valid = loss_fn(y_valid_hat, y_valid)
                    temp_valid_loss_trace.append(loss_valid.item())
                    y_valid_all.append(y_valid.detach().cpu().numpy())
                    y_valid_hat_all.append(np.argmax(y_valid_hat.detach().cpu().numpy(), axis=1))
                
                curr_train_loss = np.mean(temp_train_loss_trace)
                curr_valid_loss = np.mean(temp_valid_loss_trace)
                curr_valid_acc = accuracy_score(np.hstack(y_valid_all), np.hstack(y_valid_hat_all))

                end_time = time.time()
                print('%d/%d:\tTrain loss %.5f - Valid loss %.5f - Valid acc %.3f -- Time (s): %.1f' % (reported_times, n_times_to_report, curr_train_loss, curr_valid_loss, curr_valid_acc, (end_time - start_time)))
                sys.stdout.flush()
                
                # update lr with scheduler
                if lr_scheduler is not None:
                    lr_scheduler.step(curr_valid_loss)

                # record best model so far
                if curr_valid_loss < lowest_valid_loss:
                    lowest_valid_loss = curr_valid_loss
                    torch.save(model.state_dict(), os.path.join(experiment_dir, 'lowest_valid_loss_model.pt'))

                # record best model so far
                if curr_valid_acc > highest_valid_acc:
                    highest_valid_acc = curr_valid_acc
                    torch.save(model.state_dict(), os.path.join(experiment_dir, 'highest_valid_acc_model.pt'))

                train_loss_trace.append(curr_train_loss)
                valid_loss_trace.append(curr_valid_loss)
                valid_acc_trace.append(curr_valid_acc)

                n_steps = 0
                temp_train_loss_trace = []
                temp_valid_loss_trace = []
                start_time = time.time()

    
    # save last model
    torch.save(model.state_dict(), os.path.join(experiment_dir, 'final_model.pt'))

    # save loss traces, both as arrays and as plots
    if not os.path.exists(os.path.join(experiment_dir, 'loss_traces')):
        os.mkdir(os.path.join(experiment_dir, 'loss_traces'))

    np.save(os.path.join(experiment_dir, 'loss_traces', 'train_loss_trace.npy'), train_loss_trace)
    np.save(os.path.join(experiment_dir, 'loss_traces', 'valid_loss_trace.npy'), valid_loss_trace)
    np.save(os.path.join(experiment_dir, 'loss_traces', 'valid_acc_trace.npy'), valid_acc_trace)

    iterations = np.arange(len(train_loss_trace))

    plt.figure(figsize=(10, 4))
    plt.plot(iterations, train_loss_trace, label='train')
    plt.plot(iterations, valid_loss_trace, label='valid')
    plt.ylabel('Cross-Entropy loss')
    plt.xlabel('Evaluation iterations (%d epochs)' % (hparams['n_epochs']))
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(experiment_dir, 'loss_trace.png'))

    plt.figure(figsize=(10, 4))
    plt.plot(iterations, valid_acc_trace, label='valid')
    plt.ylabel('Accuracy')
    plt.xlabel('Evaluation iterations (%d epochs)' % (hparams['n_epochs']))
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(experiment_dir, 'acc_trace.png'))
