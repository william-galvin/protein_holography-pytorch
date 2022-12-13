
import sys, os

import numpy as np
import torch
import e3nn
from e3nn import o3
from protein_holography_pytorch import nn

from torch import Tensor
from typing import *


class SO3_ConvNet(torch.nn.Module):
    '''
    CGNet-like model, but without the invariant skip connections
    '''

    def load_hparams(self, hparams: Dict):
        self.output_dim = hparams['output_dim']
        self.n_cg_blocks = hparams['n_cg_blocks']
        self.n_fc_blocks = hparams['n_fc_blocks']
        self.fc_h_dim = hparams['fc_h_dim']
        self.fc_nonlin = hparams['fc_nonlin']
        self.dropout_rate = hparams['dropout_rate']
        self.ch_size_list = hparams['ch_size_list']
        self.ls_nonlin_rule_list = hparams['ls_nonlin_rule_list']
        self.ch_nonlin_rule_list = hparams['ch_nonlin_rule_list']
        self.do_initial_linear_projection = hparams['do_initial_linear_projection']
        self.ch_initial_linear_projection = hparams['ch_initial_linear_projection']
        self.lmax_list = hparams['lmax_list']
        self.use_additive_skip_connections = hparams['use_additive_skip_connections']
        self.use_batch_norm = hparams['use_batch_norm']
        self.weights_initializer = hparams['weights_initializer']
        self.norm_type = hparams['norm_type']
        self.normalization = hparams['normalization']
        self.norm_balanced = hparams['norm_balanced']
        self.norm_affine = hparams['norm_affine']
        self.norm_nonlinearity = hparams['norm_nonlinearity']
        self.norm_location = hparams['norm_location']
        self.linearity_first = hparams['linearity_first']
        self.filter_symmetric = hparams['filter_symmetric']
        self.input_normalizing_constant = torch.Tensor(hparams['input_normalizing_constant'], requires_grad=False) if hparams['input_normalizing_constant'] is not None else None

        assert self.n_cg_blocks == len(self.ch_size_list)
        assert self.n_cg_blocks == len(self.ls_nonlin_rule_list)
        assert self.n_cg_blocks == len(self.ch_nonlin_rule_list)
        assert self.n_cg_blocks == len(self.lmax_list)


    def __init__(self,
                 irreps_in: o3.Irreps,
                 w3j_matrices: Dict[int, Tensor],
                 hparams: Dict,
                 normalize_input_at_runtime: bool = False
                 ):
        super().__init__()

        self.irreps_in = irreps_in
        self.load_hparams(hparams)
        self.normalize_input_at_runtime = normalize_input_at_runtime

        assert self.n_cg_blocks == len(self.ch_size_list)
        assert self.lmax_list is None or self.n_cg_blocks == len(self.lmax_list)
        assert self.n_cg_blocks == len(self.ls_nonlin_rule_list)
        assert self.n_cg_blocks == len(self.ch_nonlin_rule_list)

        if self.do_initial_linear_projection:
            print(self.irreps_in.dim, self.irreps_in)
            initial_irreps = (self.ch_initial_linear_projection*o3.Irreps.spherical_harmonics(max(self.irreps_in.ls), 1)).sort().irreps.simplify()
            self.initial_linear_projection = nn.linearity(self.irreps_in, initial_irreps)
            print(initial_irreps.dim, initial_irreps)
        else:
            print(self.irreps_in.dim, self.irreps_in)
            initial_irreps = self.irreps_in


        # equivariant, cg blocks
        prev_irreps = initial_irreps
        cg_blocks = []
        for i in range(self.n_cg_blocks):
            irreps_hidden = (self.ch_size_list[i]*o3.Irreps.spherical_harmonics(self.lmax_list[i], 1)).sort().irreps.simplify()
            cg_blocks.append(nn.CGBlock(prev_irreps,
                                                irreps_hidden,
                                                w3j_matrices,
                                                linearity_first=self.linearity_first,
                                                filter_symmetric=self.filter_symmetric,
                                                use_batch_norm=self.use_batch_norm,
                                                ls_nonlin_rule=self.ls_nonlin_rule_list[i], # full, elementwise, efficient
                                                ch_nonlin_rule=self.ch_nonlin_rule_list[i], # full, elementwise
                                                norm_type=self.norm_type, # None, layer, signal
                                                normalization=self.normalization, # norm, component -> only if norm_type is not none
                                                norm_balanced=self.norm_balanced,
                                                norm_affine=self.norm_affine, # None, {True, False} -> for layer_norm, {unique, per_l, per_feature} -> for signal_norm
                                                norm_nonlinearity=self.norm_nonlinearity, # None (identity), identity, relu, swish, sigmoid -> only for layer_norm
                                                norm_location=self.norm_location, # first, between, last
                                                weights_initializer=self.weights_initializer,
                                                init_scale=1.0))

            prev_irreps = cg_blocks[-1].irreps_out
            print(prev_irreps.dim, prev_irreps)

        self.cg_blocks = torch.nn.ModuleList(cg_blocks)

        invariants_dim = [mul for (mul, _) in prev_irreps][0] # number of channels for l = 0


        # invariant, fully connected blocks
        prev_dim = invariants_dim
        fc_blocks = []
        for _ in range(self.n_fc_blocks):
            block = []
            block.append(torch.nn.Linear(prev_dim, self.fc_h_dim))
            block.append(eval(nn.NONLIN_TO_ACTIVATION_MODULES[self.fc_nonlin]))
            if self.dropout_rate > 0.0:
                block.append(torch.nn.Dropout(self.dropout_rate))

            fc_blocks.append(torch.nn.Sequential(*block))
            prev_dim = self.fc_h_dim

        if len(fc_blocks) > 0:
            self.fc_blocks = torch.nn.ModuleList(fc_blocks)
        else:
            self.fc_blocks = None


        # output head
        self.output_head = torch.nn.Linear(prev_dim, self.output_dim)

    
    def forward(self, x: Dict[int, Tensor]) -> Tensor:

        # normalize input data if desired
        if self.normalize_input_at_runtime and self.input_normalizing_constant is not None:
            for l in x:
                x[l] = x[l] / self.input_normalizing_constant

        if self.do_initial_linear_projection:
            h = self.initial_linear_projection(x)
        else:
            h = x
        
        # equivariant, cg blocks
        for i, block in enumerate(self.cg_blocks):
            h_temp = block(h)
            if self.use_additive_skip_connections:
                for l in h:
                    if l in h_temp:
                        if h[l].shape[1] == h_temp[l].shape[1]: # the shape at index 1 is the channels' dimension
                            h_temp[l] += h[l]
                        elif h[l].shape[1] > h_temp[l].shape[1]:
                            h_temp[l] += h[l][:, : h_temp[l].shape[1], :] # subsample first channels
                        else: # h[l].shape[1] < h_temp[l].shape[1]
                            h_temp[l] += torch.nn.functional.pad(h[l], (0, 0, 0, h_temp[l].shape[1] - h[l].shape[1])) # zero pad the channels' dimension
            h = h_temp
        
        invariants = h[0].squeeze(-1)


        # invariant, fully connected blocks
        h = invariants
        if self.fc_blocks is not None:
            for block in self.fc_blocks:
                h = block(h)
                # h += block(h) # skip connections

        

        # output head
        out = self.output_head(h)

        return out


    def get_inv_embedding(self, x: Dict[int, Tensor], emb_i: Union[int, str]) -> Tensor:
        '''
        Gets invariant embedding from the FC blocks (backwards, must be negative), or from the input to the FC blocks
        '''
        assert emb_i in ['cg_output', 'all'] or emb_i in [-i for i in range(1, self.n_fc_blocks + 1)]
        self.eval()

        all_output = []

        # equivariant, cg blocks
        for i, block in enumerate(self.cg_blocks):
            h_temp = block(h)
            if self.use_additive_skip_connections:
                for l in h:
                    if l in h_temp:
                        if h[l].shape[1] == h_temp[l].shape[1]: # the shape at index 1 is the channels' dimension
                            h_temp[l] += h[l]
                        elif h[l].shape[1] > h_temp[l].shape[1]:
                            h_temp[l] += h[l][:, : h_temp[l].shape[1], :] # subsample first channels
                        else: # h[l].shape[1] < h_temp[l].shape[1]
                            h_temp[l] += torch.nn.functional.pad(h[l], (0, 0, 0, h_temp[l].shape[1] - h[l].shape[1])) # zero pad the channels' dimension
            h = h_temp
        
        invariants = h[0].squeeze(-1)


        if emb_i == 'cg_output':
            return invariants
        elif emb_i == 'all':
            all_output.append(invariants)
        
        h = invariants
        if self.fc_blocks is not None:
            for n, block in enumerate(self.fc_blocks):
                h = block(h)
                if emb_i == 'all':
                    all_output.append(h)
                elif n == len(self.fc_blocks) + emb_i:
                    return h
        
        # NB: only runs if emb_i == 'all'
        return all_output
