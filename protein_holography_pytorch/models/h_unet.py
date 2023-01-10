
'''

NOT TESTED YET

Notes to self:
    Unlike H-VAE:
        1) irreps_out != irreps_in
        2) we put residual connections

'''

import sys, os
import numpy as np
import torch
import e3nn
from e3nn import o3
import nn
from utils import make_vec

from torch import Tensor
from typing import *

class H_VAE(torch.nn.Module):

    def load_hparams(self, hparams):
        # self.latent_dim = hparams['latent_dim'] # int --> unrelated
        self.net_lmax = hparams['net_lmax'] # int

        # the decoder is the same lists, but flipped!
        self.encoder_ch_size_list = hparams['encoder_ch_size_list'] # List[int]
        self.encoder_lmax_list = hparams['encoder_lmax_list'] # List[int]
        self.encoder_ls_nonlin_rule_list = hparams['encoder_ls_nonlin_rule_list'] # List[str]
        self.encoder_ch_nonlin_rule_list = hparams['encoder_ch_nonlin_rule_list'] # List[str]

        self.n_residual_blocks = hparams['n_residual_blocks']
        # self.residual_block_ch_size = hparams['residual_block_ch_size'] # int
        # self.residual_block_lmax = hparams['lmax_list'] # int
        # self.residual_block_ls_nonlin_rule = hparams['ls_nonlin_rule_list'] # str
        # self.residual_block_ch_nonlin_rule = hparams['ch_nonlin_rule_list'] # str

        self.do_initial_linear_projection = hparams['do_initial_linear_projection'] # bool
        self.ch_initial_linear_projection = hparams['ch_initial_linear_projection'] # int
        # self.weights_initializer = hparams['weights_initializer'] # Optional[str] --> just assume this is None, which will default to Kaiming init

        self.use_batch_norm = hparams['use_batch_norm'] # bool ; will be used at first

        self.norm_type = hparams['norm_type'] # Optional[str] ; None, layer, signal
        self.normalization = hparams['normalization'] # str ; norm, component
        # self.norm_balanced = hparams['norm_balanced'] # bool ; only for signal norm --> assume false
        self.norm_affine = hparams['norm_affine'] # Optional[Union[str, bool]] ; None, {True, False} -> for layer_norm, {unique, per_l, per_feature} -> for signal_norm
        self.norm_nonlinearity = hparams['norm_nonlinearity'] # Optional[str] ; None (identity), identity, relu, swish, sigmoid -> only for layer_norm
        self.norm_location = hparams['norm_location'] # str ; first, between, last
        # self.linearity_first = hparams['linearity_first'] # bool ; currently only works with this being false --> assume this to be false always
        
        # self.filter_symmetric = hparams['filter_symmetric'] # bool ; whether to exclude duplicate pairs of l's from the tensor product nonlinearity --> assume this to be true always, no reason why not
        # self.x_rec_loss_fn = hparams['x_rec_loss_fn'] # str ; mse, mse_normalized, cosine --> usually just mse. Keep it just in case we want to use mse_normalized. BUT put it in the training pipeline, not inside the model
        # self.do_final_signal_norm = hparams['do_final_signal_norm'] # bool --> unrelated
        # self.learn_frame = hparams['learn_frame'] # bool --> unrelated
        # self.is_vae = hparams['is_vae'] # bool --> unrelated

        assert len(self.encoder_ch_size_list) == len(self.encoder_lmax_list)
        assert len(self.encoder_ch_size_list) == len(self.encoder_ls_nonlin_rule_list)
        assert len(self.encoder_ch_size_list) == len(self.encoder_ch_nonlin_rule_list)
        self.n_encoder_cg_blocks = len(self.encoder_ch_size_list)
        self.net_lmax = max(self.encoder_ls_nonlin_rule_list)

    def __init__(self,
                 irreps_in: o3.Irreps,
                 irreps_out: o3.Irreps,
                 w3j_matrices: Dict[int, Tensor],
                 hparams: Dict,
                 device: str,
                 normalize_input_at_runtime: bool = False
                 ):
        super().__init__()

        self.irreps_in = irreps_in
        self.irreps_out = irreps_out
        self.load_hparams(hparams)
        self.device = device
        self.normalize_input_at_runtime = normalize_input_at_runtime


        # initial linear projection
        if self.do_initial_linear_projection:
            print(irreps_in.dim, irreps_in)
            initial_irreps = (self.ch_initial_linear_projection*o3.Irreps.spherical_harmonics(max(irreps_in.ls), 1)).sort().irreps.simplify()
            self.initial_linear_projection = nn.SO3_linearity(irreps_in, initial_irreps)
            print(initial_irreps.dim, initial_irreps)
        else:
            print(irreps_in.dim, irreps_in)
            initial_irreps = irreps_in
            channels = np.array([irr.mul for irr in initial_irreps])
            assert np.all(channels == channels[0])

        # encoder
        encoder_lmax_list = self.lmax_list
        encoder_ch_size_list = self.ch_size_list
        encoder_ls_nonlin_rule_list = self.ls_nonlin_rule_list
        encoder_ch_nonlin_rule_list = self.ch_nonlin_rule_list

        prev_irreps = initial_irreps
        encoder_cg_blocks = []
        for i in range(self.n_cg_blocks):
            temp_irreps_hidden = (encoder_ch_size_list[i]*o3.Irreps.spherical_harmonics(encoder_lmax_list[i], 1)).sort().irreps.simplify()
            encoder_cg_blocks.append(nn.CGBlock(prev_irreps,
                                                temp_irreps_hidden,
                                                w3j_matrices,
                                                linearity_first=False,
                                                filter_symmetric=True,
                                                use_batch_norm=self.use_batch_norm,
                                                ls_nonlin_rule=encoder_ls_nonlin_rule_list[i], # full, elementwise, efficient
                                                ch_nonlin_rule=encoder_ch_nonlin_rule_list[i], # full, elementwise
                                                norm_type=self.norm_type, # None, layer, signal
                                                normalization=self.normalization, # norm, component -> only if norm_type is not none
                                                norm_balanced=False,
                                                norm_affine=self.norm_affine, # None, {True, False} -> for layer_norm, {unique, per_l, per_feature} -> for signal_norm
                                                norm_nonlinearity=self.norm_nonlinearity, # None (identity), identity, relu, swish, sigmoid -> only for layer_norm
                                                norm_location=self.norm_location, # first, between, last
                                                weights_initializer=None,
                                                init_scale=1.0))


            prev_irreps = encoder_cg_blocks[-1].irreps_out
            print(prev_irreps.dim, prev_irreps)

        self.encoder_cg_blocks = torch.nn.ModuleList(encoder_cg_blocks)


        # residual blocks
        for i in range(self.n_residual_blocks):


        # decoder
        decoder_lmax_list = self.lmax_list[::-1][1:] + [max(initial_irreps.ls)]
        decoder_ch_size_list = self.ch_size_list[::-1][1:] + [[irr.mul for irr in initial_irreps][0]] # I put the 0th multiplicity, but all irreps are the same, really


    def forward(self, x: Dict[int, Tensor]) -> Dict[int, Tensor]:

        # initial linear projection


        # encoder


        # residual blocks


        # decoder

        pass