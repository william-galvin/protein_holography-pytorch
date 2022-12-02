
import numpy as np
import torch
from torch import nn
import e3nn
from e3nn import o3

from .linearity import linearity
from .nonlinearity import nonlinearity
from .normalization import signal_norm
from .blocks import CGNetBlock, FFNN_block
from loss_functions import *

from torch import Tensor
from typing import Optional, List, Dict

import sys, os

# MAX_FLOAT32 = torch.tensor(3e30).type(torch.float32)

'''
Allow to decide, for each cg block:
- number of channels (constant across ls)
- ls_nonlin_rule
- ch_nonlin_rule

Use the same kind(s) of normalization(s) for all blocks
'''

class ClebschGordanVAE_symmetric_simple_flexible(torch.nn.Module):
    def __init__(self,
                 irreps_in: o3.Irreps,
                 latent_dim: int,
                 net_lmax: int,
                 n_cg_blocks: int,
                 ch_size_list: List[int],
                 ls_nonlin_rule_list: List[str], # as they should appear in the encoder
                 ch_nonlin_rule_list: List[str], # as they should appear in the encoder
                 do_initial_linear_projection: bool,
                 ch_initial_linear_projection: int,
                 w3j_matrices: Dict[int, Tensor],
                 device: str,
                 bottleneck_hidden_dims: Optional[List[int]] = None,
                 lmax_list: Optional[List[int]] = None, # as they should appear in the encoder
                 use_additive_skip_connections: bool = False,
                 use_batch_norm: bool = True,
                 weights_initializer: Optional[str] = None,
                 norm_type: str = 'signal', # None, layer, signal
                 normalization: str = 'component', # norm, component -> only considered if norm_type is not none
                 norm_balanced: bool = False, # only for signal norm
                 norm_affine: Optional[Union[str, bool]] = 'per_l', # None, {True, False} -> for layer_norm, {unique, per_l, per_feature} -> for signal_norm
                 norm_nonlinearity: str = None, # None (identity), identity, relu, swish, sigmoid -> only for layer_norm
                 norm_location: str = 'between', # first, between, last
                 linearity_first: bool = False, # currently only works with this being false
                 filter_symmetric: bool = True, # whether to exclude duplicate pairs of l's from the tensor product nonlinearity
                 x_rec_loss_fn: str = 'mse', # mse, mse_normalized, cosine
                 do_final_signal_norm: bool = False,
                 learn_frame: bool = True,
                 is_vae: bool = True):
        super().__init__()

        self.irreps_in = irreps_in
        self.do_initial_linear_projection = do_initial_linear_projection
        self.device = device
        self.use_additive_skip_connections = use_additive_skip_connections
        self.linearity_first = linearity_first
        self.use_batch_norm = use_batch_norm
        self.x_rec_loss_fn = x_rec_loss_fn
        self.latent_dim = latent_dim
        self.do_final_signal_norm = do_final_signal_norm
        self.is_vae = is_vae

        assert n_cg_blocks == len(ch_size_list)
        assert lmax_list is None or n_cg_blocks == len(lmax_list)
        assert n_cg_blocks == len(ls_nonlin_rule_list)
        assert n_cg_blocks == len(ch_nonlin_rule_list)

        if self.do_initial_linear_projection:
            print(irreps_in.dim, irreps_in)
            initial_irreps = (ch_initial_linear_projection*o3.Irreps.spherical_harmonics(max(irreps_in.ls), 1)).sort().irreps.simplify()
            self.initial_linear_projection = linearity(irreps_in, initial_irreps)
            print(initial_irreps.dim, initial_irreps)
        else:
            print(irreps_in.dim, irreps_in)
            initial_irreps = irreps_in
        
        # prepare lmaxs for both encoder and decoder blocks
        lmaxs = [min(2**i, net_lmax) for i in range(n_cg_blocks)] + [max(initial_irreps.ls)]
        lmaxs_encoder_upper_bound = lmaxs[:-1][::-1] # exclude data irreps and reverse so it's decreasing
        lmaxs_decoder_upper_bound = lmaxs[1:] # exclude latent space irreps
        if lmax_list is not None:
            lmaxs_encoder = lmax_list
            lmaxs_decoder = lmax_list[::-1][1:] + [max(initial_irreps.ls)]

            # the provided lmaxs must be such that they are not above the maximum
            # allowed by the bottleneck that goes down to lmax=1
            assert np.all(np.array(lmaxs_encoder_upper_bound) >= np.array(lmaxs_encoder))
            assert np.all(np.array(lmaxs_decoder_upper_bound) >= np.array(lmaxs_decoder))
        else:
            lmaxs_encoder = lmaxs_encoder_upper_bound
            lmaxs_decoder = lmaxs_decoder_upper_bound

        ## encoder - cg
        prev_irreps = initial_irreps
        encoder_cg_blocks = []
        for i in range(n_cg_blocks):
            temp_irreps_hidden = (ch_size_list[i]*o3.Irreps.spherical_harmonics(lmaxs_encoder[i], 1)).sort().irreps.simplify()
            encoder_cg_blocks.append(CGNetBlock(prev_irreps,
                                                temp_irreps_hidden,
                                                w3j_matrices,
                                                linearity_first=linearity_first,
                                                filter_symmetric=filter_symmetric,
                                                use_batch_norm=self.use_batch_norm,
                                                ls_nonlin_rule=ls_nonlin_rule_list[i], # full, elementwise, efficient
                                                ch_nonlin_rule=ch_nonlin_rule_list[i], # full, elementwise
                                                norm_type=norm_type, # None, layer, signal
                                                normalization=normalization, # norm, component -> only if norm_type is not none
                                                norm_balanced=norm_balanced,
                                                norm_affine=norm_affine, # None, {True, False} -> for layer_norm, {unique, per_l, per_feature} -> for signal_norm
                                                norm_nonlinearity=norm_nonlinearity, # None (identity), identity, relu, swish, sigmoid -> only for layer_norm
                                                norm_location=norm_location, # first, between, last
                                                weights_initializer=weights_initializer,
                                                init_scale=1.0))


            prev_irreps = encoder_cg_blocks[-1].irreps_out
            print(prev_irreps.dim, prev_irreps)

        self.encoder_cg_blocks = torch.nn.ModuleList(encoder_cg_blocks)

        final_encoder_invariants = [mul for (mul, _) in prev_irreps][0] # number of channels for l = 0
        final_encoder_l1s = [mul for (mul, _) in prev_irreps][1] # number of channels for l = 1

        prev_dim = final_encoder_invariants
        if bottleneck_hidden_dims is not None and len(bottleneck_hidden_dims) > 0:
            encoder_bottleneck = []
            for hidden_dim in bottleneck_hidden_dims:
                encoder_bottleneck.append(torch.nn.Linear(prev_dim, hidden_dim))
                encoder_bottleneck.append(torch.nn.ReLU())
                prev_dim = hidden_dim
            self.encoder_bottleneck = torch.nn.Sequential(*encoder_bottleneck)
        else:
            self.encoder_bottleneck = torch.nn.Identity() # for modularity purposes

        self.encoder_mean = torch.nn.Linear(prev_dim, latent_dim)
        if self.is_vae:
            self.encoder_log_var = torch.nn.Linear(prev_dim, latent_dim)

        # component that learns the frame
        self.learn_frame = learn_frame
        if learn_frame:
            # take l=1 vectors (extract multiplicities) of last block and learn two l=1 vectors (x and pseudo-y direction)
            frame_learner_irreps_in = o3.Irreps('%dx1e' % final_encoder_l1s)
            frame_learner_irreps_out = o3.Irreps('2x1e')
            self.frame_learner = linearity(frame_learner_irreps_in, frame_learner_irreps_out)
        
        latent_irreps = o3.Irreps('%dx0e+3x1e' % (latent_dim))
        print(latent_irreps.dim, latent_irreps)

        prev_dim = latent_dim
        if bottleneck_hidden_dims is not None and len(bottleneck_hidden_dims) > 0:
            bottleneck_hidden_dims = bottleneck_hidden_dims[::-1] + [final_encoder_invariants]
            decoder_bottleneck = []
            for i, hidden_dim in enumerate(bottleneck_hidden_dims):
                decoder_bottleneck.append(torch.nn.Linear(prev_dim, hidden_dim))
                if i > 0: # only linear projection in first layer, to be symmetric with encoder
                    decoder_bottleneck.append(torch.nn.ReLU())
                prev_dim = hidden_dim
            self.decoder_bottleneck = torch.nn.Sequential(*decoder_bottleneck)
        else:
            self.decoder_bottleneck = torch.nn.Linear(latent_dim, final_encoder_invariants)

        l1_frame_irreps = o3.Irreps('3x1e')
        self.first_decoder_projection = linearity(l1_frame_irreps, o3.Irreps('%dx1e' % final_encoder_l1s)) # project back to space of irreps at the end of the encoder (l1 only)
        print(prev_irreps.dim, prev_irreps)

        ## decoder - cg
        decoder_cg_blocks = []
        
        # ch_size_list = [ch_initial_linear_projection] + ch_size_list # add channels of initial irreps
        ch_size_list = ch_size_list[::-1][1:] # reverse and exclude first channels because that's the input to the decoder
        ls_nonlin_rule_list = ls_nonlin_rule_list[::-1]
        ch_nonlin_rule_list = ch_nonlin_rule_list[::-1]
        for i in range(n_cg_blocks):
            if i == n_cg_blocks - 1:
                if self.do_initial_linear_projection:
                    temp_irreps_hidden = (ch_initial_linear_projection*o3.Irreps.spherical_harmonics(lmaxs_decoder[i], 1)).sort().irreps.simplify()
                else:
                    temp_irreps_hidden = irreps_in
            else:
                temp_irreps_hidden = (ch_size_list[i]*o3.Irreps.spherical_harmonics(lmaxs_decoder[i], 1)).sort().irreps.simplify()
            decoder_cg_blocks.append(CGNetBlock(prev_irreps,
                                                temp_irreps_hidden,
                                                w3j_matrices,
                                                linearity_first=linearity_first,
                                                filter_symmetric=filter_symmetric,
                                                use_batch_norm=self.use_batch_norm,
                                                ls_nonlin_rule=ls_nonlin_rule_list[i], # full, elementwise, efficient
                                                ch_nonlin_rule=ch_nonlin_rule_list[i], # full, elementwise
                                                norm_type=norm_type, # None, layer, signal
                                                normalization=normalization, # norm, component -> only if norm_type is not none
                                                norm_balanced=norm_balanced,
                                                norm_affine=norm_affine, # None, {True, False} -> for layer_norm, {unique, per_l, per_feature} -> for signal_norm
                                                norm_nonlinearity=norm_nonlinearity, # None (identity), identity, relu, swish, sigmoid -> only for layer_norm
                                                norm_location=norm_location, # first, between, last
                                                weights_initializer=weights_initializer,
                                                init_scale=1.0))

            # prev_irreps = decoder_cg_blocks[-1].irreps_out
            prev_irreps = temp_irreps_hidden
            print(prev_irreps.dim, prev_irreps)

        self.decoder_cg_blocks = torch.nn.ModuleList(decoder_cg_blocks)

        if self.do_initial_linear_projection: # final linear projection
            initial_irreps = (ch_initial_linear_projection*o3.Irreps.spherical_harmonics(max(irreps_in.ls), 1)).sort().irreps.simplify()
            self.final_linear_projection = linearity(initial_irreps, irreps_in)
            print(irreps_in.dim, irreps_in)

        if self.do_final_signal_norm:
            self.final_signal_norm = torch.nn.Sequential(signal_norm(irreps_in, normalization='component', affine=None))

        ## setup reconstruction loss functions
        self.signal_rec_loss = eval(NAME_TO_LOSS_FN[x_rec_loss_fn])(irreps_in, self.device)
    
    # @profile
    def encode(self, x: Tensor):
        # print('---------------------- In encoder ----------------------', file=sys.stderr)

        if self.do_initial_linear_projection:
            h = self.initial_linear_projection(x)
        else:
            h = x
        
        for i, block in enumerate(self.encoder_cg_blocks):
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
            
            if self.learn_frame and i == len(self.encoder_cg_blocks) - 1:
                last_l1_values = {1: h[1]}
        
        encoder_invariants = self.encoder_bottleneck(h[0].squeeze(-1))
        z_mean = self.encoder_mean(encoder_invariants)
        if self.is_vae:
            z_log_var = self.encoder_log_var(encoder_invariants)
        else:
            z_log_var = torch.zeros_like(z_mean) # placeholder value that won't throw errors

        if self.learn_frame:
            learned_frame = self.orthonormalize_frame(self.frame_learner(last_l1_values)[1])
        else:
            learned_frame = None

        # print('---------------------- Out of encoder ----------------------', file=sys.stderr)
        return (z_mean, z_log_var), None, learned_frame
    
    # @profile
    def decode(self, z: Tensor, frame: Tensor):
        # print('---------------------- In decoder ----------------------', file=sys.stderr)

        h = self.first_decoder_projection({1: frame})
        h[0] = self.decoder_bottleneck(z).unsqueeze(-1)
        for i, block in enumerate(self.decoder_cg_blocks):
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

        if self.do_initial_linear_projection:
            x_reconst = self.final_linear_projection(h)
        else:
            x_reconst = h

        if self.do_final_signal_norm:
            x_reconst = self.final_signal_norm(x_reconst)
        # print('---------------------- Out of decoder ----------------------', file=sys.stderr)

        return x_reconst, None

    # @profile
    def forward(self, x: Dict[int, Tensor], x_vec: Optional[Tensor] = None, frame: Optional[Tensor] = None):
        '''
        Note: this function is independent of the choice of probability distribution for the latent space,
              and of the choice of encoder and decoder. Only the inputs and outputs must be respected
        '''

        distribution_params, _, learned_frame = self.encode(x)
        if self.is_vae:
            z = self.reparameterization(*distribution_params)
        else:
            z = distribution_params[0]

        if self.learn_frame:
            frame = learned_frame

        x_reconst, _ = self.decode(z, frame)

        def make_vector(x: Dict[int, Tensor]):
            x_vec = []
            for l in sorted(list(x.keys())):
                x_vec.append(x[l].reshape((x[l].shape[0], -1)))
            return torch.cat(x_vec, dim=-1)

        # gather loss values
        x_reconst_vec = make_vector(x_reconst)
        if x_vec is None:
            x_vec = make_vector(x) # NOTE: doing this is sub-optimal!
        
        x_reconst_loss = self.signal_rec_loss(x_reconst_vec, x_vec)

        if self.is_vae:
            kl_divergence = self.kl_divergence(*distribution_params) / self.latent_dim  # KLD is summed over each latent variable, so it's better to divide it by the latent dim
                                                                                        # to get a value that is independent (or less dependent) of the latent dim size
        else:
            kl_divergence = torch.tensor(0.0) # placeholder value that won't throw errors

        return None, x_reconst_loss, kl_divergence, None, x_reconst, (distribution_params, None, None)
    
    def reparameterization(self, mean: Tensor, log_var: Tensor):

        # isotropic gaussian latent space
        stddev = torch.exp(0.5 * log_var) # takes exponential function (log var -> stddev)
        # stddev = torch.where(torch.isposinf(stddev), MAX_FLOAT32.to(self.device), stddev)
        # stddev = torch.where(torch.isneginf(stddev), (-MAX_FLOAT32).to(self.device), stddev)
        epsilon = torch.randn_like(stddev).to(self.device)        # sampling epsilon        
        z = mean + stddev*epsilon                          # reparameterization trick

        return z
    
    def kl_divergence(self, z_mean: Tensor, z_log_var: Tensor):
        # isotropic normal prior on the latent space
        return torch.mean(- 0.5 * torch.sum(1 + z_log_var - z_mean.pow(2) - z_log_var.exp(), dim=-1))
    
    def orthonormalize_frame(self, x_psy_N6):
        '''
        Gram-Schmidt process
        
        y = psy - (<x, psy> / <x, x>) x
        z = x \cross y

        x = x / ||x||
        y = y / ||y||
        z = z / ||z||
        '''
        
        x, psy = x_psy_N6[:, 0, :], x_psy_N6[:, 1, :]
        
        x_dot_psy = torch.sum(torch.mul(x, psy), dim=1).view(-1, 1)
        x_dot_x = torch.sum(torch.mul(x, x), dim=1).view(-1, 1)

        y = psy - (x_dot_psy/x_dot_x) * x
        
        z = torch.cross(x, y, dim=1)
        
        x = x / torch.sqrt(torch.sum(torch.mul(x, x), dim=1).view(-1, 1))
        y = y / torch.sqrt(torch.sum(torch.mul(y, y), dim=1).view(-1, 1))
        z = z / torch.sqrt(torch.sum(torch.mul(z, z), dim=1).view(-1, 1))
        
        xyz = torch.stack([x, y, z], dim=1)
        
        return xyz
