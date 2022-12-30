import torch
from torch import nn, Tensor
import e3nn
from e3nn import o3
from copy import deepcopy

from typing import *

def is_equivariant(function: nn.Module,
                   irreps_in: o3.Irreps,
                   irreps_out: Optional[o3.Irreps] = None,
                   signal_in: Optional[Tensor] = None,
                   rtol: float = 1e-05,  # rtol for torch.allclose()
                   atol: float = 1e-08,  # atol for torch.allclose()
                   device: str = 'cpu'):
    '''
    rtol and atol may have to be relaxed to account for numerical errors of floating-point
    operations for the model computation-intensive functions. The default values are those of
    torch.allclose(), but they tend to be too strict.
    '''
    
    # if irreps_out is not provided then it is assumed that function has an irreps_out attribute,
    # and that attribute is used
    if irreps_out is None:
        irreps_out = function.irreps_out
    
    # if signal_in is not provided, sample random signal with batch_size of 1
    if signal_in is None:
        signal_in = irreps_in.randn(1, -1)
    
    lmax_in = max(irreps_in.ls)
    lmax_out = max(irreps_out.ls)
    
    rot_matrix = o3.rand_matrix()
    alpha, beta, gamma = o3.matrix_to_angles(rot_matrix)
    wigner = {}
    for l in range(max(lmax_in, lmax_out) + 1):
        wigner[l] = o3.wigner_D(l, alpha, beta, gamma)

    signal_out = function(signal_in.to(device)).detach().cpu()
    signal_out_rotated = rotate_signal(signal_out, irreps_out, wigner)
    
    signal_in_rotated = rotate_signal(signal_in, irreps_in, wigner)
    signal_rotated_out = function(signal_in_rotated.to(device)).detach().cpu()
    
    is_equiv = torch.allclose(signal_out_rotated, signal_rotated_out, rtol=rtol, atol=atol)
    mean_diff = torch.mean(torch.abs(signal_out_rotated - signal_rotated_out))
    
    return is_equiv, mean_diff


def is_vae_equivariant(vae: nn.Module,
                       irreps_in: o3.Irreps,
                       n_reps: Optional[int] = None,
                       signal_in: Optional[Tensor] = None,
                       wigner: Optional[Dict] = None,
                       rtol: float = 1e-05,  # rtol for torch.allclose()
                       atol: float = 1e-08,  # atol for torch.allclose()
                       device: str = 'cpu'):
    '''
    rtol and atol may have to be relaxed to account for numerical errors of floating-point
    operations for the model computation-intensive functions. The default values are those of
    torch.allclose(), but they tend to be too strict.
    '''
    
    # if signal_in is not provided, sample random signals. Normalize by average norm
    if signal_in is None:
        signal_in = irreps_in.randn(n_reps, -1)
        ls_indices = torch.cat([torch.tensor([l]).repeat(2*l+1) for l in irreps_in.ls])
        norm = torch.mean(torch.sqrt(torch.einsum('bf,bf,f->b', signal_in, signal_in, 1.0 / (2*ls_indices + 1))))
        signal_in = signal_in / norm
    else:
        n_reps = signal_in.shape[0]

    signal_in_copy = deepcopy(signal_in)

    signal_in = make_dict(signal_in, irreps_in)
    signal_in = put_dict_on_device(signal_in, device)
    
    if wigner is None:
        rot_matrices = torch.stack([o3.rand_matrix() for _ in range(n_reps)])
        wigner, _ = get_wigner_D_block_from_rot_matrices(irreps_in, rot_matrices)
    
    wigner_copy = deepcopy(wigner)

    wigner = put_dict_on_device(wigner, device)

    vae = vae.float().to(device)

    (z_mean, _), _, learned_frame = vae.encode(signal_in)
    signal_out, _ = vae.decode(z_mean, learned_frame)
    signal_out_rotated = make_vec(take_dict_down_from_device(rotate_signal_batch_fibers(signal_out, wigner)))
    
    signal_in_rotated = rotate_signal_batch_fibers(signal_in, wigner)
    (z_mean, _), _, learned_frame = vae.encode(signal_in_rotated)
    signal_rotated_out, _ = vae.decode(z_mean, learned_frame)
    signal_rotated_out = make_vec(take_dict_down_from_device(signal_rotated_out))
    
    is_equiv = torch.allclose(signal_out_rotated, signal_rotated_out, rtol=rtol, atol=atol)
    mean_diff = torch.mean(torch.abs(signal_out_rotated - signal_rotated_out), dim=1)

    stats = {
        'mean': torch.mean(signal_out_rotated).item(),
        'mean_square': torch.mean(torch.square(signal_out_rotated)).item(),
        'mean_abs': torch.mean(torch.abs(signal_out_rotated)).item(),
        'std': torch.mean(torch.std(signal_out_rotated, dim=1)).item(),
        'std_square': torch.mean(torch.std(torch.square(signal_out_rotated), dim=1)).item(),
        'std_abs': torch.mean(torch.std(torch.abs(signal_out_rotated), dim=1)).item()
    }
    
    return {
        'is_equiv': is_equiv,
        'mean_diff': mean_diff,
        'stats': stats,
        'signal_in': signal_in_copy,
        'wigner': wigner_copy
    }


def is_vae_equivariant__BREAK_EQUIVARIANCE(vae: nn.Module,
                       irreps_in: o3.Irreps,
                       n_reps: Optional[int] = None,
                       signal_in: Optional[Tensor] = None,
                       rtol: float = 1e-05,  # rtol for torch.allclose()
                       atol: float = 1e-08,  # atol for torch.allclose()
                       device: str = 'cpu'):
    '''
    rtol and atol may have to be relaxed to account for numerical errors of floating-point
    operations for the model computation-intensive functions. The default values are those of
    torch.allclose(), but they tend to be too strict.
    '''
    
    # if signal_in is not provided, sample random signals. Normalize by average norm
    if signal_in is None:
        signal_in = irreps_in.randn(n_reps, -1)
        ls_indices = torch.cat([torch.tensor([l]).repeat(2*l+1) for l in irreps_in.ls])
        norm = torch.mean(torch.sqrt(torch.einsum('bf,bf,f->b', signal_in, signal_in, 1.0 / (2*ls_indices + 1))))
        signal_in = signal_in / norm
    else:
        n_reps = signal_in.shape[0]
    
    stats = {
        'mean': torch.mean(signal_in),
        'mean_square': torch.mean(torch.square(signal_in)),
        'mean_abs': torch.mean(torch.abs(signal_in)),
        'std': torch.mean(torch.std(signal_in, dim=1)),
        'std_square': torch.mean(torch.std(torch.square(signal_in), dim=1)),
        'std_abs': torch.mean(torch.std(torch.abs(signal_in), dim=1))
    }
    
    signal_in = make_dict(signal_in, irreps_in)
    signal_in = put_dict_on_device(signal_in, device)
    
    rot_matrices = torch.stack([o3.rand_matrix() for _ in range(n_reps)])

    wigner, _ = get_wigner_D_block_from_rot_matrices(irreps_in, rot_matrices)
    wigner = put_dict_on_device(wigner, device)

    vae = vae.float().to(device)

    (z_mean, _), _, learned_frame = vae.encode(signal_in)
    # learned_frame = torch.nn.functional.silu(learned_frame) # break equivariance in the frame
    signal_out, _ = vae.decode(z_mean, learned_frame)
    for l in signal_out:
        signal_out[l] *= torch.sum(signal_out[l]) # break equivariance in the output
    signal_out_rotated = make_vec(take_dict_down_from_device(rotate_signal_batch_fibers(signal_out, wigner)))
    
    signal_in_rotated = rotate_signal_batch_fibers(signal_in, wigner)
    (z_mean, _), _, learned_frame = vae.encode(signal_in_rotated)
    # learned_frame = torch.nn.functional.silu(learned_frame) # break equivariance in the frame
    signal_rotated_out, _ = vae.decode(z_mean, learned_frame)
    for l in signal_rotated_out:
        signal_rotated_out[l] *= torch.sum(signal_rotated_out[l]) # break equivariance in the output
    signal_rotated_out = make_vec(take_dict_down_from_device(signal_rotated_out))
    
    is_equiv = torch.allclose(signal_out_rotated, signal_rotated_out, rtol=rtol, atol=atol)
    mean_diff = torch.mean(torch.abs(signal_out_rotated - signal_rotated_out), dim=1)
    
    return is_equiv, mean_diff, stats




