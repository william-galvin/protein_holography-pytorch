
import os, sys
import numpy as np
import torch

from protein_holography_pytorch.utils.conversions import spherical_to_cartesian__numpy
from protein_holography_pytorch.utils.protein_naming import aa_to_ind_size, one_letter_to_aa
from protein_holography_pytorch.so3 import ZernickeRadialFunctions, RadialSphericalTensor, MultiChannelRadialSphericalTensor
from typing import *
from torch import Tensor

BACKBONE_ATOMS = [b' N  ', b' CA ', b' C  ', b' O  ']
BACKBONE_ATOMS_PLUS_CB = [b' N  ', b' CA ', b' C  ', b' O  ', b' CB ']

def get_zernikegrams(nbs: np.array, # of custom dtype
                     rcut: float,
                     rmax: int,
                     lmax: int,
                     backbone_only: bool,
                     get_H: bool,
                     get_SASA: bool,
                     get_charge: bool,
                     n_channels: int,
                     request_frame: bool = False,
                     rst_normalization: str = 'square',
                     mul_rst_normalization: Optional[str] = None,
                     convert_to_cartesian: bool = True
                    ) -> dict:

    OnRadialFunctions = ZernickeRadialFunctions(rcut, rmax+1, lmax, complex_sph = False, record_zeros = False)
    rst = RadialSphericalTensor(rmax+1, OnRadialFunctions, lmax, 1, 1)
    mul_rst = MultiChannelRadialSphericalTensor(rst, n_channels)
    
    zernikegrams = []
    data_ids = []
    frames = []
    labels = []
    for nb in nbs:

        coords, weights, frame, nb_id = process_data(nb,
                                                     convert_to_cartesian=convert_to_cartesian,
                                                     backbone_only=backbone_only,
                                                     request_frame=request_frame,
                                                     get_H=get_H,
                                                     get_SASA=get_SASA,
                                                     get_charge=get_charge)
        
        zgram, frame, nb_id = get_zernikegram_of_neighborhood(coords, weights, frame, nb_id, rst, mul_rst, rst_normalization, mul_rst_normalization, n_channels)

        if frame is None and request_frame:
            print('Error: frme is None when requested. Skipping neighborhood.', file=sys.stderr)
            continue
        elif frame is not None:
            frames.append(frame)

        zernikegrams.append(zgram)
        data_ids.append(nb_id)
        labels.append(aa_to_ind_size[one_letter_to_aa[nb_id[0].decode('utf-8')]])

    zernikegrams = np.vstack(zernikegrams)
    data_ids = np.array(data_ids)
    if request_frame:
        frames = np.vstack(frames).reshape(-1, 3, 3)
    else:
        frames = None
    labels = np.hstack(labels)

    return {'projections': zernikegrams,
            'data_ids': data_ids,
            'frames': frames,
            'labels': labels}


def get_zernikegram_of_neighborhood(coords_list: list,
                      weights_list: list,
                      frame: Tensor,
                      nb_id: str, # or maybe byte-string?
                      rst: RadialSphericalTensor,
                      mul_rst: MultiChannelRadialSphericalTensor,
                      rst_normalization: str,
                      mul_rst_normalization: Optional[str],
                      n_channels: int):
    assert (n_channels == len(coords_list) or n_channels == 1) # should be 7 for yes_sidechain: [C, O, N, S, H, SASA, charge], 4 for no_sidechain: [CA, C, N, O]
    assert (n_channels == len(weights_list) or n_channels == 1) # should be 7 for yes_sidechain: [C, O, N, S, H, SASA, charge], 4 for no_sidechain: [CA, C, N, O]

    if n_channels == 1:
        coeffs = rst.with_peaks_at(torch.cat(coords_list, dim=0), None, normalization=rst_normalization)
    else:
        disentangled_coeffs = []
        for coords, weights in zip(coords_list, weights_list):
            disentangled_coeffs.append(rst.with_peaks_at(coords, weights, normalization=rst_normalization))
        coeffs = mul_rst.combine(torch.stack(disentangled_coeffs, dim=0), normalization=mul_rst_normalization)

    if frame is not None:
        frame = torch.stack(tuple(map(lambda x: torch.tensor(x), frame)))
        return (coeffs.numpy(), frame.numpy(), nb_id)
    else:
        # print(nb_id)
        # print('Frame is None. Either it was not requested or something is wrong with central amino acid.')
        return (coeffs.numpy(), None, nb_id)


def process_data(nb: np.array,
                 convert_to_cartesian: bool = True,
                 backbone_only: bool = False,
                 request_frame: bool = False,
                 get_H: bool = False,
                 get_SASA: bool = False,
                 get_charge: bool = False):

    nb_id = nb['res_id']

    if convert_to_cartesian:
        cartesian_coords = spherical_to_cartesian__numpy(nb['coords'])
    else:
        cartesian_coords = nb['coords']

    if backbone_only:
        if len(nb['atom_names'][0]) != 4:
            print('Bug!: ', nb['atom_names'][0], file=sys.stderr)
            raise Exception
        CA_coords = torch.tensor(cartesian_coords[nb['atom_names'] == b' CA '])
        C_coords = torch.tensor(cartesian_coords[nb['atom_names'] == b' C  '])
        N_coords = torch.tensor(cartesian_coords[nb['atom_names'] == b' N  '])
        O_coords = torch.tensor(cartesian_coords[nb['atom_names'] == b' O  '])
    else:
        C_coords = torch.tensor(cartesian_coords[nb['elements'] == b'C'])
        N_coords = torch.tensor(cartesian_coords[nb['elements'] == b'N'])
        O_coords = torch.tensor(cartesian_coords[nb['elements'] == b'O'])
        S_coords = torch.tensor(cartesian_coords[nb['elements'] == b'S'])
        H_coords = torch.tensor(cartesian_coords[nb['elements'] == b'H'])
        SASA_coords = torch.tensor(cartesian_coords[nb['elements'] != b''])
        SASA_weights = torch.tensor(nb['SASAs'][nb['elements'] != b''])
        charge_coords = torch.tensor(cartesian_coords[nb['elements'] != b''])
        charge_weights = torch.tensor(nb['charges'][nb['elements'] != b''])

    if request_frame:
        try:
            central_res = np.logical_and.reduce(nb['res_ids'] == nb['res_id'], axis=-1)
            central_CA_coords = np.array([0.0, 0.0, 0.0]) # since we centered the neighborhood on the alpha carbon
            central_N_coords = np.squeeze(cartesian_coords[central_res][nb['atom_names'][central_res] == b' N  '])
            central_C_coords = np.squeeze(cartesian_coords[central_res][nb['atom_names'][central_res] == b' C  '])

            # assert that there is only one atom with three coordinates
            assert (central_CA_coords.shape[0] == 3), 'first assert'
            assert (len(central_CA_coords.shape) == 1), 'second assert'
            assert (central_N_coords.shape[0] == 3), 'third assert'
            assert (len(central_N_coords.shape) == 1), 'fourth assert'
            assert (central_C_coords.shape[0] == 3), 'fifth assert'
            assert (len(central_C_coords.shape) == 1), 'sixth assert'

            # y is unit vector perpendicular to x and lying on the plane between CA_N (x) and CA_C
            # z is unit vector perpendicular to x and the plane between CA_N (x) and CA_C
            x = central_N_coords - central_CA_coords
            x = x / np.linalg.norm(x)

            CA_C_vec = central_C_coords - central_CA_coords

            z = np.cross(x, CA_C_vec)
            z = z / np.linalg.norm(z)

            y = np.cross(z, x)

            frame = (x, y, z)
                
        except Exception as e:
            print(e)
            print('No central residue (or other unwanted error).')
            frame = None
    else:
        frame = None
    
    if backbone_only:
        coords = [CA_coords, C_coords, N_coords, O_coords]
        weights = [None, None, None, None]
    else:
        coords = [C_coords,N_coords,O_coords,S_coords] #,H_coords,SASA_coords,charge_coords]
        weights = [None,None,None,None] #,None,SASA_weights,charge_weights]
        if get_H:
            coords += [H_coords]
            weights += [None]
        if get_SASA:
            coords += [SASA_coords]
            weights += [SASA_weights]
        if get_charge:
            coords += [charge_coords]
            weights += [charge_weights]

    return coords, weights, frame, nb_id