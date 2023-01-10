import h5py
from functools import partial, reduce
import numpy as np
from sklearn.neighbors import KDTree
# import geo2 as geo

# slice array along given indices
def slice_array(arr,inds):
    return arr[inds]

# given a set of neighbor coords, slice all info in the npProtein along neighbor inds
def get_neighborhoods(neighbor_inds,npProtein):
    return list(map(partial(slice_array,inds=neighbor_inds),npProtein))


def get_neighborhoods_from_protein(np_protein, ligand_coords, nh_radius, remove_central_residue=False):
    
    pdb = np_protein["pdb"]
    
    real_locs = np_protein['atom_names'] != b''
    
    coords = np_protein['coords'][real_locs]
    res_ids = np_protein['res_ids'][real_locs]
    ca_locs = np_protein['atom_names'][real_locs] == b' CA '
    
    """
        Find residues in pocket
    """
    tree = KDTree(coords,leaf_size=2)
    
    neighbors_list = np.concatenate( 
        tree.query_radius(ligand_coords, r=nh_radius, return_distance=False) 
    )
    ligand_locs = np.concatenate( 
        tree.query_radius(ligand_coords, r=0.01, return_distance=False) 
    )
    neighbors_list = np.setdiff1d(neighbors_list, ligand_locs)
    
    ca_coords = coords[neighbors_list][ca_locs[neighbors_list]]
    nh_ids = res_ids[neighbors_list][ca_locs[neighbors_list]]
        
    """
        Tree-search neighbors of residue central CA atoms.
    """
    neighbors_list = tree.query_radius(ca_coords, r=nh_radius, count_only=False)
    
    get_neighbors_custom = partial(
        get_neighborhoods,                          
        npProtein=[np_protein[x] for x in range(1,7)]
        )
    res_ids = np_protein['res_ids'][real_locs]
    nh_atoms = np_protein['atom_names'][real_locs]

    # remove central residue if requested
    if remove_central_residue:
        for i, nh_id, neighbor_list in zip(np.arange(len(nh_ids)), nh_ids, neighbors_list):
            neighbors_list[i] = [x for x in neighbor_list if np.logical_or.reduce(res_ids[x] != nh_id, axis=-1)]
    else:
        # still must always remove central alpha carbon, otherwise we get nan values since its radius is 0.0
        for i, nh_id, neighbor_list in zip(np.arange(len(nh_ids)), nh_ids, neighbors_list):
            neighbors_list[i] = [x for x in neighbor_list if (np.logical_or.reduce(res_ids[x] != nh_id, axis=-1) or nh_atoms[x] != b' CA ')] # if (np.logical_or.reduce(res_ids[x] != nh_id, axis=-1) or nh_atoms[x] != b' CA ')

    neighborhoods = list(map(get_neighbors_custom,neighbors_list))
    
    for nh,nh_id,ca_coord in zip(neighborhoods,nh_ids,ca_coords):
        
        # nh[3] = np.array(geo.cartesian_to_spherical(nh[3] - ca_coord)) # don't need conversion to spherical coords with pytorch
        nh[3] = np.array(nh[3] - ca_coord)

        nh.insert(0,nh_id)

    return neighborhoods

# given a matrix, pad it with empty array
def pad(arr,padded_length=100):
    try:
        # get dtype of input array
        dt = arr[0].dtype
    except IndexError as e:
        print(e)
        print(arr)
        raise Exception
    # shape of sub arrays and first dimension (to be padded)
    shape = arr.shape[1:]
    orig_length = arr.shape[0]

    # check that the padding is large enough to accomdate the data
    if padded_length < orig_length:
        print('Error: Padded length of {}'.format(padded_length),
              'is smaller than original length of array {}'.format(orig_length))

    # create padded array
    padded_shape = (padded_length,*shape)
    mat_arr = np.empty(padded_shape, dtype=dt)
    
    # if type is string fill array with empty strings
    if np.issubdtype(bytes, dt):
        mat_arr.fill(b'')

    # add data to padded array
    mat_arr[:orig_length] = np.array(arr)
    
    return mat_arr

def pad_neighborhood(ragged_structure, padded_length=100):

    pad_custom = partial(pad,padded_length=padded_length)
    
    mat_structure = list(map(pad_custom,ragged_structure))

    return mat_structure

def pad_neighborhoods( neighborhoods, padded_length=600):
    padded_neighborhoods = []
    for i,nh in enumerate(neighborhoods):
        padded_neighborhoods.append(
            pad_neighborhood(
                [nh[i] for i in range(1,7)],
                padded_length=padded_length
            )
        )
    for nh,padded_neighborhood in zip(neighborhoods,padded_neighborhoods):
        padded_neighborhood.insert(0,nh[0])
        
    return padded_neighborhoods
