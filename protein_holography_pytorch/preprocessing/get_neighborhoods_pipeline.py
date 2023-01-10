#
# This file computes the atomic spherical coordinates in a given set of
# neighborhoods and outputs a file with these coordinates.
#
# It takes as arguments:
#  - The name of the ouput file
#  - Name of central residue dataset
#  - Number of threads
#  - The neighborhood radius
#  - "easy" flag to include central res
#

from turtle import back
from get_neighborhoods import get_neighborhoods_from_protein, pad_neighborhoods
from preprocessor_hdf5_proteins import PDBPreprocessor
from argparse import ArgumentParser
import numpy as np
import h5py
import sys
import logging
from progress.bar import Bar
import traceback

ALL_AAs = ['R', 'H', 'K', 'D', 'E', 'S', 'T', 'N', 'Q', 'C', 'U', 'G', 'P', 'A', 'I', 'L', 'M', 'F', 'W', 'Y', 'V']

def callback(np_protein, r, remove_central_residue, backbone_only):

    try:
        neighborhoods = get_neighborhoods_from_protein(np_protein, r=r, remove_central_residue=remove_central_residue, backbone_only=backbone_only)
        padded_neighborhoods = pad_neighborhoods(neighborhoods, padded_length=1000)
    except Exception as e:
        print(e)
        print('Error with ', np_protein[0])
        return (None,)
    
    return (padded_neighborhoods)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--input_hdf5', type=str, required=True)
    parser.add_argument('--output_hdf5', type=str, required=True)
    parser.add_argument('--dataset_name', type=str, default='data')
    parser.add_argument('--parallelism', type=int, default=40)
    parser.add_argument('--radius', type=float, default=12.5)
    parser.add_argument('--remove_central_residue', type=bool, default=False)
    parser.add_argument('--backbone_only', type=bool, default=False)
    parser.add_argument('--AAs', type=str, default='all',
                        help='List of amino-acid types to collect. Either "all" or provided in comma-separated form.')
    
    args = parser.parse_args()

    if args.AAs == 'all':
        filter_AAs = set(ALL_AAs)
    else:
        filter_AAs = set(args.AAs.split(','))
        
    logging.basicConfig(level=logging.DEBUG)
    ds = PDBPreprocessor(args.input_hdf5, args.input_key)

    max_atoms = 1000
    dt = np.dtype([
        ('res_id','S6', (6)), # S5, 5 (old) ; S6, 6 (new with 2ndary structure)
        ('atom_names', 'S4', (max_atoms)),
        ('elements', 'S1', (max_atoms)),
        ('res_ids', 'S6', (max_atoms, 6)), # S5, 5 (old) ; S6, 6 (new with 2ndary structure)
        ('coords', 'f8', (max_atoms, 3)),
        ('SASAs', 'f8', (max_atoms)),
        ('charges', 'f8', (max_atoms)),
    ])
    print(dt)
    print('writing hdf5 file')
    curr_size = 1000
    with h5py.File(args.output_hdf5, 'w') as f:
        # Initialize dataset
        f.create_dataset(args.input_key,
                         shape=(curr_size,),
                         maxshape=(None,),
                         dtype=dt)
        
    print('calling parallel process')
    with Bar('Processing', max = ds.count(), suffix='%(percent).1f%%') as bar:
        with h5py.File(args.output_hdf5, 'r+') as f:
            n = 0
            for i, neighborhoods in enumerate(ds.execute(callback,
                                                         limit = ds.size,
                                                         params = {'r': args.radius,
                                                                   'remove_central_residue': args.remove_central_residue,
                                                                   'backbone_only': args.backbone_only},
                                                         parallelism = args.parallelism)):
                
                print(i, file=sys.stderr)

                if neighborhoods[0] is None:
                    bar.next()
                    continue
                
                for neighborhood in neighborhoods:

                    if n == curr_size:
                        curr_size += 1000
                        f[args.input_key].resize((curr_size,))
                    
                    # only add neighborhoods of desired AA types
                    if neighborhood[0][0].decode('utf-8') in filter_AAs:
                        f[args.input_key][n] = (*neighborhood,)
                        n += 1
                
                bar.next()

            # finally, resize dataset to be of needed shape to exactly contain the data and nothing more
            f[args.input_key].resize((n,))
        
    print('%d total neighborhoods saved.' % (n))
    
    print('Done with parallel computing')
