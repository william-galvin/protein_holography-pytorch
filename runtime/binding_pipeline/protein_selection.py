#
# File to select proteins from pdb files and store them in hdf5 file
#


from argparse import ArgumentParser
import sys, os
import h5py
import Bio.PDB as pdb
import numpy as np

parser = ArgumentParser()

parser.add_argument(
    '--pdb_list_file',
    dest='plf',
    type=str,
    help='hdf5 file with pdb list contained within'
)
parser.add_argument(
    '--resolution',
    dest='res',
    type=float,
    help='Resolution cutoff for structures'
)
parser.add_argument(
    '--image_type',
    dest='img',
    type=str,
    nargs='+',
    help='image types allowed'
)

args = parser.parse_args()

for i,x in enumerate(args.img):
    args.img[i] = x.encode()

# get list of pdbs from hdf5 file
f = h5py.File(args.plf,'r+') 
pdb_list = np.array(f['pdb_list'])
pdb_metadata = f['pdb_metadata']

good_pdbs = []

for pdb,res,img in zip(
        pdb_list,
        pdb_metadata['resolution'],
        pdb_metadata['structure_method']
):
    if img not in args.img:
        continue
    if res > args.res:
        continue
    good_pdbs.append(pdb)

# record protein list that meets specifications
subset_name = 'pdb_subsets/img={}_max_res={}/list'.format(
        '+'.join(map(lambda x : x.decode('utf-8'),args.img))
        ,args.res)
if subset_name in f.keys():
    del f[subset_name]
dset = f.create_dataset(
    subset_name,
    data=good_pdbs
)

# close hdf5 file
f.close()
