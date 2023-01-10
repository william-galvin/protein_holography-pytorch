#
# protein_list.py -- Michael Pun -- 12 May 2021
# 
# Take a directory of pdb files as an argument and output an hdf5
# file with a list of the protein names as the dataset.
#
# This hdf5 file will serve as the foundation for the dataset
# and will later contain lists of high resolution structures
# and splits for training and test sets.
#

from argparse import ArgumentParser
import os
import h5py
import pandas as pd
import numpy as np
import logging


parser = ArgumentParser()

parser.add_argument(
    '--pdb_list',
    dest='pdb_list',
    type=str,
    help='CSV file with a column "PDB"'
)
parser.add_argument(
    '--pdb_dir',
    dest='pdb_dir',
    type=str,
    help='Directory of pdb files"'
)
parser.add_argument(
    '--filename',
    dest='filename',
    type=str,
    help='Name for the dataset'
)
parser.add_argument(
    '--data_dir',
    dest='data_dir',
    type=str,
    default='data',
    help='Directory to save data'
)

args = parser.parse_args()

if not os.path.exists("logs"):
        os.mkdir("logs")
logfile = "logs/{}.log".format(os.path.splitext(os.path.basename(__file__))[0])
logging.basicConfig(filename = logfile, filemode="w", level=logging.DEBUG)
print("Saving log to " + logfile)

pdb_names = pd.read_csv(args.pdb_list)["PDB"].to_numpy().astype(str)

logging.warning("Only keeping pdbs found in "+args.pdb_dir)
found_in_pdbdir = []
for file in os.listdir(args.pdb_dir):
    # if file.endswith(".pdb"):
    pdb = file[:4]
    if pdb in pdb_names:
        found_in_pdbdir.append(pdb)
            
logging.warning("Found {0} out of {1} pdbs in {2}.".format(len(found_in_pdbdir),
                                                           len(pdb_names),
                                                           args.pdb_dir))

filepath = os.path.join( args.data_dir, args.filename) + '.hdf5'

try:
    with h5py.File(filepath ,'w-') as f:
        dset = f.create_dataset('pdb_list',
                                data=np.array( found_in_pdbdir ).astype("|S"))
except OSError:
    logging.warning(filepath + " already exists.")
    
