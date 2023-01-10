from pyrosetta_hdf5_ligand_parsing_check import get_ligand_atom_counts
from preprocessor_hdf5_ligand_parsing_check import PDBPreprocessor
from argparse import ArgumentParser
import numpy as np
import h5py
import pandas as pd
import sys
import os
from tqdm import tqdm

def c(np_protein_ligand):
    structure, ligand = np_protein_ligand
    pdb = structure["pdb"]
    if pdb == b"":
        return b""
    else:
        try:
            return get_ligand_atom_counts(structure, ligand)
        except Exception as e:
            print(f"Error with pdb {structure['pdb']}", file=sys.stderr)
            print(e, file=sys.stderr)
            raise e


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--hdf5_out', dest='hdf5_out', type=str, help='ouptut hdf5 filename', required=True)
    parser.add_argument('--hdf5_in', dest='hdf5_in', type=str, help='file with structures', default=False)
    parser.add_argument('--pdbbind_datadir', dest='pdbbind_datadir', type=str, help='directory containing folders for each pdb with ligand mol2 files', required=True)
    parser.add_argument('--parallelism', dest='parallelism', type=int, help='ouptput file name', default=4)
    
    args = parser.parse_args()

    if not os.path.exists("logs"):
        os.mkdir("logs")
    
    ds = PDBPreprocessor(args.hdf5_in, args.pdbbind_datadir)
    
    dt = np.dtype([
        ('pdb',"S4",()),
        ('num_ligand_atoms',"int32",()),
        ('num_ligand_atoms_noH',"int32",()),
        ('num_ligand_subst',"int32",()),
        ('num_parsed_atoms',"int32",()),
        ('num_parsed_atoms_noH',"int32",()),
        ('num_parsed_res',"int32",()),
        ('size_intersect',"int32",()),
        ('size_intersect_noH',"int32",()),
    ])
    
    print(f'writing to {args.hdf5_out}')
    
    with h5py.File(args.hdf5_in,'r') as f:
        full_pdb_list = f["structures_bound"]["pdb"][:]
    
    empty_ind = []
    failed_ind = []
    with h5py.File(args.hdf5_out,'w') as f:
        
        f.create_dataset("counts",
                         shape=(len(full_pdb_list),),
                         dtype=dt)

        for i,out in tqdm( enumerate( ds.execute(
                c,
                limit = None,
                params = {},
                parallelism = args.parallelism)), total=ds.size):

            if out == (None,):
                failed_ind.append(i)
            elif out == b"":
                empty_ind.append(i)
            else:
                # ( pdb, num_ligand_atoms, num_ligand_atoms_noH, num_ligand_subst, 
                #  num_parsed_atoms, num_parsed_atoms_noH, num_parsed_res, 
                #  size_intersect, size_intersect_noH ) = out
                
                f["counts"][i] = out

    print("Done with processing.")
