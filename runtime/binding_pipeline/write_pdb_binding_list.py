
from tqdm import tqdm
import pandas as pd
import numpy as np
import h5py
import sys
import os
from argparse import ArgumentParser


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--hdf5_out', dest='hdf5_out', type=str, 
                        help='ouptut hdf5 filename', required=True)
    
    parser.add_argument('--hdf5_in', dest='hdf5_in', type=str, 
                        help='input hdf5 filename', required=True)
    parser.add_argument('--split_fullname', dest='split_fullname', type=str, 
                        help='input hdf5 filename', required=True)
    
    parser.add_argument('--binding_data', dest='binding_data', type=str, 
                        help='input filename for binding data', required=True)
    
    args = parser.parse_args()

    with h5py.File(args.hdf5_in,"r") as fin:
        binding_data = pd.read_csv(args.binding_data, dtype="|S")
        
        with h5py.File(args.hdf5_out,"w") as fout:
            for subset in ["train/", "val/", "test/"]:
                fout.create_group(subset)
                
                pdbs = fin[os.path.join(args.split_fullname,subset, "pdbs")][:]
                fout[subset].create_dataset("pdbs",
                                            data=pdbs)
                
                res = binding_data.set_index("PDB", drop=False).loc[pdbs.astype(str)]
                for (c,dt) in zip(res.columns, res.dtypes):
                    if dt == np.dtype(object):
                        res[c] = res[c].astype("S")
                        
                fout[subset].create_dataset("binding_data",
                                            data = res.to_records(index=False),
                                            dtype = [(c,dt,()) for (c,dt) in zip(res.columns, res.dtypes)])
