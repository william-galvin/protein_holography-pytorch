import sys
import os
import numpy as np
from preprocessor import PDBPreprocessor
from argparse import ArgumentParser
import Bio.PDB as pdb
import h5py
from tqdm import tqdm
from progress.bar import Bar


def c(pdb, parser, pdb_dir, keys):
    try:
        struct = parser.get_structure(pdb+'struct', os.path.join(args.pdb_dir, pdb) + '.pdb')
    except Exception as e:
        raise e
        
    pdb_metadata = []
    for k in keys:
        if k == "pdb":
            info = pdb.encode()
        elif k == 'missing_residues':
            info = len( struct.header[k] )
        else:
            info = struct.header[k]
            if type(info) == str:
                info = info.encode()
            if k == 'resolution' and info == None:
                info = 10.
        pdb_metadata.append(info)
        
    return pdb_metadata
        

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        '--pdb_list_file',
        dest='plf',
        type=str,
        help='hdf5 file with pdb list contained within'
    )
    parser.add_argument(
        '--pdb_dir',
        dest='pdb_dir',
        type=str,
        help='Directory for pdb files'
    )
    parser.add_argument('--parallelism', 
                        dest='parallelism', 
                        type=int, help='ouptput file name', default=4)
    args = parser.parse_args()
    
    if not os.path.exists("logs"):
        os.mkdir("logs")    

    with h5py.File(args.plf,'r+') as f:
        pdb_list = np.array(f['pdb_list']).astype(str)
        
        dt = np.dtype( [("pdb","S4",()),
                ('deposition_date', "|S10",()),
                ('release_date', "|S10",()),
                ('structure_method', "|S57",()),
                ('resolution', "f8",()),
                ('has_missing_residues', bool,()),
                ("missing_residues", "i8",()),])
        
        keys = dt.names
        
        if "pdb_metadata" in f.keys():
            del f["pdb_metadata"]
            
        f.create_dataset('pdb_metadata',
                         shape = (len(pdb_list), ),
                         dtype = dt
                        )
    
    parser = pdb.PDBParser(QUIET=True)
    ds = PDBPreprocessor(pdb_list)
    
    with h5py.File(args.plf,'r+') as f:
        for i,pdb_metadata in tqdm( enumerate(ds.execute(
                c,
                limit = None,
                params = {"parser":parser, "pdb_dir":args.pdb_dir, "keys":keys},
                parallelism = args.parallelism)), total = ds.size):

            f["pdb_metadata"][i] = (*pdb_metadata,)