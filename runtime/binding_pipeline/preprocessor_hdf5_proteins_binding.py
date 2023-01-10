import signal
import numpy as np
import os
import functools
from multiprocessing import Pool, TimeoutError
import h5py
from biopandas.mol2 import PandasMol2



def process_data(ind, hdf5_file, pdbbind_datadir):
    assert(process_data.callback)
    with h5py.File(hdf5_file,'r') as f: 
        protein_bound = f["structures_bound"][ind]
        protein_unbound = f["structures_unbound"][ind]
        
    pdb = protein_bound["pdb"].decode()
    if pdb == "":
        return b""
    else:
        ligand = PandasMol2().read_mol2(os.path.join(pdbbind_datadir,
                                                     f'{pdb}/{pdb}_ligand.mol2'))
        if len(np.intersect1d(ligand.df["atom_type"], ["P.3", "Br", "F", "Cl"])) > 0:
            return (None,)
        else:
            ligand_coords = ligand.df[["x","y","z"]].values
            return (process_data.callback((protein_bound, ligand_coords),
                                         **process_data.params),
                    process_data.callback((protein_unbound, ligand_coords),
                                         **process_data.params)
                   )

def initializer(init, callback, params, init_params):
    if init is not None:
        init(**init_params)
    process_data.callback = callback
    process_data.params = params
    signal.signal(signal.SIGINT, signal.SIG_IGN)


class PDBPreprocessor:
    def __init__(self, hdf5_file, pdbbind_datadir):

        with h5py.File(hdf5_file,'r') as f:
            num_proteins = np.array(f["structures_bound"].shape[0])
        
        self.pdbbind_datadir = pdbbind_datadir
        self.hdf5_file = hdf5_file
        self.size = num_proteins
        self.__data = np.arange(num_proteins)
        
    def count(self):
        return len(self.__data)

    def execute(self, callback, parallelism = 8, limit = None, params = None, init = None, init_params = None):
        if limit is None:
            data = self.__data
        else:
            data = self.__data[:limit]
        with Pool(initializer = initializer, processes=parallelism, initargs = (init, callback, params, init_params)) as pool:

    
            all_loaded = True
            if all_loaded:
                # logging.info('All PDB files are loaded.')
                pass
            else:
                raise Exception("Some PDB files could not be loaded.")
            
            process_data_hdf5 = functools.partial(
                    process_data,
                    hdf5_file = self.hdf5_file,
                    pdbbind_datadir = self.pdbbind_datadir
                )
            chunksize = min( self.size // os.cpu_count() + 1, 16)

            for res in pool.imap(process_data_hdf5, data, chunksize=chunksize):
                if res:
                    yield res
                    
                    

