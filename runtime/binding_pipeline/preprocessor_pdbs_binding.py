import signal
import os
import numpy as np
from functools import partial
from multiprocessing import Pool
import h5py
import logging
import pyrosetta
from biopandas.mol2 import PandasMol2

init_flags = '-ignore_unrecognized_res 1 -include_current -ex1 -ex2 -mute all -include_sugars -ignore_zero_occupancy false -obey_ENDMDL 1'
pyrosetta.init(init_flags)

def process_data(pdb, pdb_dir, pdbbind_dir):
    assert(process_data.callback)
    pdb = pdb.decode()
    pdb_file = os.path.join(pdb_dir , pdb + ".pdb")
    protein_file = os.path.join(pdbbind_dir , pdb, pdb + "_protein.pdb")
    ligand_file = os.path.join(pdbbind_dir , pdb, pdb + "_ligand.mol2")

    try:
        pose_pdb = pyrosetta.pose_from_pdb(pdb_file)
        pose_protein = pyrosetta.pose_from_pdb(protein_file)
        ligand = PandasMol2().read_mol2(ligand_file)
        ligand_coords = ligand.df[["x","y","z"]].values

        return process_data.callback((pdb,pose_pdb, pose_protein, ligand_coords), **process_data.params)
    except:
        return process_data.callback((pdb,None),**process_data.params)

# def process_data(pdb, pdbbind_dir):
#     assert(process_data.callback)

#     pdb = pdb.decode('utf-8')

#     pdb_file = os.path.join(pdbbind_dir , pdb, pdb + "_bound.pdb")
#     pdbbind_file = os.path.join(pdbbind_dir , pdb, pdb + "_protein.pdb")
    
#     try:
#         pose_pdb = pyrosetta.pose_from_pdb(pdb_file)
#         pose_pdbbind =  pyrosetta.pose_from_pdb(pdbbind_file)
#         # logging.info("Successfully created pose for " + pdb)
#         return ( process_data.callback(pose_pdb, **process_data.params),
#                  process_data.callback(pose_pdbbind, **process_data.params)
#                )
#     except:
#         # logging.error('Pose could not be created for protein {}'.format(pdb))
#         return process_data.callback(None,**process_data.params)
    
    

def initializer(init, callback, params, init_params):
    if init is not None:
        init(**init_params)
    process_data.callback = callback
    process_data.params = params
    signal.signal(signal.SIGINT, signal.SIG_IGN)


class PDBPreprocessor:
    def __init__(self, pdb_list, pdb_dir, pdbbind_dir):
        
        self.__pdb_dir = pdb_dir
        self.__pdbbind_dir = pdbbind_dir
        self.__data = pdb_list
        self.size = len(pdb_list)
        
    def count(self):
        return len(self.__data)

    def execute(self, callback, parallelism = 8, limit = None, params = None, init = None, init_params = None):
        if limit is None:
            data = self.__data
        else:
            data = self.__data[:limit]
        with Pool(initializer = initializer, processes=parallelism, 
                  initargs = (init, callback, params, init_params)) as pool:

            for res in pool.imap(partial(process_data, 
                                         pdbbind_dir = self.__pdbbind_dir, pdb_dir = self.__pdb_dir), 
                                 data):
                if res:
                    yield res