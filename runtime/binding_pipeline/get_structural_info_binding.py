
import os, sys

from pyrosetta_hdf5_proteins import get_structural_info,pad_structural_info
# from pyrosetta_hdf5_relax import relax_pocket
from preprocessor_pdbs_binding import PDBPreprocessor
from argparse import ArgumentParser
import numpy as np
import h5py

## LOCAL ABSOLUTE PYROSETTA PATH
sys.path.append('/gscratch/spe/gvisan01/PyRosetta4.Release.python39.linux.release-335')

import pyrosetta
from pyrosetta.toolbox.extract_coords_pose import pose_coords_as_rows
from pyrosetta.rosetta.core.pose import deep_copy
from pyrosetta.rosetta import protocols
from functools import partial, reduce
from sklearn.neighbors import KDTree
from tqdm import tqdm



def c(poses_and_coords):
    
    if poses_and_coords[1] is None:
        return (None,)
    try:
        pdb, pose_pdb, pose_protein, ligand_coords = poses_and_coords
        
        pyrosetta.rosetta.core.pose.renumber_pdbinfo_based_on_conf_chains(pose_protein)
        
        pose_unbound = deep_copy(pose_protein) 
        # relax_pocket( pose_unbound, ligand_coords, nrepeats = 1)
        
        pdb , ragged_structural_info_unbound  = get_structural_info(pose_unbound)
        mat_structural_info_unbound  = pad_structural_info(
            ragged_structural_info_unbound ,padded_length=200000
        )
        
        pdb_coords_rows = pose_coords_as_rows(pose_pdb)
        res_idx = np.concatenate( [
            [i for atom in pose_pdb.residues[i].atoms()] for i in range(1,pose_pdb.size()+1)]
        )

        tree = KDTree(pdb_coords_rows,leaf_size=2)
        ligand_locs = reduce(np.union1d, 
                             tree.query_radius(ligand_coords, r=0.01, return_distance=False)
                            )

        ligand_idx = np.unique( res_idx[ligand_locs])
        
        vec = pyrosetta.rosetta.utility.vector1_unsigned_long(len(ligand_idx))
        for (i,x) in enumerate(ligand_idx):
            vec[i+1] = x
            
        pyrosetta.rosetta.core.pose.pdbslice( pose_pdb, vec)
        pyrosetta.rosetta.core.pose.append_pose_to_pose(pose_protein, pose_pdb)
        
        pyrosetta.rosetta.core.pose.renumber_pdbinfo_based_on_conf_chains(pose_protein)
        
        # relax_pocket( pose_protein, ligand_coords, nrepeats = 1)
        
        pdb_bound, ragged_structural_info_bound = get_structural_info(pose_protein)
        mat_structural_info_bound = pad_structural_info(
            ragged_structural_info_bound,padded_length=200000
        )
        
        print("Successfully processed " + pdb.decode(), file=sys.stderr)
        return (pdb, mat_structural_info_unbound, mat_structural_info_bound)
    except Exception as e:
        print(e, file=sys.stderr)
        print('Error with ' + pdb.decode(), file=sys.stderr)
        return (None,)
    


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--hdf5_out', dest='hdf5_out', type=str, 
                        help='ouptut hdf5 filename', required=True)
    parser.add_argument('--pdb_list', dest='pdb_list', type=str, 
                        help='pdb list within hdf5_in file', required=True)
    parser.add_argument('--binding_dataset_name', dest='binding_dataset_name', type=str, 
                        help='full binding dataset name in hdf5_in file', required=True)
    parser.add_argument('--parallelism', dest='parallelism', type=int, help='ouptput file name', default=4)
    parser.add_argument('--hdf5_in', dest='hdf5_in', type=str, 
                        help='hdf5 filename', required=True)
    parser.add_argument('--pdb_dir', dest='pdb_dir', type=str,
                        help='pdb_dir data directory', required=True)
    parser.add_argument('--pdbbind_dir', dest='pdbbind_dir', type=str,
                        help='pdbbind_dir data directory', required=True)
    
    args = parser.parse_args()
    

    if not os.path.exists("logs"):
        os.mkdir("logs")
    
    

    max_atoms = 200000
    dt = np.dtype([
        ('pdb','S4',()),
        ('atom_names', 'S4', (max_atoms)),
        ('elements', 'S1', (max_atoms)),
        ('res_ids', 'S5', (max_atoms,5)),
        ('coords', 'f8', (max_atoms,3)),
        ('SASAs', 'f8', (max_atoms)),
        ('charges', 'f8', (max_atoms)),
    ])
    failed_ind = []
    with h5py.File(args.hdf5_out,'w') as f:
        
        with h5py.File(args.hdf5_in,"r") as fin:
            f.create_dataset("binding_data", data = fin[args.binding_dataset_name])
            pdb_list = np.array(fin[args.pdb_list])
            
        ds = PDBPreprocessor(pdb_list, args.pdb_dir, args.pdbbind_dir)
        
        f.create_dataset("structures_bound",
                         shape=(ds.size,),
                         dtype=dt)
        f.create_dataset("structures_unbound",
                         shape=(ds.size,),
                         dtype=dt)

        for pdb,(i,structural_info) in tqdm( zip(pdb_list, enumerate(
                ds.execute(c,limit = None,params = {},parallelism = args.parallelism))
                                          ), total=ds.size):
            
            if structural_info[0] is None:
                print(f"pose was none for {i}th pdb ({pdb})", file=sys.stderr)
                failed_ind.append(i)
            else:
                pdb, struct_unbound, struct_bound = structural_info
                f["structures_unbound"][i] = (pdb, *struct_unbound,)
                f["structures_bound"][i] = (pdb, *struct_bound,)
                print(f"Wrote {i}th pdb ({pdb}) to hdf5 file", file=sys.stderr)

            if i % 20 == 0:
                print(f"{i}/{ds.size}")
    print("Done with processing.")
    print("""
        The following pdbs failed:
        \t ind: {0}
        \t pdb: {1}
        """.format(failed_ind, pdb_list[failed_ind]))

    print("Done.")
    
