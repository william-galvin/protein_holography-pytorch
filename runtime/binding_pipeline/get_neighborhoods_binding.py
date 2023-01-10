from pyrosetta_hdf5_neighborhoods_binding import get_neighborhoods_from_protein,pad_neighborhoods
from preprocessor_hdf5_proteins_binding import PDBPreprocessor
from argparse import ArgumentParser
import numpy as np
import h5py
import pandas as pd
import sys
import os
from tqdm import tqdm

def c(np_protein_ligand, nh_radius, max_atoms, remove_central_residue):
    structure, ligand = np_protein_ligand
    pdb = structure["pdb"]
    if pdb == b"":
        return b""
    else:
        try:
            out = get_neighborhoods_from_protein(structure, ligand, nh_radius, remove_central_residue=remove_central_residue)
            if out == b"":
                print("b'' for "+pdb.decode())
                return b""
            elif out == (None,):
                print("None for "+pdb.decode())
                return (None,)
            else:
                out = pad_neighborhoods(out,padded_length=max_atoms)
                return out
        except Exception as e:
            print(f"Error with pdb {structure['pdb']}", file=sys.stderr)
            print(e, file=sys.stderr)
            raise e


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--hdf5_out', dest='hdf5_out', type=str, help='ouptut hdf5 filename', required=True)
    parser.add_argument('--hdf5_in', dest='hdf5_in', type=str, help='hdf5 filename', default=False)
    parser.add_argument('--pdbbind_datadir', dest='pdbbind_datadir', type=str, help='directory containing folders for each pdb with ligand mol2 files', required=True)
    parser.add_argument('--binding_data', dest='binding_data', type=str, help='input filename for binding data', required=True)
    parser.add_argument('--nh_radius', dest='nh_radius', type=float, help='neighborhood radius', default=10.)
    parser.add_argument('--max_atoms', dest='max_atoms', type=int, help='size of padded neighborhood', default=1000)
    parser.add_argument('--parallelism', dest='parallelism', type=int, help='output file name', default=4)
    parser.add_argument('--remove_central_residue', dest='remove_central_residue', type=bool, default=0)
        

    # parser.add_argument('--num_nhs', dest='num_nhs', type=int, help='number of neighborhoods in protein set')
    
    args = parser.parse_args()
    
    if not os.path.exists("logs"):
        os.mkdir("logs")

    ds = PDBPreprocessor(args.hdf5_in, args.pdbbind_datadir)
    n = 0
    m = 0

    dt_nh = np.dtype([
        ('res_id','S5',(5)),
        ('atom_names', 'S4', (args.max_atoms)),
        ('elements', 'S1', (args.max_atoms)),
        ('res_ids', 'S5', (args.max_atoms,5)),
        ('coords', 'f8', (args.max_atoms,3)),
        ('SASAs', 'f8', (args.max_atoms)),
        ('charges', 'f8', (args.max_atoms)),
    ])
    dt_nh_info = np.dtype([
        ("missing", bool),
        ("res_id", "S5",(5)),
        ('PDB', 'S4'), 
        ('resolution', 'S4'), 
        ('year', 'S4'), 
        ('-logKd/Ki', 'S5'), 
        ('Kd/Ki', 'S11'), 
        ('ligand', 'S9')
    ])
    
    print(f'writing to {args.hdf5_out}')
    
    with h5py.File(args.hdf5_in,'r') as f:
        full_pdb_list = f["structures_bound"]["pdb"][:]
    
    empty_ind = []
    failed_ind = []
    curr_size_n = 1000
    curr_size_m = 1000
    with h5py.File(args.hdf5_out,'w') as f:
        binding_data = pd.read_csv(args.binding_data, dtype="|S").set_index("PDB", drop=False)
        
        f.create_dataset("neighborhoods_bound",
                         shape=(curr_size_n,),
                         dtype=dt_nh)
        f.create_dataset("neighborhoods_unbound",
                         shape=(curr_size_m,),
                         dtype=dt_nh)
        f.create_dataset("nh_info_bound",
                         shape=(curr_size_n,),
                         dtype= dt_nh_info)
        f.create_dataset("nh_info_unbound",
                         shape=(curr_size_m,),
                         dtype= dt_nh_info)
        
        f["nh_info_bound"]["missing"] = np.ones(curr_size_n, dtype=bool)
        f["nh_info_unbound"]["missing"] = np.ones(curr_size_m, dtype=bool)

        for i,out in tqdm( enumerate( ds.execute(
                c,
                limit = None,
                params = {"nh_radius" : args.nh_radius,
                          "max_atoms" : args.max_atoms,
                          "remove_central_residue": args.remove_central_residue},
                parallelism = args.parallelism)), total=ds.size):

            if out == (None,):
                failed_ind.append(i)
            elif out == b"":
                empty_ind.append(i)
            else:
                out_bound, out_unbound = out
                pdb = "xxxx"

                for neighborhood in out_bound:
                    
                    if n == curr_size_n:
                        curr_size_n += 1000
                        f["nh_info_bound"].resize((curr_size_n,))
                        f["neighborhoods_bound"].resize((curr_size_n,))
                        f["nh_info_bound"]["missing"] = np.ones(curr_size_n, dtype=bool)

                    pdb = neighborhood[0][1].decode()
                    f["nh_info_bound"][n] = (False, neighborhood[0], *binding_data.loc[pdb],)
                    f["neighborhoods_bound"][n] = (*neighborhood,)
                    n += 1
                
                for neighborhood in out_unbound:

                    if m == curr_size_m:
                        curr_size_m += 1000
                        f["nh_info_bound"].resize((curr_size_m,))
                        f["neighborhoods_bound"].resize((curr_size_m,))
                        f["nh_info_unbound"]["missing"] = np.ones(curr_size_m, dtype=bool)

                    pdb = neighborhood[0][1].decode()
                    f["nh_info_unbound"][m] = (False, neighborhood[0], *binding_data.loc[pdb],)
                    f["neighborhoods_unbound"][m] = (*neighborhood,)
                    m += 1
                
                print("got nh for "+pdb)
        
        f["nh_info_bound"].resize((n,))
        f["neighborhoods_bound"].resize((n,))
        f["nh_info_unbound"].resize((m,))
        f["neighborhoods_unbound"].resize((m,))
            

    print("Done with processing.")
    print("""
        The following pdbs...
        ...were absent:
        \t ind: {}
        \t pdb: {}
        ...failed:
        \t ind: {}
        \t pdb: {}
        The final neighborhood count is:
        \t {} and {}
        """.format(empty_ind, full_pdb_list[empty_ind],
                   failed_ind, full_pdb_list[failed_ind],
                   n, m)
                   )
    print("Done.")
