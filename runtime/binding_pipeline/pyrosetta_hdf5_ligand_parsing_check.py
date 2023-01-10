import numpy as np
from sklearn.neighbors import KDTree

def get_ligand_atom_counts(np_protein, ligand):
    
    pdb = np_protein["pdb"]
    
    real_locs = np_protein['atom_names'] != b''
    
    coords = np_protein['coords'][real_locs]
    res_ids = np_protein['res_ids'][real_locs]
    atom_names = np_protein['atom_names'][real_locs].astype(str)
    elements = np_protein['elements'][real_locs].astype(str)
    
    """
        Find ligand atoms
    """
    tree = KDTree(coords,leaf_size=2)
    
    ligand_locs = np.concatenate( 
        tree.query_radius(ligand.df[["x","y","z"]].values, r=0.01, return_distance=False) 
    )
    
    num_ligand_atoms = len(ligand.df)
    num_ligand_atoms_noH = np.sum(  ligand.df["atom_type"] != "H"  )
    num_ligand_subst = np.sum( ligand.df["atom_name"] == "CA" )
    if num_ligand_subst == 0:
        num_ligand_subst = 1
    
    num_parsed_atoms = len(ligand_locs)
    num_parsed_atoms_noH = np.sum( elements[ligand_locs] != "H" )
    num_parsed_res = len(np.unique( res_ids[ligand_locs] ) )
    
    size_intersect = len(np.intersect1d( ligand.df["atom_name"], atom_names[ligand_locs]) )
    size_intersect_noH = len(np.intersect1d( ligand.df["atom_name"][ligand.df["atom_type"] != "H"], 
                                             atom_names[ligand_locs][elements[ligand_locs] != "H"]) )
    
    return ( pdb,
            num_ligand_atoms, num_ligand_atoms_noH, num_ligand_subst,
            num_parsed_atoms, num_parsed_atoms_noH, num_parsed_res,
            size_intersect, size_intersect_noH, )
    
    
    
