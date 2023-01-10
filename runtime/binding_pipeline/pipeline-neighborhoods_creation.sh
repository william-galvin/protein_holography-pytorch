
set=$1 # first argument is the desired split (train/val/test)

pdb_dir="/gscratch/spe/gvisan01/pdbbind/data/pdbs/"
pdbbind_dir="/gscratch/spe/gvisan01/pdbbind/data/"
output_dir="/gscratch/spe/gvisan01/pdbbind/data/processed/"


## Preprocessing of pdbs into pyrosetta structures
# Given a protein ligand pair, parse structure file from PDB as well as unbound biological unit
# from PDBbind into PyRosetta poses. Parse Ligand file and use coordinates to find and slice
# Ligand residues out of the PDB pose. Appending it to a copy of the unbound pose gives a bound
# pose that is directly comparable to the unbound one. For both poses PyRosetta then calculates
# atomic charge and SASA and infers Hydrogens.

# python "get_structural_info_binding.py" \
#     --hdf5_out $output_dir$set"_refined_set.hdf5" \
#     --pdb_list $set"/pdbs" \
#     --binding_dataset_name $set"/binding_data" \
#     --parallelism 40 \
#     --hdf5_in $output_dir"pdb_lists_refined_set.hdf5" \
#     --pdb_dir $pdb_dir --pdbbind_dir $pdbbind_dir"refined-set"

echo "---------- Ligand check ----------"

## Ligand check
# Count atoms in a Ligand's sdf file and compare to recovered atoms in the bound structure.
# Inferred H generally differ, only keep ligands where all non-H atoms have been successfully
# processed by PyRosetta.
# The resulting list is used after processing to filter and contains 2300, 300, 300 pdbs in
# training, validation, and test sets respectively

python "get_ligand_parsing_check.py" \
    --hdf5_out $output_dir"ligand_check_"$set"_refined_set.hdf5" \
    --parallelism 40 \
    --hdf5_in $output_dir$set"_refined_set.hdf5" \
    --pdbbind_datadir $pdbbind_dir"refined-set"


## 2. Neighborhoods from Binding Pocket
# For both bound and unbound structures: Find all residues that contain atoms in the 10A binding
# pocket around the Ligand.
# For each pocket-residue get 10A neighborhood of atoms. Neighborhoods of all proteins are concatenated.
# Do this twice:
#   - while removing central residue (H-CNN is trained to classify such vacancies in a blind-folded, self-supervised manner)
#   - without removing central residue (H-(V)AE is trained with the full neighborhood

echo "---------- Neighborhoods without central AA ----------"

python "get_neighborhoods_binding.py" \
    --hdf5_out $output_dir"neighborhoods_"$set"_refined_set-remove_central_residue=False.hdf5" \
    --hdf5_in $output_dir$set"_refined_set.hdf5" \
    --pdbbind_datadir $pdbbind_dir"refined-set/" \
    --binding_data $pdbbind_dir"refined-set_data.csv" \
    --parallelism 40 \
    --nh_radius 10. --max_atoms 1000 \
    --remove_central_residue 0


echo "---------- Neighborhoods with central AA ----------"

python "get_neighborhoods_binding.py" \
    --hdf5_out $output_dir"neighborhoods_"$set"_refined_set-remove_central_residue=True.hdf5" \
    --hdf5_in $output_dir$set"_refined_set.hdf5" \
    --pdbbind_datadir $pdbbind_dir"refined-set/" \
    --binding_data $pdbbind_dir"refined-set_data.csv" \
    --parallelism 40 \
    --nh_radius 10. --max_atoms 1000 \
    --remove_central_residue 1
