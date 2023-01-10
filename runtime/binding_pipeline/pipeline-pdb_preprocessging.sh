'''
    Code adapted from Arman Angaji''s code
'''

pdb_dir=".../data/pdbs/"
pdbbind_dir=".../data/pdbbind/"
output_dir=".../data/processing/"

## Download and prepare pdbs from PDBBind

python "download_pdbs.py" --pdb_dir $pdb_dir --pdb_list $pdbbind_dir"refined-set_data.csv"

# fail pyrosetta
rm $pdb_dir*2bmk*
rm $pdb_dir*1hps*
rm $pdb_dir*6mnc*
rm $pdb_dir*3eft*
rm $pdb_dir*2za5*
rm $pdb_dir*2jfz*
rm $pdb_dir*1mfd*
rm $pdb_dir*4bak*
rm $pdb_dir*4i71*

python "protein_list.py" \
    --pdb_list $pdbbind_dir"refined-set_data.csv" \
    --pdb_dir $pdb_dir \
    --filename "refined_set" \
    --data_dir $output_dir

python "pdb_metadata_parallel.py" \
    --pdb_dir $pdb_dir \
    --pdb_list "refined_set" \
    --parallelism 40

python "protein_selection.py" \
    --pdb_list_file $output_file \
    --resolution 100. --image_type "x-ray diffraction"

python "train_test_split.py" \
    --pdb_list $output_dir"refined_set.hdf5" \
    --resolution 100. --image_type "x-ray diffraction" \
    --train_frac 0.8 --val_frac 0.1 --test_frac 0.1

split_fullname="img=x-ray diffraction_max_res=100.0/split_0.8_0.1_0.1"

python "write_pdb_binding_list.py" \
    --hdf5_out $output_dir"pdb_lists_refined_set.hdf5" \
    --hdf5_in $output_dir"refined_set.hdf5" \
    --split_fullname "$split_fullname" \
    --binding_data $pdbbind_dir"refined-set_data.csv"
