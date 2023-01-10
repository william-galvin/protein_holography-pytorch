

pdb_dir="/gscratch/spe/gvisan01/pdbbind/data/pdbs/"
pdbbind_dir="/gscratch/spe/gvisan01/pdbbind/data/"

tar xzf /gscratch/spe/PDBbind/PDBbind_v2020_plain_text_index.tar.gz -C $pdbbind_dir
tar xzf /gscratch/spe/PDBbind/PDBbind_v2020_other_PL.tar.gz -C $pdbbind_dir
tar xzf /gscratch/spe/PDBbind/PDBbind_v2020_refined.tar.gz -C $pdbbind_dir
tar xzf /gscratch/spe/PDBbind/PDBbind_v2020_PN.tar.gz -C $pdbbind_dir

python "download_pdbs.py" --pdb_dir $pdb_dir --pdb_list $pdbbind_dir"refined-set_data.csv"

