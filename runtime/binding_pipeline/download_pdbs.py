from argparse import ArgumentParser
import os, sys
import gzip, pickle
import shutil
from pandas import read_csv
from Bio.PDB import PDBList
from tqdm import tqdm

#####################
####### Input #######

parser = ArgumentParser()

parser.add_argument(
    '--pdb_dir',
    dest='pdb_dir',
    type=str,
    help='Directory for pdb files'
)
parser.add_argument(
    '--pdb_list',
    dest='pdb_list',
    type=str,
    help='List of pdbs'
)

args = parser.parse_args()

#####################

if not os.path.exists("logs"):
        os.mkdir("logs")


pdbs = read_csv(args.pdb_list)["PDB"].to_list()

print("downloading pdbs")

pdbL = PDBList(verbose=False)
pdbL.download_pdb_files( pdbs, overwrite=False, pdir=args.pdb_dir, file_format="pdb")

print("renaming pdbs to .pdb")

missing = 0
for pdb in tqdm(pdbs, total=len(pdbs)):
    pdbfile = os.path.join(args.pdb_dir, pdb + ".pdb")
    entfile = os.path.join( args.pdb_dir, "pdb" + pdb + ".ent")
    if not os.path.exists( pdbfile ):
        if os.path.exists( entfile ):
            shutil.copy( entfile, pdbfile)
        elif os.path.exists( entfile + ".gz" ):
            print("got .gz for " + pdb)
            with gzip.open(entfile + ".gz", 'rb') as f_in:
                with open(pdbfile, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
        else:
            missing += 1
            print(pdb + " missing.", file=sys.stderr)

print("{0} out of {1} failed to download.".format(missing, len(pdbs)) )
print("Done.")