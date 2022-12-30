
import os, sys
import argparse

from protein_holography_pytorch.training import hcnn_aa_classifier_inference
from protein_holography_pytorch.utils.argparse import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_dir', type=str, default='runs/cgnet_full_full_lmax=5__v0')
    parser.add_argument('--pdb_filepath', type=str, default='/gscratch/scrubbed/mpun/data/T4/pdbs/2LZM.pdb') # /gscratch/scrubbed/mpun/data/T4/pdbs/2LZM.pdb, /gscratch/spe/gvisan01/dms_data/AMIE_PSEAE_Whitehead/2uxy.pdb
    parser.add_argument('--output_filepath', type=str, default='runs/cgnet_full_full_lmax=5__v0/2LZM-final_model.npz')
    parser.add_argument('--model_name', type=str, default='final_model')
    args = parser.parse_args()
    
    hcnn_aa_classifier_inference(args.experiment_dir, data_filepath=args.pdb_filepath, output_filepath=args.output_filepath, model_name=args.model_name)
