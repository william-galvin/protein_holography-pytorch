
import os, sys
import argparse

from protein_holography_pytorch.training import hcnn_aa_classifier_inference
from protein_holography_pytorch.utils.argparse import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_dir', type=str, default='runs/so3_convnet_lmax=6__v1')
    parser.add_argument('--pdb_filepath', type=str, default='/gscratch/scrubbed/mpun/data/T4/pdbs/2LZM.pdb')
    parser.add_argument('--output_filepath', type=str, default='runs/so3_convnet_lmax=6__v1/2LZM-lowest_valid_loss_model.npz')
    parser.add_argument('--model_name', type=str, default='lowest_valid_loss_model')
    args = parser.parse_args()
    
    hcnn_aa_classifier_inference(args.experiment_dir, data_filepath=args.pdb_filepath, output_filepath=args.output_filepath, model_name=args.model_name)
