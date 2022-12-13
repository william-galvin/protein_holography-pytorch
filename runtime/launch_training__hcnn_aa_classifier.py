
import os
import yaml
import json
import argparse

from protein_holography_pytorch.training import hcnn_aa_classifier_training, hcnn_aa_classifier_inference
from protein_holography_pytorch.utils.argparse import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_dir', type=str, default='/gscratch/spe/gvisan01/holographic_vae/cgnet_classification/runs/cgnet___IVF_mikes_params_with_silu')
    parser.add_argument('--config_file', type=optional_str, default='config/hcnn_aa_classifier/cgnet.yaml')
    args = parser.parse_args()


    # make directory if it does not already exist
    if not os.path.exists(args.experiment_dir):
        os.makedirs(args.experiment_dir)
    
    # load config if requested, if config is None, then use hparams within experiment_dir
    if args.config_file is not None:
        with open(args.config_file, 'r') as f:
            hparams = yaml.load(f, Loader=yaml.FullLoader)

        # save hparams as json file within expriment_dir
        with open(os.path.join(args.experiment_dir, 'hparams.json'), 'w+') as f:
            json.dump(hparams, f, indent=2)
        
    else:
        with open(os.path.join(args.experiment_dir, 'hparams.json'), 'r') as f:
            hparams = json.dump(f)

    # launch training script
    hcnn_aa_classifier_training(args.experiment_dir)

    # perform inference, on test data, with basic results
    hcnn_aa_classifier_inference(args.experiment_dir)



