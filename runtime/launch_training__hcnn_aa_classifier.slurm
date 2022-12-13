#!/bin/bash
#SBATCH --job-name=cgnet_full_full_lmax=5__v0
#SBATCH --account=spe
#SBATCH --partition=gpu-a40
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=12:00:00
#SBATCH --mem=96G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=gvisan01@uw.edu
#SBATCH --export=all
#SBATCH -e slurm_logs/cgnet_full_full_lmax=5__v0.err
#SBATCH -o slurm_logs/cgnet_full_full_lmax=5__v0.out

python launch_training__hcnn_aa_classifier.py --experiment_dir runs/cgnet_full_full_lmax=5__v0 --config ../config/hcnn_aa_classifier/cgnet.yaml
