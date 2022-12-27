#!/bin/bash
#SBATCH --job-name=so3_convnet_lmax=6__v1
#SBATCH --account=stf
#SBATCH --partition=gpu-2080ti
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=1-00:00:00
#SBATCH --mem=82G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=gvisan01@uw.edu
#SBATCH --export=all
#SBATCH -e slurm_logs/so3_convnet_lmax=6__v1.err
#SBATCH -o slurm_logs/so3_convnet_lmax=6__v1.out

python launch_training__hcnn_aa_classifier.py --experiment_dir runs/so3_convnet_lmax=6__v1 --config ../config/hcnn_aa_classifier/so3_convnet.yaml
