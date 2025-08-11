#!/bin/bash
#SBATCH --job-name=PPGNN
#SBATCH --account=synet

# GPU
#SBATCH --partition=gpu
#SBATCH --qos=gpushort
#SBATCH --gres=gpu:1

# resources
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G

# paths
#SBATCH --chdir=/p/tmp/junyouzh/Projects/PPGNN
#SBATCH --output=/p/tmp/junyouzh/Projects/PPGNN/Junyou_HPC_Outputs/Print_out/%x-%j.out
#SBATCH --error=/p/tmp/junyouzh/Projects/PPGNN/Junyou_HPC_Outputs/Print_out/%x-%j.err

set -euo pipefail

# load envs

# module load anaconda
source activate PPGNN 

# optional：print GPU/env info
nvidia-smi || true
python --version

python main.py
