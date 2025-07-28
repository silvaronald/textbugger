#!/bin/bash
#SBATCH -p short
#SBATCH --job-name=train_rottentomatoes
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=64
#SBATCH -o job_output.txt
#SBATCH -e job_error.txt
# Load Python and CUDA modules
module load Python/3.10

# Activate pre-created virtual environment
nividia-smi
source $HOME/textbugger_env/bin/activate

# Run training script
python3 $HOME/textbugger/training/rtmr/train.py
