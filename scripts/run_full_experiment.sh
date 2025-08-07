#!/bin/bash
#SBATCH --job-name=textbugger_full
#SBATCH --ntasks=1
#SBATCH --mem 64G
#SBATCH -c 32
#SBATCH -o logs/job_full.log
#SBATCH --output=logs/job_output_full.txt
#SBATCH --error=logs/job_error_full.txt
#SBATCH --time=04:00:00

module load Python/3.10

source $HOME/textbugger_env/bin/activate

echo "Starting TextBugger Full Experiment"
echo "==================================="

# Run API attacks first (faster, fewer resources)
echo "Starting API attacks..."
python $HOME/textbugger/scripts/run_attacks.py --target api --dataset rtmr --limit 20

echo "API attacks completed. Starting local model attacks..."

# Run local model attacks
python $HOME/textbugger/scripts/run_attacks.py --target local --dataset rtmr --limit 50

echo "Full experiment completed!"