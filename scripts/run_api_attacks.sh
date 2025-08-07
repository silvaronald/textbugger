#!/bin/bash
#SBATCH --job-name=textbugger_api
#SBATCH --ntasks=1
#SBATCH --mem 32G
#SBATCH -c 16
#SBATCH -o logs/job_api.log
#SBATCH --output=logs/job_output_api.txt
#SBATCH --error=logs/job_error_api.txt
#SBATCH --time=02:00:00

module load Python/3.10

source $HOME/textbugger_env/bin/activate

# Run API attacks using new unified script
# Customize --limit and --dataset as needed
python $HOME/textbugger/scripts/run_attacks.py --target api --dataset rtmr --limit 10