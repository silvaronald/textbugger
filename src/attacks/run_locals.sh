#!/bin/bash
#SBATCH --job-name=textbugger_local
#SBATCH --ntasks=1
#SBATCH --mem 64G
#SBATCH -c 32
#SBATCH -o logs/job.log
#SBATCH --output=logs/job_output_locals.txt
#SBATCH --error=logs/job_error_locals.txt

module load Python/3.10

source $HOME/textbugger_env/bin/activate

# Run local model attacks using new unified script
python $HOME/textbugger/scripts/run_attacks.py --target local --dataset rtmr