#!/bin/bash
#SBATCH --job-name=atk_bb
#SBATCH --ntasks=1
#SBATCH --mem 64G
#SBATCH -c 32
#SBATCH -o job.log
#SBATCH --output=job_output_locals.txt
#SBATCH --error=job_error_locals.txt

module load Python/3.10

source $HOME/textbugger_env/bin/activate

python $HOME/textbugger/attacks/run_locals.py