#!/bin/bash
#SBATCH --job-name=atk_wb
#SBATCH --ntasks=1
#SBATCH --mem 64G
#SBATCH -c 32
#SBATCH -o job.log
#SBATCH --output=job_output.txt
#SBATCH --error=job_error.txt

module load Python/3.10

source $HOME/textbugger_env/bin/activate

python $HOME/textbugger/attacks/whitebox/run.py