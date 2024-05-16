#!/bin/bash
#SBATCH --mail-type=BEGIN,END,FAIL --mem 50G

# -p gpu --gres=gpu:rtx3080:1
module load python3.11

echo "Starting job $SLURM_JOB_ID"

python -u run_tcnAE.py --alias AAPL --ticker AAPL --epochs 4000

echo "Job $SLURM_JOB_ID done"

