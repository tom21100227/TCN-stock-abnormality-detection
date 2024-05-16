#!/bin/bash
#SBATCH --mail-type=BEGIN,END,FAIL --mem 200G

# -p gpu --gres=gpu:rtx3080:1
module load python3.11

ALIAS=$1
TICKER=$2
EPOCHS=$3

echo "Starting job $SLURM_JOB_ID"

echo "python -u run_tcnAE.py --alias $ALIAS --ticker $TICKER --epochs $EPOCHS"

python -u run_tcnAE.py --alias "$ALIAS" --ticker "$TICKER" --epochs "$EPOCHS"

echo "Job $SLURM_JOB_ID done"

