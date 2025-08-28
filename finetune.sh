#!/bin/bash
#SBATCH --job-name=hapbertaft           # Name of the job
#SBATCH --output=logs/%x_%j.out         # Stdout goes to logs/jobname_jobid.out
#SBATCH --error=logs/%x_%j.err          # Stderr goes to logs/jobname_jobid.err
#SBATCH --partition=dgx-b200	        # Queue to submit to
#SBATCH --ntasks=1                      # Number of tasks (usually one per process)
#SBATCH --cpus-per-task=4               # Number of CPU cores per task
#SBATCH --mem=128G                       # Memory allocation
#SBATCH --gpus=4
#SBATCH --time=5:00:00                  # Maximum runtime (hh:mm:ss)
#SBATCH --exclude=dgx002,dgx018

torchrun --nproc_per_node=4 finetune.py

