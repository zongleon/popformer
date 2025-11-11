#!/bin/bash
#SBATCH --job-name=popf-base            # Name of the job
#SBATCH --output=logs/%x_%j.out         # Stdout goes to logs/jobname_jobid.out
#SBATCH --error=logs/%x_%j.err          # Stderr goes to logs/jobname_jobid.err
#SBATCH --partition=dgx-b200	        # Queue to submit to
#SBATCH --ntasks=1                      # Number of tasks (usually one per process)
#SBATCH --cpus-per-task=4               # Number of CPU cores per task
#SBATCH --mem=32G                       # Memory allocation
#SBATCH --gpus=4
#SBATCH --time=12:00:00                  # Maximum runtime (hh:mm:ss)

torchrun --nproc_per_node=4 analysis/train.py \
    --configuration popformer-base \
    --dataset_path ./dataset/pt_tokenized \
    --mlm_probability 0.7 \
    --num_epochs 5 \
    --batch_size 8 \
    --output_path ./models/popf-base \
    --learning_rate 1.5e-4
