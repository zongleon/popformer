#!/bin/bash
#SBATCH --job-name=ftselreg             # Name of the job
#SBATCH --output=logs/%x_%j.out         # Stdout goes to logs/jobname_jobid.out
#SBATCH --error=logs/%x_%j.err          # Stderr goes to logs/jobname_jobid.err
#SBATCH --partition=dgx-b200	        # Queue to submit to
#SBATCH --ntasks=1                      # Number of tasks (usually one per process)
#SBATCH --cpus-per-task=4               # Number of CPU cores per task
#SBATCH --mem=32G                       # Memory allocation
#SBATCH --gpus=4
#SBATCH --time=4:00:00                  # Maximum runtime (hh:mm:ss)
#SBATCH --exclude=dgx002,dgx018

torchrun --nproc_per_node=4 finetune.py \
    --mode sel \
    --dataset_path ./dataset/ft_selreg_snpwindow_tkns \
    --num_epochs 10 \
    --batch_size 8 \
    --output_path ./models/ft_sel_reg \
    --learning_rate 4e-5
