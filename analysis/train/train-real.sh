#!/usr/bin/env bash
#SBATCH --job-name=popf-rev             # Name of the job
#SBATCH --output=logs/%x_%j.out         # Stdout goes to logs/jobname_jobid.out
#SBATCH --error=logs/%x_%j.err          # Stderr goes to logs/jobname_jobid.err
#SBATCH --partition=dgx-b200	        # Queue to submit to
#SBATCH --ntasks=1                      # Number of tasks (usually one per process)
#SBATCH --cpus-per-task=4               # Number of CPU cores per task
#SBATCH --mem=32G                       # Memory allocation
#SBATCH --gpus=4
#SBATCH --time=12:00:00                 # Maximum runtime (hh:mm:ss)

set -euo pipefail

# pre-train on 1000G dataset
torchrun --nproc_per_node=4 analysis/train/train.py \
    --dataset_path ./data/dataset/pt \
    --configuration popformer-base \
    --output_path ./models/popf-base-real \
    --mlm_probability 0.75 \
    --span_mask_probability 0 \
    --num_epochs 5 \
    --batch_size 8 \
    --learning_rate 0.00015

# train models on simulated CEU
python analysis/train/finetune.py \
    --mode selbin \
    --dataset_path ./data/dataset/CEU_train \
    --test_size 0.05 \
    --num_epochs 10 \
    --batch_size 2 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-4 \
    --pretrained ./models/popf-base-real \
    --output_path ./models/selbin-popf-base-real-CEU_train

python analysis/train/finetune.py \
    --mode selbin \
    --dataset_path ./data/dataset/CEU_train \
    --test_size 0.05 \
    --num_epochs 10 \
    --batch_size 2 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-4 \
    --pretrained ./models/popf-init \
    --output_path ./models/selbin-popf-init-CEU_train

python sweep.py \
    --model ./models/popf-base-real \
    --data ./data/dataset/CEU_test \
    --save_features ./features/popf-base-real__CEU_train.npz \
    --subsample 64

python analysis/train/lp.py \
    ./features/popf-base-real__CEU_train.npz \
    ./data/dataset/CEU_train \
    0.05 # test size

python analysis/train/schrider_resnet.py \
    ./data/dataset/CEU_train \
    0.05 # test size

python analysis/train/fasternn.py \
    ./data/dataset/CEU_train \
    0.05 # test size
