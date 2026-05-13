#!/usr/bin/env bash
#SBATCH --job-name=popf-3               # Name of the job
#SBATCH --output=logs/%x_%j.out         # Stdout goes to logs/jobname_jobid.out
#SBATCH --error=logs/%x_%j.err          # Stderr goes to logs/jobname_jobid.err
#SBATCH --partition=dgx-b200	        # Queue to submit to
#SBATCH --ntasks=1                      # Number of tasks (usually one per process)
#SBATCH --cpus-per-task=1               # Number of CPU cores per task
#SBATCH --mem=32G                       # Memory allocation
#SBATCH --gpus=1
#SBATCH --time=1-00:00:00               # Maximum runtime (hh:mm:ss)

set -euo pipefail

DATASET_NAME=discoal_consts_10000
DATASET_PATH=data/dataset/$DATASET_NAME

# pretrain
PRETRAINED_MODEL=popf-base-$DATASET_NAME
PRETRAINED_MODEL_PATH=./models/$PRETRAINED_MODEL
python analysis/train/train.py \
    --dataset_path $DATASET_PATH \
    --configuration popformer-base \
    --output_path $PRETRAINED_MODEL_PATH \
    --mlm_probability 0.75 \
    --span_mask_probability 0 \
    --num_epochs 5 \
    --batch_size 2 \
    --gradient_accumulation_steps 4 \
    --learning_rate 0.00015

mkdir -p features
FEATURES_PATH=./features/${PRETRAINED_MODEL}__${DATASET_NAME}.npz

python sweep.py \
    --data $DATASET_PATH \
    --model $PRETRAINED_MODEL_PATH \
    --save_features $FEATURES_PATH \
    --subsample 64

# train models on dataset
for test_size in 0.05 0.5 0.9 0.95 0.99 0.995; do
    python analysis/train/finetune.py \
        --mode selbin \
        --dataset_path $DATASET_PATH \
        --test_size $test_size \
        --num_epochs 10 \
        --batch_size 2 \
        --gradient_accumulation_steps 4 \
        --learning_rate 1e-4 \
        --pretrained $PRETRAINED_MODEL_PATH \
        --output_path ./models/selbin-${PRETRAINED_MODEL}-${DATASET_NAME}-$test_size

    python analysis/train/finetune.py \
        --mode selbin \
        --dataset_path $DATASET_PATH \
        --test_size $test_size \
        --num_epochs 10 \
        --batch_size 2 \
        --gradient_accumulation_steps 4 \
        --learning_rate 1e-4 \
        --pretrained ./models/popf-init \
        --output_path ./models/selbin-popf-init-${DATASET_NAME}-$test_size

    python analysis/train/lp.py \
        $FEATURES_PATH \
        $DATASET_PATH \
        $test_size

    python analysis/train/schrider_resnet.py \
        $DATASET_PATH \
        test_size

    python analysis/train/fasternn.py \
        $DATASET_PATH \
        test_size

done
