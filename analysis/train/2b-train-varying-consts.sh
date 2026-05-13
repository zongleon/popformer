#!/usr/bin/env bash

set -euo pipefail

PRETRAIN_DATASET=discoal_consts_10000
PRETRAIN_DATASET_PATH=data/dataset/$PRETRAIN_DATASET
DATASET_NAME=discoal_consts_1000
DATASET_PATH=data/dataset/$DATASET_NAME

# pretrain
PRETRAINED_MODEL=popf-base-$PRETRAIN_DATASET
PRETRAINED_MODEL_PATH=./models/$PRETRAINED_MODEL
python analysis/train/train.py \
    --dataset_path $PRETRAIN_DATASET_PATH \
    --configuration popformer-base \
    --output_path $PRETRAINED_MODEL_PATH \
    --mlm_probability 0.75 \
    --span_mask_probability 0 \
    --num_epochs 5 \
    --batch_size 2 \
    --gradient_accumulation_steps 4 \
    --learning_rate 0.00015

# train models on dataset
# finetunes
python analysis/train/finetune.py \
    --mode selbin \
    --dataset_path $DATASET_PATH \
    --test_size 0.05 \
    --num_epochs 10 \
    --batch_size 2 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-4 \
    --pretrained $PRETRAINED_MODEL_PATH \
    --output_path ./models/selbin-${PRETRAINED_MODEL}-${DATASET_NAME}

python analysis/train/finetune.py \
    --mode selbin \
    --dataset_path $DATASET_PATH \
    --test_size 0.05 \
    --num_epochs 10 \
    --batch_size 2 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-4 \
    --pretrained ./models/popf-init \
    --output_path ./models/selbin-popf-init-${DATASET_NAME}

# linear probe
FEATURES_PATH=./features/${PRETRAINED_MODEL}__${DATASET_NAME}.npz

python sweep.py \
    --data $DATASET_PATH \
    --model $PRETRAINED_MODEL_PATH \
    --save_features $FEATURES_PATH \
    --subsample 64

python analysis/train/lp.py \
    $FEATURES_PATH \
    $DATASET_PATH \
    0.05

python analysis/train/schrider_resnet.py \
    $DATASET_PATH \
    0.05 # test size

python analysis/train/fasternn.py \
    $DATASET_PATH \
    0.05 # test size
