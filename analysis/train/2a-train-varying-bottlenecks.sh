#!/usr/bin/env bash

set -euo pipefail

PRETRAIN_DATASET=discoal_bottlenecks_1000
PRETRAIN_DATASET_PATH=data/dataset/$PRETRAIN_DATASET
DATASET_NAME=discoal_consts_10000
DATASET_PATH=data/dataset/$DATASET_NAME

# pretrain
PRETRAINED_MODEL=popf-base-$PRETRAIN_DATASET
PRETRAINED_MODEL_PATH=./models/$PRETRAINED_MODEL
python analysis/train/train.py \
    --dataset-path $PRETRAIN_DATASET_PATH \
    --configuration popformer-base \
    --output-path $PRETRAINED_MODEL_PATH \
    --mlm-probability 0.75 \
    --span-mask-probability 0 \
    --num-epochs 5 \
    --batch-size 2 \
    --gradient-accumulation-steps 4 \
    --learning-rate 0.00015

# train models on dataset
# finetunes
python analysis/train/finetune.py \
    --mode selbin \
    --dataset-path $DATASET_PATH \
    --test-size 0.05 \
    --num-epochs 10 \
    --batch-size 2 \
    --gradient-accumulation-steps 4 \
    --learning-rate 1e-4 \
    --pretrained $PRETRAINED_MODEL_PATH \
    --output-path ./models/selbin-${PRETRAINED_MODEL}-${DATASET_NAME}

python analysis/train/finetune.py \
    --mode selbin \
    --dataset-path $DATASET_PATH \
    --test-size 0.05 \
    --num-epochs 10 \
    --batch-size 2 \
    --gradient-accumulation-steps 4 \
    --learning-rate 1e-4 \
    --pretrained ./models/popf-init \
    --output-path ./models/selbin-popf-init-${DATASET_NAME}

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
