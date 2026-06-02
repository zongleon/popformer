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

TRAIN_DATASET=pan2_train
TRAIN_DATASET_PATH=./data/dataset/$TRAIN_DATASET

# pre-train on 1000G dataset
# if slurm use torchrun, else just python
if [ -v SLURM_JOB_ID ]; then
    cmd="torchrun --nproc_per_node=4"
else
    cmd="python"
fi
$cmd analysis/train/train.py \
    --dataset-path ./data/dataset/pt \
    --configuration popformer-base \
    --output-path ./models/popf-base-real \
    --mlm-probability 0.75 \
    --span-mask-probability 0 \
    --num-epochs 5 \
    --batch-size 8 \
    --learning-rate 0.00015

# train models on simulated CEU
python analysis/train/finetune.py \
    --mode selbin \
    --dataset-path $TRAIN_DATASET_PATH \
    --test-size 0.05 \
    --num-epochs 10 \
    --batch-size 2 \
    --gradient-accumulation-steps 4 \
    --learning-rate 1e-4 \
    --pretrained ./models/popf-base-real \
    --output-path ./models/selbin-popf-base-real-$TRAIN_DATASET

python analysis/train/finetune.py \
    --mode selbin \
    --dataset-path $TRAIN_DATASET_PATH \
    --test-size 0.05 \
    --num-epochs 10 \
    --batch-size 2 \
    --gradient-accumulation-steps 4 \
    --learning-rate 1e-4 \
    --pretrained ./models/popf-init \
    --output-path ./models/selbin-popf-init-$TRAIN_DATASET

python sweep.py \
    --model ./models/popf-base-real \
    --data $TRAIN_DATASET_PATH \
    --save_features ./features/popf-base-real__$TRAIN_DATASET.npz \
    --subsample 64

python analysis/train/lp.py \
    ./features/popf-base-real__$TRAIN_DATASET.npz \
    $TRAIN_DATASET_PATH \
    0.05 # test size

python analysis/train/schrider_resnet.py \
    $TRAIN_DATASET_PATH \
    0.05 # test size

python analysis/train/fasternn.py \
    $TRAIN_DATASET_PATH \
    0.05 # test size
