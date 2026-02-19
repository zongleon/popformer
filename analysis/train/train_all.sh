#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR="${SCRIPT_DIR}/../.."

TRAIN_NAME="pan2CEU_train"
DATASET_DIR="${ROOT_DIR}/data/dataset/${TRAIN_NAME}"
MODEL_DIR="${ROOT_DIR}/models/"

python $ROOT_DIR/sweep.py \
	--data $DATASET_DIR \
	--model $MODEL_DIR/popf-small/ \
	--subsample 64 \
	--save_features $ROOT_DIR/data/features/${TRAIN_NAME}_popf-small.npz 

for TEST_SIZE in 0.05 0.5 0.9 0.95 0.99
do
	python $ROOT_DIR/analysis/train/finetune.py \
		--mode selbin \
		--dataset-path $DATASET_DIR \
		--num-epochs 100 \
		--batch-size 2 \
		--gradient-accumulation-steps 8 \
		--output-path $MODEL_DIR/selbin-pt-sm-${TRAIN_NAME}-${TEST_SIZE} \
		--pretrained $MODEL_DIR/popf-small/ \
		--from-init \
		--test-size $TEST_SIZE

	python $ROOT_DIR/analysis/train/finetune.py \
		--mode selbin \
		--dataset-path $DATASET_DIR \
		--num-epochs 100 \
		--batch-size 2 \
		--gradient-accumulation-steps 8 \
		--output-path $MODEL_DIR/selbin-ft-sm-${TRAIN_NAME}-${TEST_SIZE} \
		--pretrained $MODEL_DIR/popf-small/ \
		--test-size $TEST_SIZE

	# if [ -d "$MODEL_DIR/selbin-lp-${TRAIN_NAME}-${TEST_SIZE}" ]; then
	#     echo "Model $MODEL_DIR/selbin-lp-${TRAIN_NAME}-${TEST_SIZE} already exists. Skipping finetuning."
	# else
	#     echo "linear probe selbin for test size $TEST_SIZE"
	# 	python $ROOT_DIR/analysis/train/finetune.py \
	# 		--mode selbin \
	# 		--dataset-path $DATASET_DIR \
	# 		--num-epochs 100 \
	# 		--batch-size 2 \
	# 		--gradient-accumulation-steps 4 \
	# 		--output-path $MODEL_DIR/selbin-lp-${TRAIN_NAME}-${TEST_SIZE} \
	# 		--pretrained $MODEL_DIR/popf-large/ \
	# 		--test-size $TEST_SIZE \
	# 		--freeze-layers-up-to 4
	# fi

	python $ROOT_DIR/analysis/train/lp.py popf-small ${TRAIN_NAME} ${TEST_SIZE}

	python $ROOT_DIR/analysis/train/fasternn.py $DATASET_DIR ${TEST_SIZE}
	python $ROOT_DIR/analysis/train/schrider_resnet.py $DATASET_DIR ${TEST_SIZE}
done
