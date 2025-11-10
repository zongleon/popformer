# end-to-end training, and evaluation script
set -euo pipefail

run_sweep_if_missing() {
    local save_path="$1"
    shift
    if [[ -f "$save_path" ]]; then
        echo "[skip] ${save_path} exists"
    else
        mkdir -p "$(dirname "$save_path")"
        echo "[run ] Generating ${save_path}"
        python sweep.py "$@" --save_features "$save_path" \
            --subsample 50
    fi
}

# dataset 
DATASET_DIR="dataset"
TRAIN_DATASETS=(
    "selbin_pan_4_snps"
    "selbin_bigregion_snps"
)

TEST_DATASETS=(
    "ghist_const4"
    "ghist_const6"
    "len200_ghist_const1"
    "len200_ghist_const2"
    "selbin_pan_4_inorder"
)

FASTERNN_DATASET_DIR="FASTER_NN/tokenized_test_128snps/"

GHIST_EVALS=(
    "ghist_singlesweep"
    "ghist_singlesweep.growth_bg"
    "ghist_multisweep"
    "ghist_multisweep.growth_bg"
)

REAL_DATA_EVALS=(
    "genome_CEU"
    # "genome_YRI"
)

# pre-trained models
MODEL_DIR="models"
MODELS=(
    "pt"
    "pt2"
    # "pt3"
)

# training
for MODEL in "${MODELS[@]}"; do
    for TEST_DATASET in "${TEST_DATASETS[@]}"; do
        run_sweep_if_missing "${DATASET_DIR}/features/${TEST_DATASET}_${MODEL}.npz" \
            --data "${DATASET_DIR}/${TEST_DATASET}" \
            --model "${MODEL_DIR}/${MODEL}"
    done
    # generate features for GHIST eval sets
    for GHIST_EVAL in "${GHIST_EVALS[@]}"; do
        run_sweep_if_missing "${DATASET_DIR}/features/${GHIST_EVAL}_${MODEL}.npz" \
            --data "${DATASET_DIR}/${GHIST_EVAL}" \
            --model "${MODEL_DIR}/${MODEL}"
    done

    # generate features for FASTER-NN eval set
    run_sweep_if_missing "${DATASET_DIR}/features/fasternn_${MODEL}.npz" \
        --data "${FASTERNN_DATASET_DIR}" \
        --model "${MODEL_DIR}/${MODEL}"

    # generate features for real data
    for REAL_DATA_EVAL in "${REAL_DATA_EVALS[@]}"; do
        run_sweep_if_missing "${DATASET_DIR}/features/${REAL_DATA_EVAL}_${MODEL}.npz" \
            --data "${DATASET_DIR}/${REAL_DATA_EVAL}" \
            --model "${MODEL_DIR}/${MODEL}"
    done

    # train linear probes on training datasets
    for TRAIN_DATASET in "${TRAIN_DATASETS[@]}"; do
        run_sweep_if_missing "${DATASET_DIR}/features/${TRAIN_DATASET}_${MODEL}.npz" \
            --data "${DATASET_DIR}/${TRAIN_DATASET}" \
            --model "${MODEL_DIR}/${MODEL}"

    done
done