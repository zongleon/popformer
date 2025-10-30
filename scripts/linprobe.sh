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
        python sweep.py "$@" --save_features "$save_path"
    fi
}

# dataset 
DATASET_DIR="dataset"
TRAIN_DATASETS=(
    "selbin_disc_pg_gan"
    "selbin_pan_2"
    "selbin_pan_3"
    "selbin_bigregion"
)

FASTERNN_DATASET_DIR="FASTER_NN/tokenized_test_50000/"

GHIST_EVAL_DIR="GHIST/samples_"
GHIST_EVALS=(
    "singlesweep"
    "singlesweep.growth_bg"
    "multisweep"
    "multisweep.growth_bg"
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
    "pt3"
)

# training
for MODEL in "${MODELS[@]}"; do
    # generate features for GHIST eval sets
    for GHIST_EVAL in "${GHIST_EVALS[@]}"; do
        run_sweep_if_missing "${DATASET_DIR}/features/ghist_${GHIST_EVAL}_${MODEL}.npz" \
            --data "${GHIST_EVAL_DIR}${GHIST_EVAL}" \
            --model "${MODEL_DIR}/${MODEL}"
            # --subsample 32
    done

    # generate features for FASTER-NN eval set
    run_sweep_if_missing "${DATASET_DIR}/features/fasternn_${MODEL}.npz" \
        --data "${FASTERNN_DATASET_DIR}" \
        --model "${MODEL_DIR}/${MODEL}"
        # --subsample 32

    # generate features for real data
    for REAL_DATA_EVAL in "${REAL_DATA_EVALS[@]}"; do
        run_sweep_if_missing "${DATASET_DIR}/features/${REAL_DATA_EVAL}_${MODEL}.npz" \
            --data "${DATASET_DIR}/${REAL_DATA_EVAL}" \
            --model "${MODEL_DIR}/${MODEL}"
            # --subsample 32
    done

    # train linear probes on training datasets
    for TRAIN_DATASET in "${TRAIN_DATASETS[@]}"; do
        echo "=== Training linear probe on ${TRAIN_DATASET} with model ${MODEL} ==="
        run_sweep_if_missing "${DATASET_DIR}/features/${TRAIN_DATASET}_${MODEL}.npz" \
            --data "${DATASET_DIR}/${TRAIN_DATASET}" \
            --model "${MODEL_DIR}/${MODEL}"
            # --subsample 32

        # combine all eval sets
        EVALS=()
        OUTS=()
        # make outs by combining eval set name and model name
        for GHIST_EVAL in "${GHIST_EVALS[@]}"; do
            EVALS+=("${DATASET_DIR}/features/ghist_${GHIST_EVAL}_${MODEL}.npz")
            OUTS+=("outs/ghist_${GHIST_EVAL}_${MODEL}_${TRAIN_DATASET}.npz")
        done
        for REAL_DATA_EVAL in "${REAL_DATA_EVALS[@]}"; do
            EVALS+=("${DATASET_DIR}/features/${REAL_DATA_EVAL}_${MODEL}.npz")
            OUTS+=("outs/${REAL_DATA_EVAL}_${MODEL}_${TRAIN_DATASET}.npz")
        done

        python -u lp.py --train-features "${DATASET_DIR}/features/${TRAIN_DATASET}_${MODEL}.npz" \
            --train-labels "${DATASET_DIR}/${TRAIN_DATASET}/" \
            --test-features "${DATASET_DIR}/features/fasternn_${MODEL}.npz" \
            --test-metadata FASTER_NN/fasternn_test_meta.csv \
            --grid \
            --predict-features "${EVALS[@]}" \
            --predict-outputs "${OUTS[@]}"

        # and plot
        python sweep-ghist.py outs/ghist_{t}_${MODEL}_${TRAIN_DATASET}.npz \
            GHIST/figs/${MODEL}_${TRAIN_DATASET}_{t}.png

        for REAL_DATA_EVAL in "${REAL_DATA_EVALS[@]}"; do
            python sweep.py --logits_path outs/${REAL_DATA_EVAL}_${MODEL}_${TRAIN_DATASET}.npz \
                --plot_preds SEL/${REAL_DATA_EVAL}_${MODEL}_${TRAIN_DATASET}.png
        done

    done
done