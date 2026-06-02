#!/usr/bin/env bash

set -euo pipefail

# 1. evaluate real data

TRAINED_ON=pan2_train
python analysis/test_selection.py \
    --models models/lp/popf-base-real__${TRAINED_ON}-0.05.pkl \
             models/selbin-popf-init-${TRAINED_ON}/ \
             models/selbin-popf-base-real-${TRAINED_ON}/ \
             models/fasternn/fasternn_${TRAINED_ON}-0.05.pt \
             models/schrider_resnet/resnet_${TRAINED_ON}-0.05.pt \
    --names popformer-lp \
            popformer-no-pretrain \
            popformer-ft \
            FASTER-NN \
            resnet34 \
    --datasets data/dataset/pan2CEU_test \
               data/dataset/pan2CHB_test \
               data/dataset/pan2YRI_test \
    --rocs


TRAINED_ON=pan2_train
python analysis/test_selection_real.py \
    --models models/lp/popf-base-real__${TRAINED_ON}-0.05.pkl \
             models/selbin-popf-init-${TRAINED_ON}/ \
             models/selbin-popf-base-real-${TRAINED_ON}/ \
             models/fasternn/fasternn_${TRAINED_ON}-0.05.pt \
             models/schrider_resnet/resnet_${TRAINED_ON}-0.05.pt \
    --names popformer-lp \
            popformer-no-pretrain \
            popformer-ft \
            FASTER-NN \
            resnet34

# 2. evaluate pretraining using varying simulations

python analysis/test_selection.py \
    --models models/selbin-popf-init-discoal_consts_10000/ \
             models/selbin-popf-base-discoal_bottlenecks_1000-discoal_consts_10000/ \
             models/lp/popf-base-discoal_bottlenecks_1000__discoal_consts_10000-0.05.pkl \
             models/fasternn/fasternn_discoal_consts_10000-0.05.pt \
             models/schrider_resnet/resnet_discoal_consts_10000-0.05.pt \
    --names popformer-no-pretrain \
            popformer-ft \
            popformer-lp \
            FASTER-NN \
            resnet34 \
    --datasets data/dataset/discoal_bottlenecks_10000 \
               data/dataset/discoal_bottlenecks_5000 \
               data/dataset/discoal_bottlenecks_2500 \
               data/dataset/discoal_bottlenecks_1000 \
               data/dataset/discoal_bottlenecks_500 \
               data/dataset/discoal_bottlenecks_100 \
    --varying \
    --metric auc

python analysis/test_selection.py \
    --models models/selbin-popf-init-discoal_consts_1000/ \
             models/selbin-popf-base-discoal_consts_10000-discoal_consts_1000/ \
             models/lp/popf-base-discoal_consts_10000__discoal_consts_1000-0.05.pkl \
             models/fasternn/fasternn_discoal_consts_1000-0.05.pt \
             models/schrider_resnet/resnet_discoal_consts_1000-0.05.pt \
    --names popformer-no-pretrain \
            popformer-ft \
            popformer-lp \
            FASTER-NN \
            resnet34 \
    --datasets data/dataset/discoal_consts_1000 \
               data/dataset/discoal_consts_5000 \
               data/dataset/discoal_consts_10000 \
               data/dataset/discoal_consts_50000 \
               data/dataset/discoal_consts_100000 \
    --varying --metric auc --trained-on 1000

python analysis/test_selection.py \
    --models models/selbin-popf-init-discoal_consts_100000/ \
             models/selbin-popf-base-discoal_consts_10000-discoal_consts_100000/ \
             models/lp/popf-base-discoal_consts_10000__discoal_consts_100000-0.05.pkl \
             models/fasternn/fasternn_discoal_consts_100000-0.05.pt \
             models/schrider_resnet/resnet_discoal_consts_100000-0.05.pt \
    --names popformer-no-pretrain-inv \
            popformer-ft-inv \
            popformer-lp-inv \
            FASTER-NN-inv \
            resnet34-inv \
    --datasets data/dataset/discoal_consts_1000 \
               data/dataset/discoal_consts_5000 \
               data/dataset/discoal_consts_10000 \
               data/dataset/discoal_consts_50000 \
               data/dataset/discoal_consts_100000 \
    --varying --metric auc --trained-on 100000

# 3. evaluate various training sizes

python analysis/test_selection.py \
    --models models/selbin-popf-init-discoal_consts_10000-0.05/ \
             models/selbin-popf-init-discoal_consts_10000-0.5/ \
             models/selbin-popf-init-discoal_consts_10000-0.9/ \
             models/selbin-popf-init-discoal_consts_10000-0.95/ \
             models/selbin-popf-init-discoal_consts_10000-0.99/ \
             models/selbin-popf-init-discoal_consts_10000-0.995/ \
             models/selbin-popf-init-discoal_consts_10000-0.9995/ \
             models/selbin-popf-base-discoal_consts_10000-discoal_consts_10000-0.05/ \
             models/selbin-popf-base-discoal_consts_10000-discoal_consts_10000-0.5/ \
             models/selbin-popf-base-discoal_consts_10000-discoal_consts_10000-0.9/ \
             models/selbin-popf-base-discoal_consts_10000-discoal_consts_10000-0.95/ \
             models/selbin-popf-base-discoal_consts_10000-discoal_consts_10000-0.99/ \
             models/selbin-popf-base-discoal_consts_10000-discoal_consts_10000-0.995/ \
             models/selbin-popf-base-discoal_consts_10000-discoal_consts_10000-0.9995/ \
             models/lp/popf-base-discoal_consts_10000__discoal_consts_10000-0.05.pkl \
             models/lp/popf-base-discoal_consts_10000__discoal_consts_10000-0.5.pkl \
             models/lp/popf-base-discoal_consts_10000__discoal_consts_10000-0.9.pkl \
             models/lp/popf-base-discoal_consts_10000__discoal_consts_10000-0.95.pkl \
             models/lp/popf-base-discoal_consts_10000__discoal_consts_10000-0.99.pkl \
             models/lp/popf-base-discoal_consts_10000__discoal_consts_10000-0.995.pkl \
             models/lp/popf-base-discoal_consts_10000__discoal_consts_10000-0.9995.pkl \
             models/fasternn/fasternn_discoal_consts_10000-0.05.pt \
             models/fasternn/fasternn_discoal_consts_10000-0.5.pt \
             models/fasternn/fasternn_discoal_consts_10000-0.9.pt \
             models/fasternn/fasternn_discoal_consts_10000-0.95.pt \
             models/fasternn/fasternn_discoal_consts_10000-0.99.pt \
             models/fasternn/fasternn_discoal_consts_10000-0.995.pt \
             models/fasternn/fasternn_discoal_consts_10000-0.995.pt \
             models/schrider_resnet/resnet_discoal_consts_10000-0.05.pt \
             models/schrider_resnet/resnet_discoal_consts_10000-0.5.pt \
             models/schrider_resnet/resnet_discoal_consts_10000-0.9.pt \
             models/schrider_resnet/resnet_discoal_consts_10000-0.95.pt \
             models/schrider_resnet/resnet_discoal_consts_10000-0.99.pt \
             models/schrider_resnet/resnet_discoal_consts_10000-0.995.pt \
             models/schrider_resnet/resnet_discoal_consts_10000-0.9995.pt \
    --names popformer-no-pretrain-0.05 \
            popformer-no-pretrain-0.5 \
            popformer-no-pretrain-0.9 \
            popformer-no-pretrain-0.95 \
            popformer-no-pretrain-0.99 \
            popformer-no-pretrain-0.995 \
            popformer-no-pretrain-0.9995 \
            popformer-ft-0.05 \
            popformer-ft-0.5 \
            popformer-ft-0.9 \
            popformer-ft-0.95 \
            popformer-ft-0.99 \
            popformer-ft-0.995 \
            popformer-ft-0.9995 \
            popformer-lp-0.05 \
            popformer-lp-0.5 \
            popformer-lp-0.9 \
            popformer-lp-0.95 \
            popformer-lp-0.99 \
            popformer-lp-0.995 \
            popformer-lp-0.9995 \
            FASTER-NN-0.05 \
            FASTER-NN-0.5 \
            FASTER-NN-0.9 \
            FASTER-NN-0.95 \
            FASTER-NN-0.99 \
            FASTER-NN-0.995 \
            FASTER-NN-0.9995 \
            resnet34-0.05 \
            resnet34-0.5 \
            resnet34-0.9 \
            resnet34-0.95 \
            resnet34-0.99 \
            resnet34-0.995 \
            resnet34-0.9995 \
    --datasets data/dataset/discoal_consts_10000 \
    --trainsizes --metric auc
