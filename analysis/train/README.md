# training experiments

We are assembling the following models and datasets into some cohesive experiments.

Models:
- Popformer linear probe
- Popformer fine-tuned
- Resnet34
- FASTER-NN

Datasets:
- 1000G real data
  - Labels from Grossman et al.
  - Labels from Akbari et al.
- Simulated CEU, CHB, YRI
- Simulated bottlenecks of varying strength
  - [100%], 50%, 25%, 10%, 5%, 1%
- Simulated constant with varying N
  - N = 1000, 5000, [10000], 50000, 100000

Below are the details of the training experiments performed for the study.

## 1. Real data pipeline

see `1-train-real.sh`

1. Pretrain on 1000G real data
2. Train on simulated CEU
3. Evaluate on simulated CEU, CHB, YRI
4. Evaluate on labeled real data

## 2. Pretraining effects

see `2a-train-varying-bottlenecks.sh` and `2b-train-varying-consts.sh`

For varying bottlenecks, constant sizes:

1. Pretrain on specific simulated dataset (10% bottleneck, N=1000)
2. Train on simulated data (e.g. [100%] bottleneck, N=[10000])
3. Evaluate on other simulated datasets

## 3. Data size effects

For constant N=10000:

1. Pretrain on held-out N=10000
2. Train on N=10000, various train sizes
3. Evaluate on N=10000
