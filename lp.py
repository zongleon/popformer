import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import LinearSVR
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
import datasets

GRID = False
best_C = 0.02

# Load data
# train_features = np.load("dataset/ft_selbin2_feats/train_allhaps_feats.npz")["features"]
# train_labels = np.asarray(datasets.load_from_disk("dataset/ft_selbin2")["label"])
# train_labels = np.asarray(datasets.load_from_disk("dataset/ft_selreg2")["label"])
train_features = np.load("dataset/ft_selbin2_feats/train_bigwindow_feats.npz")["features"]
train_labels = np.asarray(datasets.load_from_disk("dataset/ft_selbin_bigwindow")["label"])

# more_train_features = np.load("dataset/ft_selbin2_feats/train_feats2.npz")["features"]
# more_train_labels = pd.read_csv("FASTER_NN/fasternn_train_meta.csv")["label"].values

# train_features = np.concatenate([train_features, more_train_features], axis=0)
# train_labels = np.concatenate([train_labels, more_train_labels], axis=0)

# Evaluate on test set
test_features = np.load("dataset/ft_selbin2_feats/test_feats.npz")["features"]
metadata = pd.read_csv("FASTER_NN/fasternn_test_meta.csv")

if GRID:
    # Split train into train/eval
    X_tr, X_ev, y_tr, y_ev = train_test_split(
        train_features, train_labels, test_size=0.01, stratify=train_labels, random_state=0
    )

    # Hyperparameter sweep for C on eval split
    C_grid = np.logspace(-4, 4, 32)
    best_C = None
    best_acc = -1.0
    for C in C_grid:
        clf = LogisticRegression(random_state=0, C=C, max_iter=1000, verbose=0)
        clf.fit(X_tr, y_tr)
        ev_pred = clf.predict(X_ev)
        test_pred = clf.predict(test_features)
        acc = accuracy_score(y_ev, ev_pred)
        test_acc = accuracy_score(metadata["label"].values, test_pred)
        print("-" * 30)
        print(f"C={C:.4g} eval accuracy={acc:.4f}")
        print(f"C={C:.4g} test accuracy={test_acc:.4f}")
        if acc > best_acc:
            best_acc = acc
            best_C = C

    print(f"Selected C={best_C} (eval accuracy={best_acc:.4f})")

# Fit on train+eval (full training set) with best C
classifier = LogisticRegression(random_state=0, C=best_C, max_iter=1000)
# classifier = LinearRegression()
# shuffle
train_features, train_labels = shuffle(train_features, train_labels, random_state=0,
                                       n_samples=None)
classifier.fit(train_features, train_labels)

predictions = classifier.predict(test_features)

# Optional overall accuracy (if labels cover all test rows)
if "label" in metadata.columns and len(metadata) == len(predictions):
    overall_acc = (metadata["label"].values == predictions).mean()
    print(f"\nOverall Test Accuracy: {overall_acc:.4f}")
    print("-" * 30)

for dataset_name in metadata["dataset"].unique():
    idx = metadata["dataset"] == dataset_name
    dataset_labels = metadata.loc[idx, "label"].values
    dataset_preds = predictions[idx]
    dataset_acc = (dataset_labels == dataset_preds).mean()
    print(f"Dataset: {dataset_name}, Accuracy: {dataset_acc:.4f}")

for t in ["singlesweep", "singlesweep.growth_bg", "multisweep", "multisweep.growth_bg"]:
    t = "rl" + t + "_100000"
    data = np.load(f"dataset/ft_selbin2_feats/{t}.npz")
    preds = classifier.predict_proba(data["features"])
    start_pos = data["start_pos"]
    end_pos = data["end_pos"]
    chrom = data["chrom"]

    # save as preds
    np.savez(f"GHIST/logreg_selbin_pan2_{t}.npz", preds=preds, start_pos=start_pos, end_pos=end_pos, chrom=chrom)

data = np.load("dataset/ft_selbin2_feats/CEU_allhaps.npz")
preds = classifier.predict_proba(data["features"])
# expm1
# preds = np.expm1(preds)
start_pos = data["start_pos"]
end_pos = data["end_pos"]
chrom = data["chrom"]
np.savez("SEL/CEU_preds.npz", preds=preds, start_pos=start_pos, end_pos=end_pos, chrom=chrom)