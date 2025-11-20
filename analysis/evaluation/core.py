import os

from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset, load_from_disk
from torch.utils.data import DataLoader


class BaseModel:
    """Base class for all models used in evaluation."""

    def preprocess(self, batch):
        """Preprocess the given batch of data."""
        return batch
    
    
    def run(self, batch):
        """Make predictions on the given batch of data."""
        raise NotImplementedError


class BaseEvaluator:
    """Base class for all evaluators."""

    def run(self, model: BaseModel):
        """Evaluate the model by making random classifications."""
        # determine names for cache key
        self.model_name = getattr(model, "model_name", model.__class__.__name__)
        dataset_name = getattr(self, "dataset_name")

        # simple file-based cache path
        # TODO LZ: allow user to set cache dir
        cache_dir = os.path.join("preds", "evaluation")
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, f"{dataset_name}__{self.model_name}.npy")

        # return cached predictions if available
        if os.path.exists(cache_path):
            return np.load(cache_path)

        preds = []
        
        dataloader = DataLoader(
            self.dataset, batch_size=self.batch_size, shuffle=False, num_workers=4, collate_fn=model.preprocess
        )

        with torch.inference_mode():
            for batch in tqdm(dataloader):
                output = model.run(batch)
                if isinstance(output, torch.Tensor):
                    preds.append(output.numpy())
                else:
                    preds.append(output)

        predictions = np.concatenate(preds, axis=0)

        # save to cache
        np.save(cache_path, predictions)
        return predictions

    def evaluate(self, predictions):
        """Compute evaluation metrics."""
        raise NotImplementedError


    def trues(self):
        """Get the true labels for the dataset."""
        raise NotImplementedError


class BaseHFEvaluator(BaseEvaluator):
    """Base class for evaluators that use Hugging Face datasets."""

    def __init__(
        self,
        dataset_path,
        labels_path_or_labels=None,
        *,
        dataset_name=None,
        batch_size=1,
    ):
        try:
            self.dataset = load_dataset(dataset_path)
        except ValueError:
            self.dataset = load_from_disk(dataset_path)
        
        if dataset_name is not None:
            self.dataset_name = dataset_name
        else:
            self.dataset_name = os.path.basename(dataset_path)

        if "label" in self.dataset.column_names:
            self.labels = np.asarray(self.dataset["label"])
        elif isinstance(labels_path_or_labels, str):
            if os.path.splitext(labels_path_or_labels)[1] == ".csv":
                self.labels = pd.read_csv(labels_path_or_labels)["label"].to_numpy()
            else:
                raise ValueError("Unsupported labels file format.")
        elif isinstance(labels_path_or_labels, list):
            self.labels = np.asarray(labels_path_or_labels)

        # some datasets we have selection strength for
        if "s" in self.dataset.column_names:
            self.s = np.asarray(self.dataset["s"])

        # some datasets we have position info for
        if "positions" in self.dataset.column_names:
            self.start_pos = [p[0] for p in self.dataset["positions"]]
            self.end_pos = [p[-1] for p in self.dataset["positions"]]
            self.chrom = self.dataset["chrom"]
        elif (
            "start_pos" in self.dataset.column_names
            and "end_pos" in self.dataset.column_names
        ):
            self.start_pos = self.dataset["start_pos"]
            self.end_pos = self.dataset["end_pos"]
            self.chrom = self.dataset["chrom"]

        self.batch_size = batch_size

    def trues(self):
        """Get the true labels for the dataset."""
        return self.labels if hasattr(self, "labels") else None