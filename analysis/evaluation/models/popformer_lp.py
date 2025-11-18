from ..core import BaseModel
import torch
from popformer.models import PopformerForWindowClassification
from popformer.collators import HaploSimpleDataCollator
import pickle
from sklearn.linear_model import LogisticRegression


class PopformerLPModel(BaseModel):
    """Popformer model with linear probe for evaluation."""

    def __init__(
        self,
        model_path: str,
        lp_path: str,
        model_name: str,
        device: torch.device | None = None,
        subsample = None,
        subsample_type = "diverse",
    ):
        self.model = PopformerForWindowClassification.from_pretrained(
            model_path, torch_dtype=torch.float16
        )
        self.model_name = model_name

        with open(lp_path, "rb") as f:
            self.lp_model = pickle.load(f)
            assert isinstance(self.lp_model, LogisticRegression), (
                "Loaded model is not a LogisticRegression instance."
            )

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.model.to(device)
        self.model.eval()

        self.collator = HaploSimpleDataCollator(
            subsample=subsample, subsample_type=subsample_type
        )

    
    def preprocess(self, batch):
        # collator
        batch = self.collator(batch)

        # Move tensors to device
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(self.device, non_blocking=True)

        return batch


    def run(self, batch):
        """Make predictions on the given batch of data."""
        # collator
        output = self.model(
            batch["input_ids"],
            batch["distances"],
            batch["attention_mask"],
            return_hidden_states=True,
        )

        features = output["hidden_states"].mean(dim=(1, 2)).detach().cpu()

        return self.lp_model.predict_proba(features)
