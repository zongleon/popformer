from ..core import BaseModel
import torch
import warnings
from popformer.models import PopformerForWindowClassification
from popformer.collators import HaploSimpleDataCollator


class PopformerModel(BaseModel):
    """Popformer model for evaluation."""

    def __init__(
        self,
        model_path: str,
        model_name: str,
        device: torch.device | None = None,
        subsample=None,
        subsample_type="diverse",
    ):
        self.model_path = model_path
        self.model_name = model_name
        self.model = PopformerForWindowClassification.from_pretrained(
            model_path, torch_dtype=torch.float16
        )
        assert all(not torch.isnan(p).any() for p in self.model.parameters()), "NaN detected in model weights!"
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
        return batch

    def run(self, batch):
        """Make predictions on the given batch of data."""
        # Move tensors to device
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(self.device, non_blocking=True)

        output = self.model(
            batch["input_ids"],
            batch["distances"],
            batch["attention_mask"],
        )
        logits = torch.clamp(output["logits"], min=-999, max=999)
        preds = torch.softmax(logits, dim=1)

        if torch.isnan(preds).any():
            warnings.warn("NaN detected in predictions!")
            preds = torch.nan_to_num(preds, nan=0.0)

        # # l_1 - l_0
        # # print(preds)
        # preds = preds[:, 1] - preds[:, 0]

        return preds.cpu().numpy()
