from ..core import BaseModel
import torch
from popformer.models import PopformerForWindowClassification
from popformer.collators import HaploSimpleDataCollator


class PopformerModel(BaseModel):
    """Popformer model for evaluation."""

    def __init__(
        self, model_path: str, model_name: str, device: torch.device | None = None,
        subsample = None, subsample_type = "diverse"
    ):
        self.model_path = model_path
        self.model_name = model_name
        self.model = PopformerForWindowClassification.from_pretrained(
            model_path, torch_dtype=torch.float16
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

        output = self.model(
            batch["input_ids"],
            batch["distances"],
            batch["attention_mask"],
        )

        return output["logits"].detach().cpu()
