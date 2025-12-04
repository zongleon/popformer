from ..core import BaseModel
import numpy as np
from popformer.collators import RawMatrixCollator
import allel

class Predictor:
    def predict(self, mat, pos):
        raise NotImplementedError("implement in subclass")
    

class Pi(Predictor):
    def predict(self, mat, pos):
        ac = allel.HaplotypeArray(mat).count_alleles()
        pi = allel.sequence_diversity(pos=pos, ac=ac)
        return -pi


class TajimasD(Predictor):
    def predict(self, mat, pos):
        ac = allel.HaplotypeArray(mat).count_alleles()
        tajimas_d = allel.tajima_d(ac, pos=pos)
        return -tajimas_d
    
class iHS(Predictor):
    def predict(self, mat, pos):
        hap = allel.HaplotypeArray(mat)
        ihs = allel.ihs(hap, pos=pos, include_edges=True)
        return np.nan_to_num(np.nanmean(np.abs(ihs)))

class PopformerModel(BaseModel):
    """Popformer model for evaluation."""

    def __init__(
        self, model_name: str, summary_stat: str = "pi"
    ):
        self.model_name = model_name
        self.collator = RawMatrixCollator()
        if summary_stat == "pi":
            self.predictor = Pi()
        elif summary_stat == "tajimas_d":
            self.predictor = TajimasD()
        elif summary_stat == "ihs":
            self.predictor = iHS()
        else:
            raise ValueError(f"Unsupported summary statistic: {summary_stat}")
        
    def preprocess(self, batch):
        # collator
        out = self.collator(batch)
        return out

    def run(self, batch):
        """Make predictions on the given batch of data."""
        
        mat = batch["input_ids"]  # shape (n_haps, n_snps)
        distances = batch["distances"]   # shape (n_snps,)

        preds = np.empty((len(mat), 2))
        for i, (m, d) in enumerate(zip(mat, distances)):
            m = np.array(m).T # transpose to (n_snps, n_haps)
            pos = np.cumsum(d, axis=-1)

            preds[i, 1] = self.predictor.predict(m, pos)
        
        return preds
