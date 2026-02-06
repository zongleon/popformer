from functools import cache
import os
from ..core import BaseModel
import numpy as np
import pandas as pd
from popformer.collators import RawMatrixCollator, MetadataCollator
import allel
import h5py


class Predictor:
    def predict(self, mat, pos):
        raise NotImplementedError("implement in subclass")


class Pi(Predictor):
    def predict(self, mat, pos):
        ac = allel.HaplotypeArray(mat).count_alleles()
        pi = allel.sequence_diversity(pos=pos, ac=ac)
        return pi


class TajimasD(Predictor):
    def predict(self, mat, pos):
        ac = allel.HaplotypeArray(mat).count_alleles()
        tajimas_d = allel.tajima_d(ac, pos=pos)
        return tajimas_d


class SFS(Predictor):
    def __init__(self, sfs_index: int, proportional: bool = True):
        self.sfs_index = sfs_index
        self.proportional = proportional

    def predict(self, mat, pos):
        ac = allel.HaplotypeArray(mat).count_alleles()
        dac = ac[:, 1]  # derived allele counts
        sfs = allel.sfs(dac)
        if self.proportional:
            sfs = sfs / np.sum(sfs)

        if self.sfs_index >= len(sfs):
            return 0.0
        return sfs[self.sfs_index]


class nSNPs(Predictor):
    def predict(self, mat, pos):
        return mat.shape[0]


class SummaryStatModel(BaseModel):
    """
    Summary statistic model for methods that do not need positional information.
    (operate only on windows).
    """

    def __init__(self, model_name: str, summary_stat: str = "pi", **kwargs):
        self.model_name = model_name
        self.collator = RawMatrixCollator()
        if summary_stat == "pi":
            self.predictor = Pi()
        elif summary_stat == "tajimas_d":
            self.predictor = TajimasD()
        elif summary_stat == "sfs":
            self.predictor = SFS(**kwargs)
        elif summary_stat == "n_snps":
            self.predictor = nSNPs()
        else:
            raise ValueError(f"Unsupported summary statistic: {summary_stat}")

    def preprocess(self, batch):
        # collator
        out = self.collator(batch)
        return out

    def run(self, batch):
        """Make predictions on the given batch of data."""

        mat = batch["input_ids"]  # shape (n_haps, n_snps)
        distances = batch["distances"]  # shape (n_snps,)

        preds = np.empty((len(mat), 2))
        for i, (m, d) in enumerate(zip(mat, distances)):
            m = np.array(m).T  # transpose to (n_snps, n_haps)
            pos = np.cumsum(d, axis=-1)

            preds[i, 1] = self.predictor.predict(m, pos)

        return preds


class PosPredictor:
    def predict(self, ex):
        raise NotImplementedError("implement in subclass")


class iHS(PosPredictor):
    def __init__(self, chroms, positions, haplotypes, score="max"):
        self.chroms = chroms
        self.positions = positions
        self.haplotypes = haplotypes
        self.score = score

    @cache
    def chrom_ihs(self, chrom):
        chrom_mask = self.chroms == chrom
        hap_chr = self.haplotypes[chrom_mask]
        pos_chr = self.positions[chrom_mask]
        ihs = allel.ihs(hap_chr, pos_chr)
        return ihs, pos_chr

    def predict(self, ex):
        start = ex["start_pos"]
        end = ex["end_pos"]
        chrom = ex["chrom"]
        ihs, pos_chr = self.chrom_ihs(chrom)
        ihs_window = ihs[(pos_chr >= start) & (pos_chr <= end)]
        if self.score == "max":
            ihs_score = np.nanmax(np.abs(ihs_window))  # take max ihs in the region
            # proportion of |ihs| > 2
        else:
            ihs_score = (
                np.sum(np.abs(ihs_window) > 2) / len(ihs_window)
                if len(ihs_window) > 0
                else 0
            )
        return ihs_score


class RecombinationRate(PosPredictor):
    genetic_map_dir = "/bigdata/smathieson/1000g-share/genetic_map/"
    recomb_maps = {}

    def load_genetic_map(self, chrom):
        if chrom in self.recomb_maps:
            return self.recomb_maps[chrom]
        # load from file or database
        recomb_map = pd.read_csv(
            os.path.join(self.genetic_map_dir, f"genetic_map_GRCh37_chr{chrom}.txt"),
            sep="\t",
            header=0,
            names=["chrom", "pos", "rate", "map"],
            dtype={"chrom": str, "pos": int, "rate": float, "map": float},
        )
        recomb_map["chrom"] = recomb_map["chrom"].str.replace("chr", "").astype(int)
        self.recomb_maps[chrom] = recomb_map
        return recomb_map

    def predict(self, ex):
        if "recomb_rate" in ex:
            return ex["recomb_rate"]
        start = ex["start_pos"]
        end = ex["end_pos"]
        chrom = ex["chrom"]
        # get from genetic map
        recomb_map = self.load_genetic_map(chrom)

        recomb_start_window = np.searchsorted(recomb_map["pos"], start)
        recomb_end_window = np.searchsorted(recomb_map["pos"], end)

        recomb_rate_window = recomb_map["rate"].iloc[
            recomb_start_window:recomb_end_window
        ]
        recomb_rate = recomb_rate_window.max()
        return recomb_rate


class SummaryStatPosModel(BaseModel):
    """Summary statistic model for methods that need positional information."""

    def __init__(
        self,
        model_name: str,
        variant_file: str | None = None,
        summary_stat: str = "ihs",
        **kwargs,
    ):
        self.model_name = model_name
        self.collator = MetadataCollator()

        # Open the HDF5 file
        if variant_file is not None:
            with h5py.File(variant_file, "r") as f:
                variant_data = {
                    "chrom": f["variants/CHROM"][:],
                    "pos": f["variants/POS"][:],
                    "gt": f["calldata/GT"][:],
                }
                # chrom to int
                variant_data["chrom"] = np.array(
                    [int(c.decode("utf-8")) for c in variant_data["chrom"]]
                )
                self.chroms = variant_data["chrom"]
                self.positions = variant_data["pos"]
                genotypes = allel.GenotypeArray(variant_data["gt"])
                self.haplotypes = genotypes.to_haplotypes()

        if summary_stat == "ihs":
            self.predictor = iHS(self.chroms, self.positions, self.haplotypes, **kwargs)
        elif summary_stat == "recomb":
            self.predictor = RecombinationRate()
        else:
            raise ValueError(f"Unsupported summary statistic: {summary_stat}")

    def preprocess(self, batch):
        # collator
        out = self.collator(batch)
        return out

    def run(self, batch):
        """Make predictions on the given batch of data."""
        n = len(batch[batch.keys()[0]])
        preds = np.empty((n, 2))
        for i, ex in enumerate(batch):
            preds[i, 1] = self.predictor.predict(ex)

        return preds
