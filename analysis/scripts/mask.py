import pandas as pd
import numpy as np
from cyvcf2 import VCF, Writer
from tqdm import tqdm
import allel

INPUT_VCF = "data/imputation/raw/KHV.chr20.phase3_shapeit2_mvncall_integrated_v5a.20130502.genotypes.vcf.gz"
INPUT2_VCF = "/bigdata/smathieson/1000g-share/VCF/ALL.chr20.snps.phase3_shapeit2_mvncall_integrated_v5a.20130502.genotypes.vcf.gz"
OUTPUT_DIR = "data/imputation/masked/KHV_{ratio}_{seed}.vcf.gz"
OUTPUT2_DIR = "data/imputation/masked/KHV_{seed}.vcf.gz"


def infinium_mask(input_vcf, chr):
    # build 37 for these infinium arrays
    # bmp = "IMP/InfiniumExome-24v1-1_A1.csv"
    bmp = "data/imputation/raw/InfiniumOmniExpress-24v1-3_A1.csv"
    df: pd.DataFrame = pd.read_csv(
        bmp,
        usecols=["Name", "Chr", "MapInfo"],
        skiprows=7,
        dtype={"Name": str, "Chr": str, "MapInfo": pd.Int32Dtype()},
    )

    df = df.dropna(subset=["MapInfo"])

    df = df[df["Chr"] == chr]

    df = df[["Chr", "MapInfo"]]
    df = df.rename(columns={"Chr": "chrom", "MapInfo": "pos"})

    # only keep positions from the vcf that aren't in the df
    # chrom, pos = [], []
    # for idx, v in enumerate(VCF(input_vcf)):
    #     if str(v.CHROM) != chr:
    #         continue
    #     if v.POS not in df["pos"].values:
    #         chrom.append(v.CHROM)
    #         pos.append(v.POS)

    # df = pd.DataFrame({"chrom": chrom, "pos": pos})

    return df


def random_mask(input_vcf, mask_ratio, seed=0):
    rng = np.random.default_rng(seed)

    # slow but whatever
    n = len([v for v in VCF(input_vcf)])

    num_samples = int(mask_ratio * n)
    indices = rng.choice(n, num_samples, replace=False)

    chrom, pos = [], []
    for idx, v in enumerate(VCF(input_vcf)):
        if idx not in indices:
            continue
        chrom.append(v.CHROM)
        pos.append(v.POS)

    df = pd.DataFrame({"chrom": chrom, "pos": pos})

    return df


def apply_mask(
    input_vcf,
    mask_df,
    output_ref_vcf,
    ref_ratio,
    strategy="remove",
    max_samples=None,
    seed=0,
):
    """
    Split input_vcf into two VCFs using a mask:
      - output_ref_vcf: contains only ref_samples
      - output_masked_vcf: contains masked test samples

    Parameters:
        input_vcf (str): Path to the input VCF file.
        mask_df (dataframe): DataFrame indicating sites to be masked.
        output_vcf (str): Path to the output VCF.
        ref_samples (int): Number of samples to use as reference.
        strategy (str, optional): If "remove", masked sites are dropped from masked output.
                                  If "replace", masked sites' genotypes are set to "./." in masked output.
    """
    rng = np.random.default_rng(seed)

    # Load mask
    mask_set = set(zip(mask_df["chrom"].astype(str), mask_df["pos"].astype(int)))

    # Validate samples and split
    vcf_all = VCF(input_vcf)

    OLD = True
    if OLD:
        if max_samples is not None:
            sample_idxs = list(range(len(vcf_all.samples)))
            rng.shuffle(sample_idxs)
            sample_idxs = sample_idxs[:max_samples]
            vcf_all.set_samples([vcf_all.samples[i] for i in sample_idxs])
        all_samples = list(vcf_all.samples)
        n_samples = len(all_samples)
        if ref_ratio < 1:
            ref_set = list(rng.choice(n_samples, int(ref_ratio * n_samples), replace=False))
        elif type(ref_ratio) is int:
            ref_set = list(rng.choice(n_samples, ref_ratio, replace=False))
        else:
            raise ValueError(
                "ref_ratio should be a float (ratio) or an int (number of ref samples)"
            )

        ref_set = [all_samples[x] for x in ref_set]
        target_samples = [s for s in all_samples if s not in ref_set]

        if type(ref_ratio) is int:
            target_samples = target_samples[:ref_ratio]
        # print(target_samples)
        # print(all_samples)

        print(len(list(ref_set)))
        print(len(target_samples))
    else:
        # read sample table
        ped = pd.read_csv("data/imputation/raw/20130606_g1k.ped", sep="\t")
        in_vcf = set(vcf_all.samples)
        
        # target is 2 random KHV, 2 random CHS, 2 CDX
        target_samples = []
        for pop in ["KHV", "CHS", "CDX"]:
            pop_samples = ped[ped["Population"] == pop]["Individual ID"].tolist()
            pop_samples = [s for s in pop_samples if s in in_vcf]
            rng.shuffle(pop_samples)
            target_samples.extend(pop_samples[:10])
        
        # ref_set are 10 random KHV not in target, 10 CHS, 10 CDX
        ref_set = []
        for pop in ["KHV", "CHS", "CDX"]:
            pop_samples = ped[ped["Population"] == pop]["Individual ID"].tolist()
            pop_samples = [s for s in pop_samples if s not in target_samples and s in in_vcf]
            rng.shuffle(pop_samples)
            ref_set.extend(pop_samples[:2])

    # Prepare per-group VCF readers with sample subsetting
    vcf_ref = VCF(input_vcf)
    vcf_ref.set_samples(list(ref_set))

    vcf_tgt = VCF(input_vcf)
    vcf_tgt.set_samples(target_samples)

    # Writers
    output_base = output_ref_vcf.rstrip(".vcf.gz")
    w_ref = Writer(output_base + "_ref.vcf.gz", vcf_ref)
    w_tgt = Writer(output_base + "_tgt.vcf.gz", vcf_tgt)

    # Write ref output (unmasked)
    non_seg = set()
    for v in tqdm(vcf_ref, desc="Writing ref VCF"):
        if all(gt[0] == 0 and gt[1] == 0 for gt in v.genotypes):
            non_seg.add(int(v.POS))
            continue
        w_ref.write_record(v)
    w_ref.close()

    # Write masked output for target samples
    masked_mafs = []
    masked_snps = []  # New list to store masked SNP data
    for v in tqdm(vcf_tgt, desc="Writing target VCF"):
        key = (str(v.CHROM), int(v.POS))
        if key[1] in non_seg:
            continue
        if key in mask_set:
            gt_types = v.gt_types  # 0=hom_ref, 1=het, 2=hom_alt, 3=unknown
            n_alleles = 2 * np.sum(gt_types != 3)
            n_alt = np.sum(gt_types == 1) + 2 * np.sum(gt_types == 2)
            maf = n_alt / n_alleles if n_alleles > 0 else np.nan
            masked_mafs.append(maf)

            # Collect genotypes as a comma-separated string
            genotypes_str = "".join([str(gt[0]) + str(gt[1]) for gt in v.genotypes])
            # genotypes_str += "".join([str(gt[1]) for gt in v.genotypes])
            masked_snps.append(
                {"chrom": v.CHROM, "pos": v.POS, "MAF": maf, "genotypes": genotypes_str}
            )

            if strategy == "remove":
                continue
            elif strategy == "replace":
                # Replace genotypes with missing for masked targets only
                v.set_format("GT", ["./."] * v.num_samples)
        w_tgt.write_record(v)
    w_tgt.close()

    # Write masked SNPs to CSV
    if masked_snps:
        masked_df = pd.DataFrame(masked_snps)
        masked_df.to_csv(f"{output_base}_snps.csv", index=False)
        print(f"Saved masked SNPs ({len(masked_snps)})")

    print(
        f"Masked sites MAF: mean={np.nanmean(masked_mafs):.4f}, std={np.nanstd(masked_mafs):.4f}"
    )
    return output_base + "_ref.vcf.gz", output_base + "_tgt.vcf.gz"


def convert_vcf(vcf_filename):
    """Convert vcf_filename"""
    # here we save only CHROM, GT (genotypes) and POS (SNP positions)
    # see: https://scikit-allel.readthedocs.io/en/stable/io.html
    allel.vcf_to_hdf5(
        vcf_filename,
        vcf_filename.replace(".vcf.gz", ".h5"),
        fields=["CHROM", "GT", "POS"],
        overwrite=True,
    )


if __name__ == "__main__":
    for seed in range(3):
        # mask_df = infinium_mask(INPUT_VCF, "20")
        # output_vcf = OUTPUT2_DIR.format(seed=seed)
        # ref_vcf, tgt_vcf = apply_mask(
        #     INPUT_VCF,
        #     mask_df,
        #     output_vcf,
        #     ref_ratio=32,
        #     strategy="remove",
        #     seed=seed,
        #     max_samples=None,
        # )

        # convert_vcf(ref_vcf)
        # convert_vcf(tgt_vcf)

        for ratio in [0.2, 0.4, 0.6, 0.8]:
            mask_df = random_mask(INPUT_VCF, ratio, seed=seed)
            print(f"Generated random mask with {mask_df.shape[0]} positions for ratio {ratio}")

            output_vcf = OUTPUT_DIR.format(ratio=int(ratio*100), seed=seed)
            ref_vcf, tgt_vcf = apply_mask(INPUT_VCF, mask_df, output_vcf, ref_ratio=32, strategy="remove", seed=seed)
            convert_vcf(ref_vcf)
            convert_vcf(tgt_vcf)
