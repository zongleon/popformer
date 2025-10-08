import sys
import pandas as pd
import numpy as np
from cyvcf2 import VCF, Writer
import allel

rng = np.random.default_rng()

def infinium_mask(chr):
    # build 37 for these infinium arrays
    # bmp = "IMP/InfiniumExome-24v1-1_A1.csv"
    bmp = "IMP/InfiniumOmniExpress-24v1-3_A1.csv"
    df: pd.DataFrame = pd.read_csv(bmp, 
                    usecols = ["Name", "Chr", "MapInfo"],
                    skiprows=7, dtype={"Name": str, "Chr": str, "MapInfo": pd.Int32Dtype()})

    df = df.dropna(subset=["MapInfo"])

    df = df[df["Chr"] == chr]

    df[["Chr", "MapInfo"]].to_csv("IMP/infinium_mask.bed", sep="\t", index=False, header=False)
    print(f"Saved infinium mask with {df.shape[0]} kept positions.")


def random_mask(input_vcf, mask_ratio):
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

    df.to_csv("IMP/random_mask.bed", sep="\t", index=False, header=False)
    print(f"Saved random mask with {df.shape[0]} kept positions")

def apply_mask(input_vcf, mask_path, output_ref_vcf, ref_ratio, strategy="remove"):
    """
    Split input_vcf into two VCFs using a mask:
      - output_ref_vcf: contains only ref_samples
      - output_masked_vcf: contains masked test samples

    Parameters:
        input_vcf (str): Path to the input VCF file.
        mask_path (str): Path to the mask file indicating sites to be masked.
        output_vcf (str): Path to the output VCF.
        ref_samples (int): Number of samples to use as reference.
        strategy (str, optional): If "remove", masked sites are dropped from masked output.
                                  If "replace", masked sites' genotypes are set to "./." in masked output.
    """
    # Load mask
    mask_df = pd.read_csv(mask_path, sep="\t", header=None, names=["chrom", "pos"])
    mask_set = set(zip(mask_df["chrom"].astype(str), mask_df["pos"].astype(int)))

    # Validate samples and split
    vcf_all = VCF(input_vcf)
    all_samples = list(vcf_all.samples)
    n_samples = len(all_samples)
    if ref_ratio < 1:
        ref_set = list(rng.choice(n_samples, int(ref_ratio * n_samples), replace=False))
    elif type(ref_ratio) is int:
        ref_set = list(rng.choice(n_samples, ref_ratio, replace=False))
    else:
        raise ValueError("ref_ratio should be a float (ratio) or an int (number of ref samples)")
         
    ref_set = [all_samples[x] for x in ref_set]
    target_samples = [s for s in all_samples if s not in ref_set]

    if type(ref_ratio) is int:
        target_samples = target_samples[:ref_ratio]
    # print(target_samples)
    # print(all_samples)

    print(len(list(ref_set)))
    print(len(target_samples))

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
    for v in vcf_ref:
        w_ref.write_record(v)
    w_ref.close()

    # Write masked output for target samples
    masked_mafs = []
    masked_snps = []  # New list to store masked SNP data
    for v in vcf_tgt:
        key = (str(v.CHROM), int(v.POS))
        if key in mask_set:
            gt_types = v.gt_types  # 0=hom_ref, 1=het, 2=hom_alt, 3=unknown
            n_alleles = 2 * np.sum(gt_types != 3)
            n_alt = np.sum(gt_types == 1) + 2 * np.sum(gt_types == 2)
            maf = n_alt / n_alleles if n_alleles > 0 else np.nan
            masked_mafs.append(maf)

            # Collect genotypes as a comma-separated string
            genotypes_str = "".join([str(gt[0]) + str(gt[1]) for gt in v.genotypes])
            # genotypes_str += "".join([str(gt[1]) for gt in v.genotypes])
            masked_snps.append({
                "chrom": v.CHROM,
                "pos": v.POS,
                "MAF": maf,
                "genotypes": genotypes_str
            })

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

    print(f"Masked sites MAF: mean={np.nanmean(masked_mafs):.4f}, std={np.nanstd(masked_mafs):.4f}, n={len(masked_mafs)}")
    return output_base + "_ref.vcf.gz", output_base + "_tgt.vcf.gz"

def convert_vcf(vcf_filename):
    """Convert vcf_filename"""
    # here we save only CHROM, GT (genotypes) and POS (SNP positions)
    # see: https://scikit-allel.readthedocs.io/en/stable/io.html
    allel.vcf_to_hdf5(vcf_filename, vcf_filename.replace(".vcf.gz", ".h5"), 
                      fields=['CHROM','GT','POS'],
                      overwrite=True)

if __name__ == "__main__":
    input_vcf = sys.argv[1]
    mask_path = sys.argv[2]
    output_vcf = sys.argv[3]

    ref_vcf, tgt_vcf = apply_mask(input_vcf, mask_path, output_vcf, ref_ratio=32, strategy="remove")
    convert_vcf(ref_vcf)
    convert_vcf(tgt_vcf)
