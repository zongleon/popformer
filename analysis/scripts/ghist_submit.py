import numpy as np
import os
import subprocess
import sys
import pandas as pd
sys.path.insert(0, "analysis/")
import matplotlib.pyplot as plt
from evaluation.evaluators.genome_classification import plot_region
from scipy.signal import find_peaks

plt.switch_backend("webagg")

FINAL = True
SUBMISSION_ID = "f12"
MARGINS = 100000
THRESH = 0.5
LD = True

ts = [
    "singlesweep", "singlesweep.growth_bg",
    "multisweep", "multisweep.growth_bg"
]
ts = [f"{t}_final" if FINAL else t for t in ts]

test_vcf_stub = "data/GHIST/raw/GHIST_2025_{t}.21.testing.vcf"
final_vcf_stub = "data/GHIST/raw/GHIST_2025_{t}.15.final.vcf"

pred_stub = "preds/ghist_{t}_popf-small-lp-low-s_region_plot_data.npz"

def ld_filter(significant_regions, test_vcf):
    # use plink to find regions in LD with top significant region - remove them
    # from the list, and continue until no regions remain
    top_region = significant_regions[0]
    ld_removed_regions = []
    while significant_regions:
        current_region = significant_regions.pop(0)
        ld_removed_regions.append(current_region)
        start, end, z = current_region
        # call plink to find regions in LD
        # use --ld-snps to limit pairwise calculations to SNPs in current region
        # ld-snps takes a list of SNP IDs, but we only have positions
        # first generate a list of SNP IDs in the region using PLINK
        plink_cmd = [
            "plink2",
            "--vcf", test_vcf,
            "--max-alleles", "2",
            "--set-all-var-ids", "@:#:$r:$a",
            "--chr", "21" if not FINAL else "15",
            "--from-bp", str(start),
            "--to-bp", str(end),
            "--write-snplist",
            "--out", "temp_region"
        ]
        subprocess.run(plink_cmd, check=True, stdout=subprocess.DEVNULL)

        # now calculate LD with these SNPs
        ld_cmd = [
            "plink2",
            "--vcf", test_vcf,
            "--max-alleles", "2",
            "--set-all-var-ids", "@:#:$r:$a",
            "--r2-phased",
            "--ld-snp-list", "temp_region.snplist",
            "--ld-window-r2", "0.05",
            "--out", "temp_ld"
        ]
        subprocess.run(ld_cmd, check=True, stdout=subprocess.DEVNULL)

        ld_snps = set()
        with open("temp_ld.vcor", "r") as f:
            next(f)  # skip header
            for line in f:
                parts = line.strip().split()
                snp1, snp2, r2 = parts[2], parts[5], float(parts[6])
                ld_snps.add(snp1)
                ld_snps.add(snp2)
    
        # remove regions that overlap with LD SNPs
        for region in significant_regions:
            region_start, region_end, region_z = region
            # check if any SNP in the region is in ld_snps
            plink_cmd = [
                "plink2",
                "--vcf", test_vcf,
                "--max-alleles", "2",
                "--set-all-var-ids", "@:#:$r:$a",
                "--chr", "21" if not FINAL else "15",
                "--from-bp", str(region_start),
                "--to-bp", str(region_end),
                "--write-snplist",
                "--out", "temp_check_region"
            ]
            subprocess.run(plink_cmd, check=True, stdout=subprocess.DEVNULL)

            overlap = False
            with open("temp_check_region.snplist", "r") as f:
                for line in f:
                    snp_id = line.strip()
                    if snp_id in ld_snps:
                        overlap = True
                        break
            if overlap:
                significant_regions.remove(region)

        # print(f"Regions remaining after LD pruning: {len(significant_regions)}")
    return ld_removed_regions
    

def top_regions(significant_regions, p95):
    # get top peaks
    zs = [r[2] for r in significant_regions]
    peaks, _ = find_peaks(zs, distance=5, height=p95)
    regions = []
    for peak in peaks:
        start = starts[peak]
        end = ends[peak]
        z = z_scores[peak]
        regions.append((start, end, z))
    return regions

for t in ts:
    test_vcf = test_vcf_stub.format(t=t) if not FINAL else final_vcf_stub.format(t=t.replace("_final", ""))
    pred = pred_stub.format(t=t)
    os.makedirs(f"figs/ghist_submit/{SUBMISSION_ID}", exist_ok=True)
    out_file = f"figs/ghist_submit/{SUBMISSION_ID}/ghist_{t}_{SUBMISSION_ID}.bed"

    p = np.load(pred)
    starts, ends, probs = p["start_pos"], p["end_pos"], p["preds"]
    print(f"\nLoaded predictions for {test_vcf}:", probs.shape)

    # window predictions
    # window_size = 5
    # windowed_probs = np.convolve(probs, np.ones(window_size)/window_size, mode='valid')

    # say the first 100Mb set the baseline mean and std
    baseline_size = 100_000_000 // 50000
    baseline_mean = np.mean(probs[:baseline_size])
    baseline_std = np.std(probs[:baseline_size])

    # calculate z-scores based on baseline
    # z_scores = (probs - baseline_mean) / baseline_std

    # calculate z-scores for probabilities
    z_scores = (probs - np.mean(probs)) / np.std(probs)

    # print 95th percentile of z-scores
    p95 = np.percentile(z_scores, 95)
    thresh = (THRESH - np.mean(probs)) / np.std(probs)
    if thresh > p95:
        p95 = thresh
    print(f"95th percentile of z-scores: {p95}")
    
    regions = []
    for i in range(len(starts)):
        regions.append((starts[i], ends[i], z_scores[i]))

    if LD:
        # filter regions with z-score above 95th percentile
        # and sort by z-score descending
        significant_regions = []
        for reg in regions:
            if reg[2] > p95:
                significant_regions.append(reg)
        significant_regions.sort(key=lambda x: x[2], reverse=True)
        ld_removed_regions = ld_filter(significant_regions, test_vcf)
    else:
        ld_removed_regions = top_regions(regions, p95)

    # print("Regions after LD pruning:")
    # for region in ld_removed_regions:
    #     print(f"Start: {region[0]}, End: {region[1]}, Z-score: {region[2]}")

    # extend regions by 100kb on each side
    ld_removed_regions = [
        (max(0, r[0] - MARGINS), r[1] + MARGINS, r[2]) for r in ld_removed_regions
    ]

    # if "single" in t:
    #     # keep only top region for singlesweep
    #     ld_removed_regions = [ld_removed_regions[0]]

    # merge overlapping regions in ld_removed_regions
    ld_removed_regions.sort(key=lambda x: x[0])
    merged_regions = []
    current_start, current_end, current_z = ld_removed_regions[0]
    for region in ld_removed_regions[1:]:
        start, end, z = region
        if start <= current_end:
            # overlap - extend current region
            current_end = max(current_end, end)
            current_z = max(current_z, z)
        else:
            merged_regions.append((current_start, current_end, current_z))
            current_start, current_end, current_z = start, end, z
    merged_regions.append((current_start, current_end, current_z))

    # merged_regions = [r for r in merged_regions if r[2] > THRESH]

    print(f"Regions after LD pruning for {test_vcf}:")
    for region in merged_regions:
        print(f"Start: {region[0]}, End: {region[1]}, Z-score: {region[2]}")

    # write final regions to BED file
    with open(out_file, "w") as f:
        for region in merged_regions:
            f.write(f"{'21' if not FINAL else '15'}\t{region[0]}\t{region[1]}\n")

    print(f"Wrote final regions to {out_file}")

    # and plot final regions using evaluation/evaluators/genome_classification.py's plot_region
    plot_region(
        [probs],
        [""],
        starts,
        ends,
        save_path=f"figs/ghist_submit/{SUBMISSION_ID}/ghist_{t}_{SUBMISSION_ID}.png",
        window=5,
        line=True,
        label_df=pd.DataFrame({"start": [r[0] for r in merged_regions], "end": [r[1] for r in merged_regions]})
    )

if LD:
    # delete temp files including
    os.remove("temp_region.snplist")
    os.remove("temp_ld.vcor")
    os.remove("temp_check_region.snplist")
    # and their corresponding logs
    os.remove("temp_region.log")
    os.remove("temp_ld.log")
    os.remove("temp_check_region.log")