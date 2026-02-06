import sys
import pandas as pd

# paths
STRICT_MASK_BED = sys.argv[1]
HOTSPOT_BED = sys.argv[2]
OUTPUT_BED = sys.argv[3]

# read bed files
strict_mask = pd.read_csv(
    STRICT_MASK_BED, sep="\t", header=None, names=["chr", "start", "end", "type"]
)
hotspots = pd.read_csv(
    HOTSPOT_BED, sep="\t", header=None, names=["chr", "start", "end", "name", "score"]
)


def merge_intervals(df):
    if df.empty:
        return []
    df = df.sort_values(["start", "end"])
    merged = []
    cur_s, cur_e = int(df.iloc[0]["start"]), int(df.iloc[0]["end"])
    for _, r in df.iloc[1:].iterrows():
        s, e = int(r["start"]), int(r["end"])
        if s <= cur_e:
            cur_e = max(cur_e, e)
        else:
            merged.append((cur_s, cur_e))
            cur_s, cur_e = s, e
    merged.append((cur_s, cur_e))
    return merged


def subtract_intervals(base_intervals, mask_intervals):
    out = []
    j = 0
    for bs, be in base_intervals:
        cur = bs
        while j < len(mask_intervals) and mask_intervals[j][1] <= bs:
            j += 1
        k = j
        while k < len(mask_intervals) and mask_intervals[k][0] < be:
            ms, me = mask_intervals[k]
            if ms > cur:
                out.append((cur, min(ms, be)))
            cur = max(cur, me)
            if cur >= be:
                break
            k += 1
        if cur < be:
            out.append((cur, be))
    return out


result_rows = []
for chrom in strict_mask["chr"].unique():
    sm_chr = strict_mask[strict_mask["chr"] == chrom][["start", "end"]].copy()
    hs_chr = hotspots[hotspots["chr"] == chrom][["start", "end"]].copy()

    base = merge_intervals(sm_chr)
    masks = merge_intervals(hs_chr)

    kept = subtract_intervals(base, masks)
    for s, e in kept:
        if e > s:
            result_rows.append((chrom, int(s), int(e)))

out_df = pd.DataFrame(result_rows, columns=["chr", "start", "end"])
out_df = out_df.sort_values(["chr", "start", "end"])
out_df.to_csv(OUTPUT_BED, sep="\t", header=False, index=False)
