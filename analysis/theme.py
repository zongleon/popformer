import matplotlib.pyplot as plt

# style = "seaborn-v0_8-poster"
style = "seaborn-v0_8-talk"
plt.style.use(style)

if "poster" in style:
    # legend needs to be smaller for poster style
    plt.rcParams["legend.fontsize"] = "large"

# define colormap using
colors = [
    "#002ba1",
    "#5c79d5",
    "#b6c3fc",
    "#ddb310",
    "#b51d14",
    "#00beff",
    "#fb49b0",
    "#00b25d",
    "#787878",
]
model_color_map = {
    # Popformer variants (shades of blue)
    "popformer-no-pretrain": colors[0],
    "popformer-ft": colors[1],
    "popformer-lp": colors[2],
    "popformer": colors[0],  # base pretrained / short-name alias
    "popformer-base": colors[0],  # base model alias
    # Competing neural models
    "FASTER-NN": colors[3],
    "resnet34": colors[4],
    # Summary statistics
    "tajimas_d": colors[5],
    "sfs_1": colors[6],
    "sfs_1_count": colors[6],
    "sfs_2": colors[7],
    "n_snps": colors[5],
    # Imputation-task models
    "IMPUTE5": colors[3],
    "Nearest Neighbor": colors[4],
    "Column Frequency": colors[5],
}

INVERT_SCORE_MODELS = ["tajimas_d"]

dataset_rename_map = {
    "pan2CEU_test": "CEU",
    "pan2CHB_test": "CHB",
    "pan2YRI_test": "YRI",
    "pan_3_demoid-0_balanced": "Strong Bottleneck",
    "pan_3_demoid-1_balanced": "Old Migration",
}


def get_model_base_name(model: str) -> str:
    """Strip version/parameter suffixes, returning the canonical base name.

    Keys are checked longest-first so more specific names (e.g.
    'popformer-no-pretrain') match before shorter prefixes ('popformer').
    """
    for key in sorted(model_color_map, key=len, reverse=True):
        if model.lower().startswith(key.lower()):
            return key
    return model


def model_to_color(model: str) -> str:
    base = get_model_base_name(model)
    return model_color_map.get(base, colors[hash(base) % len(colors)])


pop_to_color = {
    "EAS": "#778500",
    "SAS": "#c44cfd",
    "AFR": "#ffd845",
    "EUR": "#018ead",
    "AMR": "#710027",
    "CEU": "#018ead",
    "YRI": "#ffd845",
    "CHB": "#778500",
}
