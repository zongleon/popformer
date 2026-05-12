from itertools import cycle

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
    "popformer-ft": colors[1],
    "popformer-lp": colors[2],
    "popformer": colors[0],
    "popformer-base": colors[0],
    "popformer-no-pretrain": colors[0],
    "FASTER-NN": colors[3],
    "resnet34": colors[4],
    "tajimas_d": colors[8],
    "IMPUTE5": colors[3],
    "Nearest Neighbor": colors[4],
    "Column Frequency": colors[5],
}


# assign by default tab10 colors to models not in the above map
def get_color(model):
    return model_color_map.setdefault(model, cycle(colors))


dataset_rename_map = {
    "pan2CEU_test": "CEU",
    "pan2CHB_test": "CHB",
    "pan2YRI_test": "YRI",
    "pan_3_demoid-0_balanced": "Strong Bottleneck",
    "pan_3_demoid-1_balanced": "Old Migration",
}


def get_model_base_name(model: str):
    """Extract base model name by removing version/parameter suffixes."""
    for key in model_color_map:
        if model.lower().startswith(key.lower()):
            if key.lower() == "popformer":
                # hardcoded, im sorry
                return "popformer-no-pretrain"
            return key
    return model


def model_to_color(model: str):
    if model in model_color_map:
        return model_color_map[model]

    if get_model_base_name(model) in model_color_map:
        return model_color_map[get_model_base_name(model)]

    return cycle(colors)


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
