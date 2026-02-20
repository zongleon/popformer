import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8-ticks")

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
    "#cacaca",
]
model_color_map = {
    "popformer-ft": colors[1],
    "popformer-lp": colors[2],
    "popformer": colors[0],
    "FASTER-NN": colors[3],
    "resnet34": colors[4],
    "tajimas_d": colors[5],
    "IMPUTE 5": colors[3],
    "Nearest Neighbor": colors[4],
}


def get_model_base_name(model: str):
    """Extract base model name by removing version/parameter suffixes."""
    for key in model_color_map:
        if model.lower().startswith(key.lower()):
            return key
    return model


def model_to_color(model: str):
    if model in model_color_map:
        return model_color_map[model]

    if get_model_base_name(model) in model_color_map:
        return model_color_map[get_model_base_name(model)]

    return "#000000"  # default to black


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
