from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

plt.style.use("seaborn-v0_8-ticks")

model_id = 0


def darken(color):
    """Darken a given color by 20%."""
    c = mcolors.to_rgb(color)
    factor = 0.8
    darkened = tuple(max(0, min(1, channel * factor)) for channel in c)
    return darkened


def lighten(color):
    """Lighten a given color by 20%."""
    c = mcolors.to_rgb(color)
    factor = 1.2
    lightened = tuple(max(0, min(1, channel * factor)) for channel in c)
    return lightened


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
    "popformer-lp-panooa": colors[-2],
    "popformer-ft": colors[1],
    "popformer-lp": colors[2],
    "popformer": colors[0],
    "FASTER-NN": colors[3],
    "resnet34": colors[4],
    "tajimas_d": colors[5],
    "IMPUTE 5": colors[3],
    "Nearest Neighbor": colors[4],
}


def model_to_color(model: str):
    if model in model_color_map:
        return model_color_map[model]
    else:
        model_lower = model.lower()
        for key in model_color_map:
            if model_lower.startswith(key.lower()):
                return model_color_map[key]


pop_to_color = {
    "EAS": "#778500",
    "SAS": "#c44cfd",
    "AFR": "#ffd845",
    "EUR": "#018ead",
    "AMR": "#710027",
}
