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

model_to_color = {
    "popformer": colors[0],
    "popformer-ft": colors[1],
    "popformer-lp": colors[2],
    "FASTER-NN": colors[3],
    "resnet34": colors[4],
    "tajimas_d": colors[5],
    "IMPUTE 5": colors[3],
    "Nearest Neighbor": colors[4],
}

pop_to_color = {
    "EAS": "#778500",
    "SAS": "#c44cfd",
    "AFR": "#ffd845",
    "EUR": "#018ead",
    "AMR": "#710027",
}
