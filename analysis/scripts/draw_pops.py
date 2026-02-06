from demesdraw import tubes
import math

import msprime
import numpy as np
import stdpopsim

import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use("seaborn-v0_8-talk")


def make_param_dict(names, values):
    param_dict = {}
    assert len(names) == len(values)
    for i in range(len(names)):
        param_dict[names[i]] = values[i]
    return param_dict

def exp(param_names, param_values):
    """Note this is a 1 population model"""
    params = make_param_dict(param_names, param_values)

    T2 = params["T2"]
    N2 = params["N2"]

    N0 = N2 / math.exp(-params["growth"] * T2)

    demography = msprime.Demography()
    demography.add_population(name="POP", initial_size=N0, growth_rate=params["growth"])

    # use s_onset as another color
    demography.add_population_parameters_change(time=T2, initial_size=N2, growth_rate=0)
    demography.add_population_parameters_change(time=params["T1"], initial_size=params["N1"])

    return stdpopsim.DemographicModel(id="exp",
        description="exponential growth",
        long_description="exponential growth",
        model=demography), demography

param_names = ["N1", "N2", "T1", "T2", "growth"]
param_ranges = [[14941.43038, 28349.61056], # N1
                [1791.089232, 28128.56808], # N2
                [1816.208475, 4949.464129], # T1
                [302.7920375, 1414.100021], # T2
                #[0.00171393, 0.04719168913]] # growth
                [0, 0.001],] # growth, shifted range
s_onset = [100, 1000] # s_onset

fig, axs = plt.subplots(4, 8, figsize=(12, 8), layout="tight", sharex=True, sharey=True)
axs = axs.flatten()
rng = np.random.default_rng(42)
for i, ax in enumerate(axs):
    params = []
    # randomly sample each param within its range
    for r in param_ranges:
        sampled_value = rng.uniform(r[0], r[1])
        params.append(sampled_value)
    
    model, _ = exp(param_names, params)
    tubes(model.model.to_demes(), ax)
        
    # turn yticks off for all but first, leftmost
    # and turn y spine off
    if i % 8 != 0:
        ax.spines['left'].set_visible(False)
        ax.set_yticks([])
        # ax.tick_params(left=False, labelleft=False)
    
    ax.set_ylabel("")
    ax.set_xlabel("")
    ax.set_xticks([])
    ax.label_outer()

    # Highlight s_onset range on the y-axis (time)
    ax.axhspan(s_onset[0], s_onset[1], color='orange', alpha=0.3, label='Selection onset range')

# overlay a legend for s_onset
handles, labels = axs[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper right')

fig.supylabel("time ago (generations)")

plt.savefig("figs/demes_pan4.png")
plt.clf()  # Clear figure for next plot