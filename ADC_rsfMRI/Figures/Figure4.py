"""
File that plots the ADC-to-BOLD agreement of mean functional connectivity (FC). 


The FC was derived by taking the Fisher-transformed Pearson's correlation coefficient between 
gray matter (GM) and white matter (WM) nodes.

    Atlas GM: either Neuromorphometrics (NMM, Neuromorphometrics, Inc.)
    
    Atlas WM: John Hopkins University (JHU, https://identifiers.org/neurovault.collection:264)

The regression lines between FC values derived from each contrast were measured for positive and
negative group-mean edge weights separately.

"""

import scipy
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import numpy as np
from copy import deepcopy
import sys
import os
sys.path.insert(1, os.path.dirname(sys.path[0]))
from utils import common_regions, load_FC
from fig_style import fig_size
import json

def add_identity(axes, *line_args, **line_kwargs):
    """
        Function that adds an identity line on axes

        Args:
            axes (matplotlib.axes)
            line_args  (tuple) : empty
            line_kwargs (dict) : 'color', 'ls', 'linewidth',...
    """

    identity, = axes.plot([], [], *line_args, **line_kwargs)

    def callback(axes):
        low_x, high_x = axes.get_xlim()
        low_y, high_y = axes.get_ylim()
        low = max(low_x, low_y)
        high = min(high_x, high_y)
        identity.set_data([low, high], [low, high])

    callback(axes)
    axes.callbacks.connect('xlim_changed', callback)
    axes.callbacks.connect('ylim_changed', callback)

    return axes


params_path = 'params.json'

# Read and parse the JSON file
with open(params_path, 'r') as file:
    params = json.load(file)

# "se_rs" (BOLD-fMRI), "diff_rs" (ADC-fMRI)
contrasts      = ["se_rs", "diff_rs"]
savefig_path   = params["paths"]["savefig_path"]
data_folder    = Path(params["paths"]["data_path"])
# % of subjects in which the ROI should be present
perc_common    = 1
# "NMM" = neuromorphometric, gray matter atlas
GM_region      = "NMM"
fieldstrengths = ['3T', '7T']
# Color of the scatter points
colors         = 'black'

# Global signal regression
GSR = True

# Max and min limit for plotting
lim_max  = 1.7
lim_min  = -.9

# Color of scatter points
color_dots = "#005AB5"
# Color of regression line & identity line
color_line = "#DC3220"

# Swap the axis. If False, BOLD is on x-axis
swap_axis = False


subjects_folders = [data_folder/ contrasts[0],
                    data_folder/ contrasts[1]]

for field in fieldstrengths:

    FC_name = f"FC_{GM_region}_JHU_{GM_region}_JHU_GSR_only_all_{field}_10vx_reg.npy"

    fig, ax  = plt.subplots(figsize=fig_size(fraction=0.5))

    # Find the ROIs that are common to all contrasts for that field
    GM_all_ROIs, WM_all_ROIs = common_regions(field, 0, data_folder)

    FC_all_contrasts_all_subjects, _, _ = load_FC(perc_common, GM_all_ROIs, WM_all_ROIs, FC_name, data_folder)
    # FC_all_contrasts_all_subjects[0] = BOLD-fMRI
    # FC_all_contrasts_all_subjects[1] =  ADC-fMRI
    FC_all_contrasts_all_subjects = [FC_all_contrasts_all_subjects[0], FC_all_contrasts_all_subjects[1]]
    # For each fieldstrength, calculate the BOLD mask where significant edges
    FC_all = deepcopy(FC_all_contrasts_all_subjects[0])
    mask   = np.ones((FC_all.shape[1], FC_all.shape[1]))
    # Remove the diagonal
    mask   = np.logical_and(mask, ~np.identity(mask.shape[0]).astype('bool'))

    # Between groups comparisons
    if swap_axis:
        order = [1, 0]
    else:
        order = [0, 1]

    # Ref connectivity (BOLD if swap_axis=False)
    FC_ref        = deepcopy(FC_all_contrasts_all_subjects[order[0]])
    # Calculate the mean FC across subjects
    mean_FC_ref   = np.nanmean(FC_ref, axis=0)
    # Put the self-connections to 0 to remove warning in np.arctanh
    mean_FC_ref[np.identity(mean_FC_ref.shape[0]).astype("bool")] = 0
    FC_ref_df     = pd.DataFrame(np.arctanh(mean_FC_ref))
    FC_ref_df     = FC_ref_df.where(mask)
    data1_type    = contrasts[order[0]].split('/')[-1]
    FC_ref_values = np.squeeze(FC_ref_df.values.reshape((-1, 1)))

    # Connectivity to compare with BOLD
    FC_comp        = deepcopy(FC_all_contrasts_all_subjects[order[1]])
    # Calculate the mean FC across subjects
    mean_FC_comp   = np.nanmean(FC_comp, axis=0)
    # Put the self-connections to 0 to remove warning in np.arctanh
    mean_FC_comp[np.identity(mean_FC_comp.shape[0]).astype("bool")] = 0
    FC_comp_df     = pd.DataFrame(np.arctanh(mean_FC_comp))
    FC_comp_df     = FC_comp_df.where(mask)
    data2_type     = contrasts[order[1]].split('/')[-1]
    FC_comp_values = np.squeeze(FC_comp_df.values.reshape((-1, 1)))

    # Negative BOLD idx
    idx_neg = (FC_ref_values < 0) & (FC_comp_values < 0) & (~np.isnan(FC_ref_values)) & (~np.isnan(FC_comp_values))
    if np.sum(idx_neg) != 0:
        # Fit FC ref vs FC comp
        res_neg = scipy.stats.linregress(FC_ref_values[idx_neg], FC_comp_values[idx_neg])
        def p_neg(a): return res_neg.intercept + res_neg.slope * a

    # Positive BOLD idx
    idx_pos = (FC_ref_values > 0) & (FC_comp_values > 0) & (~np.isnan(FC_ref_values)) & (~np.isnan(FC_comp_values))
    res_pos = scipy.stats.linregress(FC_ref_values[idx_pos], FC_comp_values[idx_pos])
    def p_pos(a): return res_pos.intercept + res_pos.slope * a

    # Plot connectivity strength of G1 vs G2
    # Add horizontal, vertical & diagonal lines
    ax.axhline(y=0, color='black', linewidth=1)
    ax.axvline(x=0, color='black', linewidth=1)
    add_identity(ax, color=color_line, ls='--', linewidth=3)
    lim_max = np.max([np.max(FC_ref_values[~np.isnan(FC_ref_values)]), np.max(FC_comp_values[~np.isnan(FC_comp_values)])])
    lim_max = lim_max * 1.1
    lim_min = -lim_max
    ax.scatter(FC_ref_values, FC_comp_values, color=color_dots, s=20, alpha=0.05, edgecolor="black", linewidths=1.5)
    if np.sum(idx_neg) != 0:
        x_neg = np.linspace(lim_min, 0, 10)
        if res_neg.pvalue < 0.05:
            pval = "< 0.05"
        else:
            pval = ">= 0.05"

        ax.plot(x_neg, 
                p_neg(x_neg), 
                color=color_line, 
                linestyle="dotted",
                label=f'slope: {res_neg.slope:.2f}, R²: {res_neg.rvalue**2:.2f}, p {pval}',
                linewidth=3)

    x_pos = np.linspace(0, lim_max, 10)
    if res_pos.pvalue < 0.05:
        pval = "< 0.05"
    else:
        pval = ">= 0.05"
    ax.plot(x_pos, 
            p_pos(x_pos), 
            color=color_line,
            label=f'slope: {res_pos.slope:.2f}, R²: {res_pos.rvalue**2:.2f}, p {pval}',
            linewidth=3)
    ax.set_xlim([lim_min, lim_max])
    ax.set_ylim([lim_min, lim_max])
    if contrasts[order[0]]   == "se_rs":
        label = "BOLD-fMRI"
    elif contrasts[order[0]] == "diff_rs":
        label = "ADC-fMRI"

    ax.set_xlabel(label + ' functional connectivity')
    if contrasts[order[1]]   == "se_rs":
        label = "BOLD"
    elif contrasts[order[1]] == "diff_rs":
        label = "ADC-fMRI"

    ax.set_ylabel(label + ' functional connectivity')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2))
    plt.tight_layout()
    # plt.legend()
    print("Saved in : ", f'{savefig_path}/{data1_type}_vs_{data2_type}_all_{field}.pdf')
    if GSR:
        suffix = ""
    else:
        suffix = "_no_GSR"
    plt.savefig(f'{savefig_path}/{data1_type}_vs_{data2_type}_all_{field}{suffix}.pdf')
    plt.savefig(f'{savefig_path}/{data1_type}_vs_{data2_type}_all_{field}{suffix}.png')
    plt.close()
