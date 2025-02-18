"""
Function that separates FC edges amplitudes according to 3 connectivity regions:
- gray-to-gray matter (GM-GM)
- gray-to-white matter (GM-WM)
- white-to-white matter (WM-WM)

Atlas GM: either Neuromorphometrics (NMM) or NMM x default mode network (DMN) atlas (10.1152/jn.00338.2011)
Atlas WM: John Hopkins University (JHU, https://identifiers.org/neurovault.collection:264)

BOLD vs ADC differences are assessed with two-sided Mann-Whitney-Wilcoxon tests, 
with Bonferroni correction for multiple comparisons.
"""

import numpy as np
from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import statannot
from copy import deepcopy
import sys
import os
sys.path.insert(1, os.path.dirname(sys.path[0]))
from utils import common_regions, load_FC, wilcoxon_sign_edges
from fig_style import fig_size
import json

# "se_rs" (BOLD), "diff_rs" (ADC-fMRI)
contrasts     = ["se_rs", "diff_rs"]
fieldstrength = ["3T", "7T"]
# percentage of subjects in which the ROI should be present
perc_common = 1

params_path = 'params.json'

# Read and parse the JSON file
with open(params_path, 'r') as file:
    params = json.load(file)

data_folder  = Path(f"{params["paths"]["data_path"]}")
savefig_path = params["paths"]["savefig_path"]
# "NMM" = neuromorphometric, gray matter atlas
GM_region = "NMM" 
GSR       = True

for field in fieldstrength:
    
    FC_name = f"FC_{GM_region}_JHU_{GM_region}_JHU_GSR_only_all_{field}_10vx_reg.npy"

    df = pd.DataFrame(columns=["correlation", "Contrast", "ROI"])
    # Find the ROIs that are common to all contrasts for that field
    GM_all_ROIs, WM_all_ROIs = common_regions(field, 0, data_folder)

    # Only the ROIs that are present in all the subjects and all contrasts
    # FC_all_contrasts_all_subjects contains the Pearson's corr coeff
    FC_all_contrasts_all_subjects, GM_common_ROIs, WM_common_ROIs = load_FC(
        perc_common, GM_all_ROIs, WM_all_ROIs, FC_name, data_folder
    )

    GM_WM_common_ROIs = list(GM_common_ROIs) + list(WM_common_ROIs)

    for idx, contrast in enumerate(contrasts):

        FC_all = deepcopy(FC_all_contrasts_all_subjects[idx])

        mask = wilcoxon_sign_edges(FC_all)

        if not np.allclose(mask.T, mask):
            mask = mask + mask.T
        # Remove the diagonal
        mask = np.logical_and(mask, ~np.identity(mask.shape[0]).astype("bool"))
        
        # Calculate the mean FC across subjects
        mean_FC   = np.nanmean(FC_all, axis=0)
        # Put the self-connections to 0 to remove warning in np.arctanh
        mean_FC[np.identity(mean_FC.shape[0]).astype("bool")] = 0

        # Fisher transformed Pearson corr => FC
        FC_fisher = np.arctanh(mean_FC)
        FC_all_df = pd.DataFrame(FC_fisher)
        # Put NaNs where no mask
        FC_all_df_masked = FC_all_df.where(mask)
    
        FC_all_values     = FC_all_df_masked.values
        FC_all_values_all = np.squeeze(FC_all_values[np.triu_indices(len(list(GM_WM_common_ROIs)))].reshape((-1, 1)))
        # Remove the NaN values 
        FC_all_values_all = FC_all_values_all[~np.isnan(FC_all_values_all)]

        d = {
            "correlation": FC_all_values_all,
            "Contrast": [contrast] * len(FC_all_values_all),
            "ROI": ["All"] * len(FC_all_values_all),
        }
        
        df_tmp = pd.DataFrame(data=d)
        df = pd.concat([df, df_tmp])



        FC_WM_WM = np.squeeze(FC_all_values[len(list(GM_common_ROIs)):, len(list(GM_common_ROIs)):][np.triu_indices(len(list(WM_common_ROIs)))].reshape((-1, 1)))
        # Remove the NaN values 
        FC_WM_WM = FC_WM_WM[~np.isnan(FC_WM_WM)]

        d = {
            "correlation": FC_WM_WM,
            "Contrast": [contrast] * len(FC_WM_WM),
            "ROI": ["WM - WM"] * len(FC_WM_WM),
        }
        df_tmp = pd.DataFrame(data=d)
        df = pd.concat([df, df_tmp])




        FC_GM_GM = np.squeeze(FC_all_values[:len(list(GM_common_ROIs)), :len(list(GM_common_ROIs))][np.triu_indices(len(list(GM_common_ROIs)))].reshape((-1, 1)))
        # Remove the NaN values 
        FC_GM_GM = FC_GM_GM[~np.isnan(FC_GM_GM)]

        d = {
            "correlation": FC_GM_GM,
            "Contrast": [contrast] * len(FC_GM_GM),
            "ROI": ["GM - GM"] * len(FC_GM_GM),
        }
        df_tmp = pd.DataFrame(data=d)
        df = pd.concat([df, df_tmp])



        FC_GM_WM = np.squeeze(FC_all_values[len(list(GM_common_ROIs)):, :len(list(GM_common_ROIs))].reshape((-1, 1)))
        # Remove the NaN values 
        FC_GM_WM = FC_GM_WM[~np.isnan(FC_GM_WM)]

        d = {
            "correlation": FC_GM_WM,
            "Contrast": [contrast] * len(FC_GM_WM),
            "ROI": ["GM - WM"] * len(FC_GM_WM),
        }
        df_tmp = pd.DataFrame(data=d)
        df = pd.concat([df, df_tmp])


    fig, ax = plt.subplots(1, 1, figsize=fig_size(0.5))
    # my_pal = ["#D9300C", "#ef9481", "#4ddc0a", "#a4ef81", "#0aa0dc", "#8fd0ea"]
    my_pal = np.array(sns.color_palette("colorblind"))[[0, 3]]
    median_bold = df[(df.correlation >= 0) & (df.Contrast == "se_rs") & (df.ROI == "All")].correlation.median()
    
    pos_plot = sns.violinplot(
        data=df[(df.correlation >= 0) & (df.ROI != "All")],
        y="correlation",
        x="ROI",
        hue="Contrast",
        hue_order=contrasts,
        ax=ax,
        # xlab="",
        palette=my_pal,
        order=['GM - GM', 'WM - WM', 'GM - WM']
        )
    
    df_pos_bold = df[(df.correlation >= 0) & (df.Contrast == "se_rs")]
    df_pos_adc  = df[(df.correlation >= 0) & (df.Contrast == "diff_rs")]

    sns.stripplot(x="ROI", 
                y="correlation", 
                data=df[(df.correlation >= 0) & (df.ROI != "All")], 
                hue="Contrast", 
                hue_order=contrasts,
                jitter=True, 
                zorder=1, 
                ax=ax, 
                dodge=True, 
                palette=my_pal,
                edgecolor='darkgray',
                order=['GM - GM', 'WM - WM', 'GM - WM'],
                linewidth=1,
                alpha=0.8)

    for collection in ax.collections:
        if isinstance(collection, matplotlib.collections.PolyCollection):
            collection.set_edgecolor(collection.get_facecolor())
            collection.set_facecolor(collection.get_facecolor())
            collection.set_alpha(0.1)

    handles, labels = ax.get_legend_handles_labels()
    # When creating the legend, only use the first two elements
    # to effectively remove the last two.
    plt.legend(handles[2:], ["BOLD-fMRI", "ADC-fMRI"], bbox_to_anchor=(.5, 1.05), loc="lower center", borderaxespad=0., ncol=2, frameon=False)
    ax.tick_params(bottom=False)
    ax.set(ylabel="Positive functional connections", xlabel=None)
    pos_plot.set_ylim([-0.3, 1.1])

    # Statistical tests, positive correlations
    couples = []
    for ROI_ in df[(df.correlation >= 0) & (df.ROI != "All")].ROI.unique():
        for Contrast in df[(df.correlation >= 0) & (df.ROI != "All")].Contrast.unique():
            if (
                df[
                    (df.correlation >= 0) & (df.Contrast == Contrast) & (df.ROI == ROI_)
                ].shape[0]
                > 5
            ):
                couples.append((ROI_, Contrast))

    couples_end = []
    for ROI_ in df[(df.correlation >= 0) & (df.ROI != "All")].ROI.unique():
        for i in range(len(couples)):
            if (
                (i + 1 < len(couples))
                and (couples[i][0] == ROI_)
                and (couples[i + 1][0] == ROI_)
                and (couples[i][1][-2:] == couples[i + 1][1][-2:])
            ):
                couples_end.append((couples[i], couples[i + 1]))
            if (
                (i + 2 < len(couples))
                and (couples[i][0] == ROI_)
                and (couples[i + 2][0] == ROI_)
                and (couples[i][1][-2:] == couples[i + 2][1][-2:])
            ):
                couples_end.append((couples[i], couples[i + 2]))

    statannot.add_stat_annotation(
        ax,
        data=df[(df.correlation >= 0) & (df.ROI != "All")],
        y="correlation",
        x="ROI",
        hue="Contrast",
        hue_order=contrasts,
        box_pairs=couples_end,
        test="Mann-Whitney",
        text_format="star",
        loc="inside",
        order=['GM - GM', 'WM - WM', 'GM - WM']
    )

    
    pos_plot.set_xticklabels(['GM - GM\n' + '$N_{BOLD}$' + f'={len(df_pos_bold[df_pos_bold.ROI == "GM - GM"].values)}\n' + '$N_{ADC}$' + f'={len(df_pos_adc[df_pos_adc.ROI == "GM - GM"].values)}', 
                              'WM - WM\n' + '$N_{BOLD}$' + f'={len(df_pos_bold[df_pos_bold.ROI == "WM - WM"].values)}\n' + '$N_{ADC}$' + f'={len(df_pos_adc[df_pos_adc.ROI == "WM - WM"].values)}', 
                              'GM - WM\n' + '$N_{BOLD}$' + f'={len(df_pos_bold[df_pos_bold.ROI == "GM - WM"].values)}\n' + '$N_{ADC}$' + f'={len(df_pos_adc[df_pos_adc.ROI == "GM - WM"].values)}'])

    plt.tight_layout()
    if GSR:
        suffix = ""
    else:
        suffix = "_no_GSR"
    print("Saved in : ", f'{savefig_path}/corr_pos_violin_{field}{suffix}.pdf')
    fig.savefig(f'{savefig_path}/corr_pos_violin_{field}{suffix}.pdf')
    fig.savefig(f'{savefig_path}/corr_pos_violin_{field}{suffix}.png')
    plt.close()




    fig, ax = plt.subplots(1, 1, figsize=fig_size(0.5))

    neg_plot = sns.violinplot(
        data=df[(df.correlation < 0) & (df.ROI != "All")],
        y="correlation",
        x="ROI",
        hue="Contrast",
        hue_order=contrasts,
        ax=ax,
        palette=my_pal,
        order=['GM - GM', 'WM - WM', 'GM - WM']
    )

    df_neg_bold = df[(df.correlation < 0) & (df.Contrast == "se_rs")]
    df_neg_adc  = df[(df.correlation < 0) & (df.Contrast == "diff_rs")]

    sns.stripplot(x="ROI", 
                y="correlation", 
                data=df[(df.correlation < 0) & (df.ROI != "All")], 
                hue="Contrast", 
                hue_order=contrasts,
                jitter=True, 
                zorder=1, 
                ax=ax, 
                dodge=True, 
                palette=my_pal,
                edgecolor='darkgray',
                order=['GM - GM', 'WM - WM', 'GM - WM'],
                linewidth=1,
                alpha=0.8)

    for collection in ax.collections:
        if isinstance(collection, matplotlib.collections.PolyCollection):
            collection.set_edgecolor(collection.get_facecolor())
            collection.set_facecolor(collection.get_facecolor())
            collection.set_alpha(0.1)

    handles, labels = ax.get_legend_handles_labels()
    # When creating the legend, only use the first two elements
    # to effectively remove the last two.
    plt.legend(handles[2:], ["BOLD-fMRI", "ADC-fMRI"], bbox_to_anchor=(.5, 1.05), loc="lower center", borderaxespad=0., ncol=2, frameon=False)
    ax.tick_params(bottom=False)
    ax.set(ylabel="Negative functional connections", xlabel=None)
    neg_plot.set_ylim([-0.75, 1.1])


    # Statistical tests, negative correlations
    couples = []
    for ROI_ in df[(df.correlation < 0) & (df.ROI != "All") & (df.ROI != "WM - WM")].ROI.unique():
        for Contrast in df[(df.correlation < 0) & (df.ROI != "All") & (df.ROI != "WM - WM")].Contrast.unique():
            if (
                df[
                    (df.correlation < 0) & (df.Contrast == Contrast) & (df.ROI == ROI_)
                ].size
                > 5
            ):
                couples.append((ROI_, Contrast))


    couples_end = []
    for ROI_ in df[(df.correlation < 0)].ROI.unique():
        for i in range(len(couples)):
            if (
                (i + 1 < len(couples))
                and (couples[i][0] == ROI_)
                and (couples[i + 1][0] == ROI_)
                and (couples[i][1][-2:] == couples[i + 1][1][-2:])
            ):
                couples_end.append((couples[i], couples[i + 1]))
            if (
                (i + 2 < len(couples))
                and (couples[i][0] == ROI_)
                and (couples[i + 2][0] == ROI_)
                and (couples[i][1][-2:] == couples[i + 2][1][-2:])
            ):
                couples_end.append((couples[i], couples[i + 2]))

    if len(couples_end) > 0:
        statannot.add_stat_annotation(
            ax,
            data=df[(df.correlation < 0) & (df.ROI != "All")],
            y="correlation",
            x="ROI",
            hue="Contrast",
            hue_order=contrasts,
            box_pairs=couples_end,
            test="Mann-Whitney",
            text_format="star",
            order=['GM - GM', 'WM - WM', 'GM - WM'],
            loc="inside"
        )

    neg_plot.set_xticklabels(['GM - GM\n' + '$N_{BOLD}$' + f'={len(df_neg_bold[df_neg_bold.ROI == "GM - GM"].values)}\n' + '$N_{ADC}$' + f'={len(df_neg_adc[df_neg_adc.ROI == "GM - GM"].values)}', 
                              'WM - WM\n' + '$N_{BOLD}$' + f'={len(df_neg_bold[df_neg_bold.ROI == "WM - WM"].values)}\n' + '$N_{ADC}$' + f'={len(df_neg_adc[df_neg_adc.ROI == "WM - WM"].values)}', 
                              'GM - WM\n' + '$N_{BOLD}$' + f'={len(df_neg_bold[df_neg_bold.ROI == "GM - WM"].values)}\n' + '$N_{ADC}$' + f'={len(df_neg_adc[df_neg_adc.ROI == "GM - WM"].values)}'])
    
    plt.tight_layout()
    print("Saved in : ", f'{savefig_path}/corr_neg_violin_{field}{suffix}.pdf')
    fig.savefig(f'{savefig_path}/corr_neg_violin_{field}{suffix}.pdf')
    fig.savefig(f'{savefig_path}/corr_neg_violin_{field}{suffix}.png')
    plt.close()
