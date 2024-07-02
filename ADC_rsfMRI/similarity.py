"""
Function that calculates the inter-subject similarity by calculating the Pearson's correlation coefficient 
between the significant edges, excluding self-connections, split in 3 connectivity regions:

- gray-to-gray matter (GM-GM)
- gray-to-white matter (GM-WM)
- white-to-white matter (WM-WM)

mean + 95% CI is depicted


"""

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from copy import deepcopy
import seaborn as sns
import scipy.stats as st

from utils import common_regions, wilcoxon_sign_edges, load_FC, similarity_GM_GM, similarity_GM_WM, similarity_WM_WM
from fig_style import fig_size
import json

fieldstrength = ['3T', '7T']
# percentage of subjects in which the ROI should be present
perc_common   = 1 

params_path = 'params.json'

# Read and parse the JSON file
with open(params_path, 'r') as file:
    params = json.load(file)
params_path  = 'params.json'
data_folder  = Path(f"{params["paths"]["data_path"]}")
savefig_path = params["paths"]["savefig_path"]

# "se_rs" (BOLD), "diff_rs" (ADC-fMRI)
contrasts     = ["se_rs", "diff_rs"]
# "NMM" = neuromorphometric, gray matter atlas
GM_region = "NMM" 
GSR       = True
for field in fieldstrength:

    fig, ax = plt.subplots(len(contrasts), 4, figsize=fig_size(1, 0.7), gridspec_kw={'width_ratios':[1,1,1,0.08]})
   
    # Load FC
    # Find the ROIs that are common to all contrasts for that field
    GM_all_ROIS, WM_all_ROIS = common_regions(field, 0, data_folder)
    FC_name = f"FC_{GM_region}_JHU_{GM_region}_JHU_GSR_only_all_{field}_10vx_reg.npy"
    # FC_all_contrasts_all_subjects[0] = BOLD-fMRI FC
    # FC_all_contrasts_all_subjects[1] =  ADC-fMRI FC
    FC_all_contrasts_all_subjects, GM_common_ROIs, WM_common_ROIs = load_FC(perc_common, GM_all_ROIS, WM_all_ROIS, FC_name, data_folder)

    for idx, contrast in enumerate(contrasts):

        subjects_folder = Path(f"{data_folder}/{contrast}/") 
        FC_all          = deepcopy(FC_all_contrasts_all_subjects[idx])
         
        mask = wilcoxon_sign_edges(FC_all)
        if not np.allclose(mask.T, mask):
            mask = mask + mask.T
        mask     = np.logical_and(mask, ~np.identity(mask.shape[0]).astype('bool'))
        mask_rep = np.repeat(mask[np.newaxis, ...], FC_all.shape[0], axis=0)
        FC_all[~mask_rep] = np.nan
        # FC: Fisher's transform of Pearson's correlation coeff
        FC_all = np.arctanh(FC_all)

        similarity_GM_GM_matrix, n_GM_GM = similarity_GM_GM(FC_all, GM_common_ROIs)
        similarity_WM_WM_matrix, n_WM_WM = similarity_WM_WM(FC_all, GM_common_ROIs, WM_common_ROIs)
        similarity_GM_WM_matrix, n_GM_WM = similarity_GM_WM(FC_all, GM_common_ROIs)

        if contrast == "se_rs":
            label_name = "BOLD-fMRI \nSubjects"
        elif contrast == "diff_rs":
            label_name = "ADC-fMRI \nSubjects"
        sns.heatmap(
                    ax=ax[idx, 0], 
                    data=similarity_GM_GM_matrix, 
                    vmin=-1, 
                    vmax=1, 
                    xticklabels=False, 
                    yticklabels=False,
                    cbar=False, 
                    cmap="turbo"
                    )


        mean_GM_GM = np.mean(similarity_GM_GM_matrix.reshape(-1, 1))
        CI_GM_GM   = st.t.interval(
                                   0.95, 
                                   df=len(similarity_GM_GM_matrix.reshape(-1, 1))-1, 
                                   loc=mean_GM_GM, 
                                   scale=st.sem(similarity_GM_GM_matrix.reshape(-1, 1))
                                   )

        mean_WM_WM = np.mean(similarity_WM_WM_matrix.reshape(-1, 1))
        CI_WM_WM   = st.t.interval(
                                   0.95, 
                                   df=len(similarity_WM_WM_matrix.reshape(-1, 1))-1,
                                   loc=mean_WM_WM, 
                                   scale=st.sem(similarity_WM_WM_matrix.reshape(-1, 1))
                                   )
        
        mean_GM_WM = np.mean(similarity_GM_WM_matrix.reshape(-1, 1))
        CI_GM_WM   = st.t.interval(
                                   0.95, 
                                   df=len(similarity_GM_WM_matrix.reshape(-1, 1))-1, 
                                   loc=mean_GM_WM, 
                                   scale=st.sem(similarity_GM_WM_matrix.reshape(-1, 1))
                                   )
        
        if idx == 0:
            ax[0, 0].set_title("GM - GM \n" + f"{mean_GM_GM:.2f} [{CI_GM_GM[0][0]:.2f}, {CI_GM_GM[1][0]:.2f}]\n" + 
                               r"N$_e$" + f"={n_GM_GM}")
        else:
            ax[idx, 0].set_title(f"{mean_GM_GM:.2f} [{CI_GM_GM[0][0]:.2f}, {CI_GM_GM[1][0]:.2f}]\n" + r"N$_e$" + 
                                 f"={n_GM_GM}")
        sns.heatmap(ax=ax[idx, 1], data=similarity_WM_WM_matrix, vmin=-1, vmax=1, xticklabels=False, yticklabels=False,
                    cbar=False, cmap="turbo")
        if idx == 0:
            ax[0, 1].set_title("WM - WM \n" + f"{mean_WM_WM:.2f} [{CI_WM_WM[0][0]:.2f}, {CI_WM_WM[1][0]:.2f}]\n" + 
                               r"N$_e$" + f"={n_WM_WM}")
        else:
            ax[idx, 1].set_title(f"{mean_WM_WM:.2f} [{CI_WM_WM[0][0]:.2f}, {CI_WM_WM[1][0]:.2f}]\n" + r"N$_e$" + 
                                 f"={n_WM_WM}")
        sns.heatmap(ax=ax[idx, 2], data=similarity_GM_WM_matrix, vmin=-1, vmax=1, xticklabels=False, yticklabels=False,
                    cmap="turbo", cbar_ax=ax[idx, 3], cbar_kws={'label': 'Pearson coefficient'})
        if idx == 0:
            ax[0, 2].set_title("GM - WM \n" + f"{mean_GM_WM:.2f} [{CI_GM_WM[0][0]:.2f}, {CI_GM_WM[1][0]:.2f}]\n" + 
                               r"N$_e$" + f"={n_GM_WM}")
        else:
            ax[idx, 2].set_title(f"{mean_GM_WM:.2f} [{CI_GM_WM[0][0]:.2f}, {CI_GM_WM[1][0]:.2f}]\n" + r"N$_e$" + 
                                 f"={n_GM_WM}")
        ax[idx, 0].set_ylabel(label_name)
        ax[1, 0].set_xlabel("Subjects")
        ax[1, 1].set_xlabel("Subjects")
        ax[1, 2].set_xlabel("Subjects")

    plt.tight_layout()
    if GSR:
        suffix = ""
    else:
        suffix = "_no_GSR"
    print("File saved in : ", f'{savefig_path}/similarity_{field}{suffix}.pdf')
    plt.savefig(f'{savefig_path}/similarity_{field}{suffix}.pdf')
    plt.savefig(f'{savefig_path}/similarity_{field}{suffix}.png')
    plt.close()


   


