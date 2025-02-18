"""
File that:

- Extracts the BOLD-fMRI and ADC-fMRI timeseries per ROIs, at 3T and at 7T
- Calculate the functional connectivity -FC- (Pearson's correlation) between the ROIs 
  (with or without Global Signal Regression -GSR-)
- Plot the mean FC matrix (upper triangle: ADC-fMRI FC, lower triangle: BOLD-fMRI FC)


"""
import numpy as np
from pathlib import Path
import pandas as pd
import sys
import os
sys.path.insert(1, os.path.dirname(sys.path[0]))
from utils import common_regions, read_labels_NMM_JHU, calculate_correlations, extract_timecourses, load_FC, order_LR, plot_mean_FC
from copy import deepcopy
import json

# percentage of subjects in which the ROI should be present
perc_common       = 1
fieldstrength     = ["3T", "7T"]
# Group the data by L and R part
order_L_R         = False

params_path = 'params.json'

# Read and parse the JSON file
with open(params_path, 'r') as file:
    params = json.load(file)

data_folder  = Path(f"{params["paths"]["data_path"]}")
savefig_path = params["paths"]["savefig_path"]

NMM_JHU_labels_path = data_folder / "labels_Neuromorphometrics_JHU_abbrev.txt"
NMM_JHU_ROIs_nb, NMM_JHU_ROIs_name = read_labels_NMM_JHU(NMM_JHU_labels_path)

# "se_rs" (BOLD), "diff_rs" (ADC-fMRI)
contrasts = ["se_rs", "diff_rs"] 
# "NMM" = neuromorphometric, gray matter atlas
GM_region = "NMM" 
GSR       = True

for field in fieldstrength:
  
    # Calculate the timecourses (DMNxNMM, JHU, GSR, ...)
    # for contrast in contrasts:
    #     extract_timecourses(data_folder / contrast, NMM_JHU_ROIs_name, NMM_JHU_ROIs_nb, field)

    
    # Find the ROIs that are common to all contrasts for that field
    GM_all_ROIs, WM_all_ROIs = common_regions(field, 0, data_folder)
    GM_WM_all_ROIs = GM_all_ROIs + WM_all_ROIs

    FC_name = f"FC_{GM_region}_JHU_{GM_region}_JHU_GSR_only_all_{field}_10vx_reg.npy" 

    # for contrast in contrasts:
    #     calculate_correlations(field, GM_all_ROIs, WM_all_ROIs, GSR, [f"FC_{GM_region}_JHU_{GM_region}_JHU_GSR_only_{field}_10vx_reg",
    #                                                                   f"FC_{GM_region}_JHU_{GM_region}_JHU_GSR_only_all_{field}_10vx_reg"],
    #                            Path(f"{data_folder}/{contrast}/"))

    # Only the ROIs that are present in all the subjects and all contrasts
    FC_all_contrasts_all_subjects, GM_common_ROIs, WM_common_ROIs = load_FC(perc_common, GM_all_ROIs, WM_all_ROIs, FC_name, data_folder)
    GM_common_ROIs    = list(GM_common_ROIs)
    WM_common_ROIs    = list(WM_common_ROIs)
    GM_WM_common_ROIs = GM_common_ROIs + WM_common_ROIs
    
    # Process BOLD
    FC_bold        = deepcopy(FC_all_contrasts_all_subjects[0])
    # Calculate the mean FC across subjects
    mean_FC_bold   = np.nanmean(FC_bold, axis=0)
    # Put the self-connections to 0 to remove warning in np.arctanh
    mean_FC_bold[np.identity(mean_FC_bold.shape[0]).astype("bool")]     = 0
    # Plot upper FC triangle: ADC, lower triangle: BOLD
    fisher_FC_bold = np.arctanh(mean_FC_bold)
    fisher_FC_bold[np.identity(fisher_FC_bold.shape[0]).astype('bool')] = 1

    # Same but with ADC
    FC_ADC        = deepcopy(FC_all_contrasts_all_subjects[1])
    # Calculate the mean FC across subjects
    mean_FC_adc   = np.nanmean(FC_ADC, axis=0)
    # Put the self-connections to 0 to remove warning in np.arctanh
    mean_FC_adc[np.identity(mean_FC_adc.shape[0]).astype("bool")]     = 0
    fisher_FC_adc = np.arctanh(mean_FC_adc)
    fisher_FC_adc[np.identity(fisher_FC_adc.shape[0]).astype('bool')] = 1
    
    # Merge both
    fisher_FC     = np.zeros_like(fisher_FC_adc)
    fisher_FC     = np.triu(fisher_FC_adc) + np.tril(fisher_FC_bold)
    FC_df         = pd.DataFrame(fisher_FC)
    FC_df.index   = GM_WM_common_ROIs
    FC_df.columns = GM_WM_common_ROIs

    # order the ROIs, split by right & left hemisphere (within GM and WM)
    if order_L_R:
        reordered_names_GM_WM = order_LR(GM_common_ROIs, WM_common_ROIs)
    else:
        reordered_names_GM_WM = GM_WM_common_ROIs

    colors = [
        "rosybrown"
        if (idx_ >= len(GM_common_ROIs))
        else "maroon"
        for idx_ in range(len(GM_WM_common_ROIs))
    ]
    
    plot_mean_FC(FC_df, reordered_names_GM_WM, colors, savefig_path, field, "ADC_BOLD", plot=True)
    print(f"Image saved under {savefig_path}/FC_ADC_BOLD_{field}")