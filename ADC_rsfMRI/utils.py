from collections import Counter
from pathlib import Path
import os
import pandas as pd
import nibabel as nib
import numpy as np 
from copy import deepcopy
import pingouin as pg
import matplotlib.pyplot as plt
import scipy
from statsmodels.sandbox.stats.multicomp import multipletests
from scipy.stats import pearsonr
import seaborn as sns
import matplotlib as mpl
from fig_style import fig_size

def save_nifti(input_img: np.ndarray, save_name: str, affine_transf: np.ndarray = np.eye(4),
               header: nib.nifti1.Nifti1Header = []) -> None:
    """
    Function that creates a Nifti1image and save it as .nii.gz
    
    Args:
        input_img           (np.ndarray): array to save
        save_name                  (str): name to use to save
        affine_transf       (np.ndarray): 4x4 array, affine transformation
        header (nib.nifti1.Nifti1Header): header of the Nifti file
    
    """
    if header:
        img = nib.Nifti1Image(input_img, affine_transf, header)
    else:
        img = nib.Nifti1Image(input_img, affine_transf)

    nib.save(img, save_name)

def similarity_GM_GM(FC, GM_ROIs):
    """
    Function that calculates the inter-subject similarity with Pearson's correlation, 
    Taking on the interaction GM-GM into account, GM = Gray Matter
    
    Args:
        FC      (np.ndarray) : (Nb_subjects , Nb_ROIs, Nb_ROIs), functional connectivity matrix
        GM_ROIs (np.ndarray) : GM ROIs' names
    
    Returns:
        similarity_matrix (np.ndarray) : (Nb_subjects, Nb_subjects), intersubject Pearson's correlations
        nb_ROIs (int) : number of ROIs used to calculate the Pearson's correlations
    """

    similarity_matrix = np.zeros((FC.shape[0], FC.shape[0]))
    for subject_id_1 in range(FC.shape[0]):
        for subject_id_2 in range(subject_id_1, FC.shape[0]):
            # If same subject, corr = 1 (no need to calculate)
            if subject_id_1 == subject_id_2 :
                similarity_matrix[subject_id_1, subject_id_2] = 1
                similarity_matrix[subject_id_2, subject_id_1] = 1
            else:
                # Take only the upper triangle, because the matrix is symmetrical
                triu1_vect = FC[subject_id_1, :len(GM_ROIs), :len(GM_ROIs)][np.triu_indices(len(GM_ROIs))]
                # Remove nan values (where edge was not significant)
                triu1_vect = triu1_vect[~np.isnan(triu1_vect)]
                triu2_vect = FC[subject_id_2, :len(GM_ROIs), :len(GM_ROIs)][np.triu_indices(len(GM_ROIs))]
                triu2_vect = triu2_vect[~np.isnan(triu2_vect)] 

                corr = pearsonr(triu1_vect, triu2_vect).statistic
                similarity_matrix[subject_id_1, subject_id_2] = corr
                similarity_matrix[subject_id_2, subject_id_1] = corr

    # Store the number of significant edges used to calculate the correlation
    nb_ROIs = len(triu2_vect)

    return similarity_matrix, nb_ROIs

def similarity_GM_WM(FC, GM_ROIs):
    """
    Function that calculates the inter-subject similarity with Pearson's correlation, 
    Taking on the interaction GM-WM into account, GM = Gray Matter, WM = White Matter
    
    Args:
        FC      (np.ndarray) : (Nb_subjects , Nb_ROIs, Nb_ROIs), functional connectivity matrix
        GM_ROIs (np.ndarray) : GM ROIs' names
    
    Returns:
        similarity_matrix (np.ndarray) : (Nb_subjects, Nb_subjects), intersubject Pearson's correlations
        nb_ROIs (int) : number of ROIs used to calculate the Pearson's correlations
    """

    similarity_matrix = np.zeros((FC.shape[0], FC.shape[0]))
    for subject_id_1 in range(FC.shape[0]):
        for subject_id_2 in range(subject_id_1, FC.shape[0]):
            # If same subject, corr = 1 (no need to calculate)
            if subject_id_1 == subject_id_2 :
                similarity_matrix[subject_id_1, subject_id_2] = 1
                similarity_matrix[subject_id_2, subject_id_1] = 1
            else:
                # Because the matrix is not squared (and thus not symmetrical), take the whole rectangle
                triu1_vect = FC[subject_id_1, :len(GM_ROIs), len(GM_ROIs):].flatten()
                # Remove nan values (where edge was not significant)
                triu1_vect = triu1_vect[~np.isnan(triu1_vect)]
                triu2_vect = FC[subject_id_2, :len(GM_ROIs), len(GM_ROIs):].flatten()
                triu2_vect = triu2_vect[~np.isnan(triu2_vect)]
                 
                corr = pearsonr(triu1_vect, triu2_vect).statistic
                similarity_matrix[subject_id_1, subject_id_2] = corr
                similarity_matrix[subject_id_2, subject_id_1] = corr

    nb_ROIs = len(triu2_vect)
    return similarity_matrix, nb_ROIs

def similarity_WM_WM(FC, GM_ROIs, WM_ROIs):
    """
    Function that calculates the inter-subject similarity with Pearson's correlation, 
    Taking on the interaction WM-WM into account, GM = Gray Matter, WM = White Matter
    
    Args:
        FC      (np.ndarray) : (Nb_subjects , Nb_ROIs, Nb_ROIs), functional connectivity matrix
        GM_ROIs (np.ndarray) : GM ROIs' names
        WM_ROIs (np.ndarray) : WM ROIs' names
    
    Returns:
        similarity_matrix (np.ndarray) : (Nb_subjects, Nb_subjects), intersubject Pearson's correlations
        nb_ROIs (int) : number of ROIs used to calculate the Pearson's correlations
    """

    similarity_matrix = np.zeros((FC.shape[0], FC.shape[0]))
    for subject_id_1 in range(FC.shape[0]):
        for subject_id_2 in range(subject_id_1, FC.shape[0]):
            # If same subject, corr = 1 (no need to calculate)
            if subject_id_1 == subject_id_2 :
                similarity_matrix[subject_id_1, subject_id_2] = 1
                similarity_matrix[subject_id_2, subject_id_1] = 1
            else:
                # Take only the upper triangle, because the matrix is symmetrical
                triu1_vect = FC[subject_id_1, len(GM_ROIs):, len(GM_ROIs):][np.triu_indices(len(WM_ROIs))]
                # Remove nan values (where edge was not significant)
                triu1_vect = triu1_vect[~np.isnan(triu1_vect)]
                triu2_vect = FC[subject_id_2, len(GM_ROIs):, len(GM_ROIs):][np.triu_indices(len(WM_ROIs))]
                triu2_vect = triu2_vect[~np.isnan(triu2_vect)]
                
                corr = pearsonr(triu1_vect, triu2_vect).statistic
                similarity_matrix[subject_id_1, subject_id_2] = corr
                similarity_matrix[subject_id_2, subject_id_1] = corr

    nb_ROIs = len(triu2_vect) 
    return similarity_matrix, nb_ROIs

def load_func(subject_folder):
    """
    Function that loads the functional image.
    
    Args:
        subject_folder (pathlib.Posix): path of the subject
    
    Returns:
        func (np.ndarray) : (x, y, z, t) fMRI, single subject
    
    """

    # BOLD spin-echo
    if "se_rs" in str(subject_folder):
        func_path = subject_folder / "ardfMRI_denoised_unringed_sdc_m_ic40_clean.nii.gz"
    # ADC, interleaved b = 200 vs b = 1000 s/mm²
    elif "diff_rs" in str(subject_folder):
        func_path = subject_folder / "adc_m_ic40_clean.nii.gz"

    func   = nib.load(func_path)
    header = func.header
    affine = func.affine
    func   = func.get_fdata()

    return func, affine, header
   
def extractBetween(full_str, start_str, stop_str):
    """
    Function that extracts a substring from full_str, between start_str and stop_str
    
    Args:
        full_str  (str): full string
        start_str (str): string after which to start the substring
        stop_str  (str): string before which to stop the substring
    
    Returns:
        substring contained between start_str and stop_str
    
    """

    return full_str[full_str.find(start_str)+len(start_str):full_str.rfind(stop_str)]

def common_regions(fieldstrength, perc_common, path_data):
    """
    Function that finds common region between a certain percentage of common subject (perc_common), between the 2 
    contrasts ("se_rs", "diff_rs").

    Args :
        fieldstrength (str)          : MRI fieldstrength, either "3T" or "7T"
        perc_common (float)          : between 0 and 1. 
                                       If 0, takes all the regions and do the intercept between contrasts.
                                       If 1, takes only the regions that are present in all the subjects 
                                       and do the intercept between contrasts.
                                       If 0.5, takes only the regions that are present in 50% of the 
                                       subjects and do the intercept between contrasts.
        path_data (pathlib.PosixPath): path of the data folder
    """

    # Gray matter common regions
    GM_common_regions = []
    # White matter common regions
    WM_common_regions = []
    contrasts = ["se_rs", "diff_rs"]
    for contrast in contrasts:
        GM_regions_all_subjects = []
        WM_regions_all_subjects = []
        subjects_folder = path_data / contrast
        files           = os.listdir(subjects_folder)
        # Get all the subject names within the group
        subject_names   = []
        for fl in files:
            if (fieldstrength in fl) and ("BB" in fl) and (not "FC_" in fl) and (not "p_" in fl):
                subject_names.append(fl)

        for subject in subject_names:
            subject_folder = subjects_folder / subject

            mean_GM_ROI_timecourses = pd.read_csv(subject_folder / "NMM_GM_timecourses_all.csv")
            mean_WM_ROI_timecourses = pd.read_csv(subject_folder / "WM_timecourses_all.csv")

            WM_regions   = mean_WM_ROI_timecourses.columns.to_list()
            # Don't take the timecourses that are filled with 0s (should have been removed beforehand normally...)
            WM_idx_all_0 = mean_WM_ROI_timecourses.columns[mean_WM_ROI_timecourses.eq(0).all()].values
            WM_regions   = [item for item in WM_regions if item not in WM_idx_all_0]
            WM_regions_all_subjects.append(WM_regions)
            GM_regions   = mean_GM_ROI_timecourses.columns.to_list()
            # Don't take the timecourses that are filled with 0s (should have been removed beforehand normally...)
            GM_idx_all_0 = mean_GM_ROI_timecourses.columns[mean_GM_ROI_timecourses.eq(0).all()].values
            GM_regions   = [item for item in GM_regions if item not in GM_idx_all_0]
            GM_regions_all_subjects.append(GM_regions) 
    
        
        flattened_list_GM = [region for sublist in GM_regions_all_subjects for region in sublist]
        flattened_list_WM = [region for sublist in WM_regions_all_subjects for region in sublist]

        # Count occurrences of each GM region, per 1 contrast
        region_counter_GM = Counter(flattened_list_GM)
        # Count occurrences of each WM region, per 1 contrast
        region_counter_WM = Counter(flattened_list_WM)

        # The GM regions should be found in at least perc_common subjects
        for region, count in region_counter_GM.items():
            if count >= perc_common * len(subject_names):
                GM_common_regions.append(region)
        # The WM regions should be found in at least perc_common subjects
        for region, count in region_counter_WM.items():
            if count >= perc_common * len(subject_names):
                WM_common_regions.append(region)

    # The GM regions should be found in all 3 contrasts
    region_counter_GM = Counter(GM_common_regions)
    GM_common_regions = []
    for region, count in region_counter_GM.items():
        if count == len(contrasts):
            GM_common_regions.append(region)

    # The WM regions should be found in all 3 contrasts
    region_counter_WM = Counter(WM_common_regions)
    WM_common_regions = []
    for region, count in region_counter_WM.items():
        if count == len(contrasts):
            WM_common_regions.append(region)

    return GM_common_regions, WM_common_regions

def WM_timecourses(NMM_JHU_name, NMM_JHU_ROIs_name, NMM_JHU_ROIs_nb, func):
    """
    Function that calculates the mean timecourse per WM ROI. 
    Only overlap with at least 10 voxels are kept. The timecourse should not be null.

    Args:
        NMM_JHU_name   (pathlib.PosixPath) : path of the NMM-JHU atlas
        NMM_JHU_ROIs_name           (list) : name of the NMM-JHU ROIs
        NMM_JHU_ROIs_nb       (np.ndarray) : index of the NMM-JHU ROIs
        func                  (np.ndarray) : (X, Y, Z, t) fMRI data, for one subject

    """
    
    subject_name = os.path.dirname(NMM_JHU_name)

    # Neuromorphometrics + John Hopkins University atlas
    NMM_JHU_atlas = nib.load(NMM_JHU_name)
    NMM_JHU_atlas = NMM_JHU_atlas.get_fdata()

    NMM_JHU_atlas_masked = deepcopy(NMM_JHU_atlas)
    # Keep only JHU data (indices < 1000)
    NMM_JHU_atlas_masked[NMM_JHU_atlas_masked < 1000] = 0
    # ROIs indices (NMM JHU value), without the 0 (background)
    atlas_ROIs_nb  = np.unique(NMM_JHU_atlas_masked)[1:]
    # ROIs indices (python indices)
    atlas_ROIs_idx = np.array([np.argwhere(NMM_JHU_ROIs_nb == nb)[0][0] for nb in atlas_ROIs_nb])
    
    NMM_JHU_ROIs_name   = np.array(NMM_JHU_ROIs_name)
    mean_ROI_timecourse = pd.DataFrame(columns=NMM_JHU_ROIs_name[atlas_ROIs_idx])

    # Iterate over all ROIs to keep (JHU)
    for idx, ROI in enumerate(atlas_ROIs_nb):
        # Check that at least 10 voxels are present
        if np.sum(NMM_JHU_atlas_masked == ROI) >= 10:
            voxels_ids = np.isin(NMM_JHU_atlas_masked, ROI)
            mean_timecourse = np.mean(func[voxels_ids], axis=0)
            # Check that the timecourse is not null
            if (np.sum(mean_timecourse) != 0):
                mean_ROI_timecourse.iloc[:, idx] = mean_timecourse

    mean_ROI_timecourse = mean_ROI_timecourse.dropna(axis=1, how='all')
    mean_ROI_timecourse.to_csv(f"{subject_name}/WM_timecourses_all.csv", index=False)

def GSR_timecourses(subject_folder, func):
    """
    Function that calculates the mean timecourse for GSR (Global Signal Regression). 
    In short, it means the signal from the whole brain. 

    Args:
        subject_folder (pathlib.PosixPath) : path of the subject
        func                  (np.ndarray) : (X, Y, Z, t) fMRI data, for one subject

    """

    whole_brain_mask_path = subject_folder / "brain_mask_fillh.nii.gz"

    brain_mask = nib.load(whole_brain_mask_path)
    brain_mask = brain_mask.get_fdata() > 0.5

    # Create a 4D mask to avoid loops
    brain_mask_rep = np.repeat(brain_mask[..., np.newaxis], func.shape[3], axis=3)

    GSR = pd.DataFrame(np.mean(func[brain_mask_rep].reshape(-1, func.shape[3]), axis=0), columns=["GSR"])
    GSR.to_csv(subject_folder / "GSR_timecourses_all.csv", index=False)

def NMM_GM_timecourses(NMM_JHU_name, NMM_JHU_ROIs_name, NMM_JHU_ROIs_nb, func):
    """
    Function that calculates the mean timecourse per NMM ROI. 
    Only overlap with at least 10 voxels are kept. The timecourse should not be null.

    Args:
        NMM_JHU_name   (pathlib.PosixPath) : path of the NMM-JHU atlas
        NMM_JHU_ROIs_name           (list) : name of the NMM-JHU ROIs
        NMM_JHU_ROIs_nb       (np.ndarray) : index of the NMM-JHU ROIs
        func                  (np.ndarray) : (X, Y, Z, t) fMRI data, for one subject

    """
    subject_name = os.path.dirname(NMM_JHU_name)
    
    # Neuromorphometrics + John Hopkins University atlas
    NMM_JHU_atlas = nib.load(NMM_JHU_name)
    NMM_JHU_atlas = NMM_JHU_atlas.get_fdata()

    NMM_JHU_atlas_masked = deepcopy(NMM_JHU_atlas)
    # Keep only JHU data (indices < 1000)
    NMM_JHU_atlas_masked[NMM_JHU_atlas_masked >= 1000] = 0    
    
     # ROIs indices (NMM JHU value), without the 0 (background)
    atlas_ROIs_nb  = np.unique(NMM_JHU_atlas_masked)[1:]
    roi_remove_nb  = NMM_JHU_ROIs_nb[np.isin(NMM_JHU_ROIs_name, 
                                     ["Lat. Vent L", "Lat. Vent R", "3rd Ventricle", "4th Ventricle", "CSF", "WM R", "WM L"])]
    # ROIs indices (mask indices)
    atlas_ROIs_nb  = np.delete(atlas_ROIs_nb, np.where(np.isin(atlas_ROIs_nb, roi_remove_nb)))    
    # ROIs indices (python indices)
    atlas_ROIs_idx = np.array([np.argwhere(NMM_JHU_ROIs_nb == nb)[0][0] for nb in atlas_ROIs_nb])
    NMM_JHU_ROIs_name   = np.array(NMM_JHU_ROIs_name)
    mean_ROI_timecourse = pd.DataFrame(columns=NMM_JHU_ROIs_name[atlas_ROIs_idx])

    # Iterate over all ROIs to keep (JHU)
    for idx, ROI in enumerate(atlas_ROIs_nb):
        # Check that at least 10 voxels are present
        if np.sum(NMM_JHU_atlas_masked == ROI) >= 10:
            voxels_ids = np.isin(NMM_JHU_atlas_masked, ROI)
            
            mean_timecourse = np.mean(func[voxels_ids].reshape(-1, func.shape[3]), axis=0)
            # Check that the timecourse is not null
            if (np.sum(mean_timecourse) != 0):
                    mean_ROI_timecourse.iloc[:, idx] = mean_timecourse
    
    mean_ROI_timecourse = mean_ROI_timecourse.dropna(axis=1, how='all')
    mean_ROI_timecourse.to_csv(f"{subject_name}/NMM_GM_timecourses_all.csv", index=False)

def extract_timecourses(path_data, NMM_JHU_ROIs_name, NMM_JHU_ROIs_nb, fieldstrength):
    """
    Function that extracts the timecourses of the different brain regions
    
    Args:
        path_data   (pathlib.PosixPath): path of the data folder 
                                         (either "se_rs" -BOLD- or "diff_rs" -ADC)
        NMM_JHU_ROIs_name       (list) : name of the NMM ROIs
        NMM_JHU_ROIs_nb   (np.ndarray) : index of the NMM ROIs
        fieldstrength            (str) : MRI fieldstrength, either "3T" or "7T"
    
    """
    
    files           = os.listdir(path_data)
    # Get all the subject names within the group
    subject_names   = []
    for fl in files:
        if (fieldstrength in fl) and ("BB" in fl) and (not "FC_" in fl) and (not "p_" in fl):
            subject_names.append(fl)

    for subject in subject_names:

        subject_folder = path_data / subject

        # Neuromorphometric (NMM, gray matter -GM-) and John Hopkins University (JHU, white matter -WM-)
        NMM_JHU_name = subject_folder / "NMM_JHU_atlases_subject_space.nii.gz"

        func, _, _ = load_func(subject_folder)

        # Timecourses of the WM
        WM_timecourses(NMM_JHU_name, NMM_JHU_ROIs_name, NMM_JHU_ROIs_nb, func)

        # Timecourses of the GM
        NMM_GM_timecourses(NMM_JHU_name, NMM_JHU_ROIs_name, NMM_JHU_ROIs_nb, func)

        # Global signal regression - GSR = average signal of the whole brain
        GSR_timecourses(subject_folder, func)

def calculate_correlations(fieldstrength, GM_all_ROIs, WM_all_ROIs, GSR, filenames, path_contrast):
    """
    Function that calculates Pearson's pairwise partial correlations between ROIs
    
    Args:
        fieldstrength           (str) : MRI fieldstrength, either "3T" or "7T"
        GM_all_ROIs            (list) : list of all GM elements presents in the 3 contrasts
        WM_all_ROIs            (list) : list of all WM elements presents in the 3 contrasts
        GSR                    (bool) : whether to use data with global signal regression (GSR) or not
        filenames              (list) : list of names to store the individual and group FC
        path_contrast (pathlib.Posix) : path to the subjects folder for one contrast 
                                        ("se_rs" -BOLD-, "diff_rs" -ADC-)
    """

    files         = os.listdir(path_contrast)
    # Get all the subject names within the group
    subject_names = []
    for fl in files:
        if (fieldstrength in fl) and ("BB" in fl) and (not "FC_" in fl) and (not "p_" in fl):
            subject_names.append(fl)

    all_GM_WM_elements = GM_all_ROIs + WM_all_ROIs
    
    FC_all = np.zeros((len(subject_names), len(all_GM_WM_elements), len(all_GM_WM_elements)))
    for subject_id, subject in enumerate(subject_names):

        FC_subject     = np.zeros((len(all_GM_WM_elements), len(all_GM_WM_elements)))
        subject_folder = path_contrast / subject

        # Load the timecourses
        mean_GM_ROI_timecourses = pd.read_csv(subject_folder / "NMM_GM_timecourses_all.csv")
        mean_WM_ROI_timecourses = pd.read_csv(subject_folder / "WM_timecourses_all.csv")
        mean_GSR_timecourses    = pd.read_csv(subject_folder / "GSR_timecourses_all.csv")
        # GM + WM ROIs
        mean_ROIs_timecourses   = pd.concat([mean_GM_ROI_timecourses, mean_WM_ROI_timecourses], axis=1)



        for i in range(len(all_GM_WM_elements)):
            for j in range(i, len(all_GM_WM_elements)):
                if ((all_GM_WM_elements[i] in mean_ROIs_timecourses.columns) and 
                    (all_GM_WM_elements[j] in mean_ROIs_timecourses.columns)):

                    corr_df = pd.concat([mean_ROIs_timecourses[[all_GM_WM_elements[i]]], 
                                        mean_ROIs_timecourses[[all_GM_WM_elements[j]]], 
                                        mean_GSR_timecourses], 
                                        axis=1)
                    # Same ROI => correlation = 1                    
                    if all_GM_WM_elements[i] == all_GM_WM_elements[j]:
                        FC_subject[i, j] = 1
                        FC_subject[j, i] = 1
                    # One of the timecourse is filled with 0s => np.nan (should not happen)
                    elif (corr_df[all_GM_WM_elements[i]] == 0).all() or (corr_df[all_GM_WM_elements[j]] == 0).all():
                        FC_subject[i, j] = np.nan
                        FC_subject[j, i] = np.nan
                    # Everything is fine
                    else:
                        if GSR:
                            corr = pg.pairwise_corr(
                                                    corr_df,
                                                    columns=[all_GM_WM_elements[i], all_GM_WM_elements[j]],
                                                    covar=corr_df.columns.to_list()[2],
                                                    )
                        else:
                            corr = pg.pairwise_corr(
                                                corr_df,
                                                columns=[all_GM_WM_elements[i], all_GM_WM_elements[j]],
                                                )

                        FC_subject[i, j] = corr["r"].values[0]
                        FC_subject[j, i] = corr["r"].values[0]

                # One of the region is not found in the timecourses
                else:
                    FC_subject[i, j] = np.nan
                    FC_subject[j, i] = np.nan

        np.save(path_contrast / f"{subject}/{filenames[0]}", FC_subject)

        FC_all[subject_id, ...] = FC_subject

    np.save(path_contrast / filenames[1], FC_all)

def load_FC(perc_common, GM_all_ROIs, WM_all_ROIs, FC_name, path_data):
    """
    Function that loads the functional connectivity (FC) map, with regions present in perc_common % of the subjects, 
    in all 2 contrasts
    
    Args:
        perc_common          (float) : between 0 and 1. 
                                       If 0, takes all the regions and do the intercept between contrasts.
                                       If 1, takes only the regions that are present in all the subjects and do the intercept 
                                       between contrasts.
                                       If 0.5, takes only the regions that are present in 50% of the subjects and do the intercept 
                                       between contrasts.
        GM_all_ROIs           (list) : list of all GM ROIs presents in the 3 contrasts
        WM_all_ROIs           (list) : list of all WM ROIs presents in the 3 contrasts
        FC_name                (str) : name of the FC to load. 
                                       f"FC_{GM_region}_JHU_{GM_region}_JHU_GSR_only_all_{fieldstrength}_10vx_reg.npy" 
        path_data (pathlib.PosixPath): path of the data folder 
                                       (either "se_rs" -BOLD- or "diff_rs" -ADC)

    Returns:
        FC_all_contrasts_all_subjects (list) : list of length 3, containing the contrast FC (N_subjects, ROIs, ROIs)
        GM_labels                     (list) : GM labels corresponding to the FC
        WM_labels                     (list) : WM labels corresponding to the FC

    """    
    
    contrasts   = ["se_rs", "diff_rs"]
    common_ROIs = []
    for contrast in contrasts:

        subjects_folder = path_data / contrast
        FC_1_contrast_all_subjects = np.load(subjects_folder / FC_name)

        # Count non-NaN values along the first axis, (Nb_ROIs, Nb_ROIs)
        non_nan_count   = np.sum(~np.isnan(FC_1_contrast_all_subjects), axis=0)
        # Identify pixels where only perc_common % of the subjects are not NaN
        threshold       = perc_common * FC_1_contrast_all_subjects.shape[0]
        # (Nb_selected entries, 2)
        selected_2D_idx = np.argwhere(non_nan_count >= threshold)
        selected_ROI    = np.unique(selected_2D_idx)
        common_ROIs.append(selected_ROI)

    # Keep the ROIs that are common in the 3 contrasts
    ROIs_to_keep = set(common_ROIs[0])
    for i in range(1, len(common_ROIs)):
        ROIs_to_keep = ROIs_to_keep.intersection(set(common_ROIs[i]))
    ROIs_to_keep = np.array(list(ROIs_to_keep))

    # Select subFC
    GM_WM_all_ROIs = np.array(GM_all_ROIs + WM_all_ROIs)
    labels    = GM_WM_all_ROIs[ROIs_to_keep]
    GM_labels = labels[np.isin(labels, GM_all_ROIs)]
    WM_labels = labels[np.isin(labels, WM_all_ROIs)]
    # list of FC for all contrasts, all subjects
    FC_all_contrasts_all_subjects = []
    for j, contrast in enumerate(contrasts):
        subjects_folder = path_data / contrast

        FC_1_contrast_all_subjects    = np.load(subjects_folder / FC_name)

        subFC_1_contrast_all_subjects = np.zeros((FC_1_contrast_all_subjects.shape[0], ROIs_to_keep.size, ROIs_to_keep.size))
        
        for i in range(FC_1_contrast_all_subjects.shape[0]):
            subFC_1_contrast_all_subjects[i, ...] = FC_1_contrast_all_subjects[i, ...][ROIs_to_keep, :][:, ROIs_to_keep]  

        FC_all_contrasts_all_subjects.append(subFC_1_contrast_all_subjects)   

    return FC_all_contrasts_all_subjects, GM_labels, WM_labels   

def read_labels_NMM_JHU(labels_path):
    """
    Function that reads the Neuromorphometrics (NMM, gray matter -GM-) and John Hopkins University (JHU, white matter -WM-) atlas
    
    Args:
        labels_path (pathlib.PosixPath): path of the NMM_JHU labels text file

    Returns:
        NMM_JHU_ROIs_nb   (np.ndarray) : indices of the DMN ROIs
        NMM_JHU_ROIs_name (np.ndarray) : names of the DMN ROIs
    """

    f     = open(labels_path, "r")
    lines = f.readlines()
    # Remove first line corresponding to ROI header
    lines = lines[1:]
    # Contains the id of the regions
    NMM_JHU_ROIs_nb = np.zeros(len(lines))
    # Contains the name of the regions
    NMM_JHU_ROIs_name = []
    for i in range(len(lines)):
        # str to double
        NMM_JHU_ROIs_nb[i] = float(extractBetween(lines[i], "<index>", "</index>"))
        name = extractBetween(lines[i], "<name>", "</name>")
        name = name.replace("Left", "L")
        name = name.replace("Right", "R")
        name = name.replace("gyrus", "G")
        name = name.replace("cortex", "C")
        name = name.replace("the", "")
        if "Cerebellar" in name:
            name = name.replace("Cerebellar Vermal Lobules", "Cb")
        if "Cerebral" in name:
            name = name.replace("Cerebral White Matter", "WM")
        if "Cerebellum" in name:
            name = name.replace("Cerebellum Exterior", "Cb exterior")
            name = name.replace("Cerebellum White Matter", "Cb WM")
        if "Accumbens" in name:
            name = name.replace("Accumbens Area", "NAc")
        name = name.replace("Hippocampus", "Hipp")
        name = name.replace("Lateral Ventricle", "Lat. Vent")
        name = name.replace("Thalamus", "Thal")
        NMM_JHU_ROIs_name.append(name)
    
    return NMM_JHU_ROIs_nb, NMM_JHU_ROIs_name

def wilcoxon_sign_edges(FC):
    """
    Function that calculates the functional connectivity (FC) edges that are significant, using wilcoxon signed-rank tests and 
    correction for multiple comparisons with fdr_bh

    Args:
        FC (np.ndarray) : (Nb_subjects, ROIs, ROIs), FC map
    
    Returns:
        mask (np.ndarray) : (ROIs, ROIs), mask of the significant edges
    """

    # To avoid warnings with arctanh(1), in the diagonal
    FC[:, np.identity(FC.shape[1]).astype("bool")] = 0.99
    # Calculate the significant edges on Fisher-transformed FC matrix
    _, pvals = scipy.stats.wilcoxon(np.arctanh(FC), axis=0, nan_policy='omit')

    tril            = np.tril(pvals)
    tril[tril == 0] = np.nan
    # must remove nan
    flat            = tril.flatten()
    flat_no_nan     = flat[~np.isnan(flat)]
    # Correct for multiple comparisons with FDR
    _, padj, _, _   = multipletests(flat_no_nan, method='fdr_bh')
    tmpl                   = np.zeros_like(tril.flatten())
    tmpl[np.isnan(flat)]   = np.nan
    tmpl[~ np.isnan(flat)] = padj

    corrp = np.reshape(tmpl, tril.shape)
    mask  = corrp < 0.05

    return mask

def order_LR(GM_all_ROIs, WM_all_ROIs):
    """
    Function that re-order the indices of the functional connectivity (FC) matrix so that it is separated right (R) - left (L). 
    The order gray matter - white matter is kept.

    Args:
        GM_all_ROIs   (list) : list of all GM elements presents in the 3 contrasts
        WM_all_ROIs   (list) : list of all WM elements presents in the 3 contrasts
    
    Returns:
        reordered_names (list): list of reordered label names for the FC matrix
    """
     
    l_GM = [
        GM_all_ROIs[index]
        for index, string in enumerate(GM_all_ROIs)
        if string[-1] == "L"
    ]
    r_GM = [
        GM_all_ROIs[index]
        for index, string in enumerate(GM_all_ROIs)
        if string[-1] == "R"
    ]

    other_GM            = list(set(GM_all_ROIs) - set(l_GM) - set(r_GM))
    # If re order idx right-left => non-specified regions, right regions, left regions
    reordered_names_GM  = other_GM + r_GM + l_GM

    l_WM = [
        WM_all_ROIs[index]
        for index, string in enumerate(WM_all_ROIs)
        if string[-1] == "L"
    ]
    r_WM = [
        WM_all_ROIs[index]
        for index, string in enumerate(WM_all_ROIs)
        if string[-1] == "R"
    ]

    other_WM           = list(set(WM_all_ROIs) - set(l_WM) - set(r_WM))
    # If re order idx right-left => non-specified regions, right regions, left regions
    reordered_names_WM = other_WM + r_WM + l_WM

    reordered_names = reordered_names_GM + reordered_names_WM

    return reordered_names

def plot_mean_FC(FC_df, labels, colors, savepath, field, contrast, plot):

    """
    Function that plots the mean functional connectivity (FC) matrix 
    
    Args:
        FC_df      (pd.DataFrame): FC matrices of all subjects
        labels            (list) : list of all GM and WM elements
        colors            (list) : list of string, containing the colors corresponding to the labels, for the dendrogram bar
        savepath (path.PosixPath): path where to save the matrices
        field               (str): either "3T" or "7T"
        contrast            (str): either "se_rs" (BOLD) or "diff_rs" (ADC-fMRI)
        plot               (bool): if the plot should be displayed or not
    
    """

    names = deepcopy(labels)
    for i in range(len(labels)):
        if (i + 1 < len(labels)):
            if (labels[i][:-2] == labels[i+1][:-2]):
                names[i] = ""
                names[i+1] = names[i+1][:-2]
    min_lim = -0.8
    max_lim = 0.8
    g = sns.clustermap(
                       FC_df.loc[labels, labels], 
                       xticklabels=labels, 
                       yticklabels=labels, 
                       vmin=min_lim, 
                       vmax=max_lim, 
                       cmap="turbo", 
                       row_cluster=False, 
                       col_cluster=False,                                
                       row_colors=[colors],
                       col_colors=[colors],
                       figsize=fig_size(1)
                       )

    g.cax.set_visible(False)
    col = list(set(colors))
    for label in range(2):
        g.ax_col_dendrogram.bar(
                                0,
                                0,
                                color=col[label],
                                label=["White Matter" if "rosybrown" in col[label] else "Gray Matter"],
                                linewidth=0,
                                )
    g.gs.update(bottom=0.2, top=1)

    cmap = mpl.cm.turbo
    norm = mpl.colors.Normalize(vmin=min_lim, vmax=max_lim)
    #create new gridspec for the right part
    gs2 = mpl.gridspec.GridSpec(1, 1, top=0.1, bottom=0.09, left=0.45, right=0.68)
    # create axes within this new gridspec
    ax2 = g.fig.add_subplot(gs2[0])
    cb1 = mpl.colorbar.ColorbarBase(ax2, cmap=cmap, norm=norm, orientation="horizontal")
    cb1.ax.set_title('Pearson correlation')
    cb1.set_label('Functional connectivity')

    if plot:
        plt.subplots_adjust(top = 1, bottom = 0, right = 0.9, left = 0, 
            hspace = 0, wspace = 0)
        plt.margins(0, 0)
        plt.tight_layout()
        plt.show()

    g.savefig(f"{savepath}/FC_{contrast}_{field}.pdf")
    g.savefig(f"{savepath}/FC_{contrast}_{field}.png", bbox_inches='tight', dpi=500)
