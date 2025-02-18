"""
Plot the common slab between all ADC-fMRI volumes. 
For each subject, it binarizes the slab, and then calculates the average between subjects
"""

from pathlib import Path
import numpy as np 
import sys
import os
sys.path.insert(1, os.path.dirname(sys.path[0]))
from utils import save_nifti
import os
import nibabel as nib
import json

params_path = 'params.json'

# Read and parse the JSON file
with open(params_path, 'r') as file:
    params = json.load(file)

data_folder     = Path(f"{params["paths"]["data_path"]}")
ADC_folder      = data_folder / "diff_rs/"
files           = os.listdir(ADC_folder)
fieldstrength   = "3T"
# Get all the subject names within the group
subject_names   = []
for fl in files:
    if (fieldstrength in fl) and ("BB" in fl) and (not "FC_" in fl) and (not "p_" in fl):
        subject_names.append(fl)

func_path   = ADC_folder / subject_names[0] / "registration/func2mni.nii.gz"
func        = nib.load(func_path)
header      = func.header
affine      = func.affine
func        = func.get_fdata()
common_slab = np.zeros((func.shape[0], func.shape[1], func.shape[2]))

for subject in subject_names:

    subject_folder = ADC_folder / subject
    func_path      = subject_folder / "registration/func2mni.nii.gz"

    func   = nib.load(func_path)
    func   = func.get_fdata()
    # Calculate the temporal mean
    func   = np.mean(func, axis=3)
    # Binarize the slab
    func   = func > 0

    common_slab = common_slab + func

save_nifti(common_slab, ADC_folder / "common_slab.nii.gz", affine, header)