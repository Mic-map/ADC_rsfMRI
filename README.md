<div style="text-align: center;">
<img src="https://wp.unil.ch/mic-map/files/2022/01/cropped-2-MicrostructureMappingLab-01.png" width="50%" style="background-color:white;padding:20px;" >
</div>

# Apparent Diffusion coefficient fMRI shines a new light on white matter resting-state connectivity, as compared to BOLD

Ines de Riedmatten<sup>1,2∗</sup>, Arthur P C Spencer<sup>1,2</sup>, Wiktor Olszowy<sup>3,4</sup>, Ileana O Jelescu<sup>1,2</sup>

<sup>1</sup>Department of Radiology, Lausanne University Hospital (CHUV), Lausanne, Switzerland,

<sup>2</sup>School of Biology and Medicine, University of Lausanne, Lausanne, Switzerland,

<sup>3</sup>Ecole Polytechnique Fédérale de Lausanne (EPFL), Lausanne, Switzerland,

<sup>4</sup>Data Science Unit, Science and Research, dsm-firmenich AG, Kaiseraugst, Switzerland,


<sup>*</sup>Corresponding author: Ines de Riedmatten (ines.de-riedmatten@chuv.ch, https://wp.unil.ch/mic-map/)


## Introduction
This git summarizes the code needed to process the data and generate the paper figures. The data will be made available upon publication.

## Table of Contents
- [Introduction](#introduction)
- [Getting Started](#getting-started)
  - [Installation](#installation)
  - [Running the Program](#running-the-program)
- [Files description](#files-description)

## Getting Started

### Installation
To run the different scripts, you will need a minimal conda environment : 
  ```shell
  git clone https://github.com/ideriedm/ADC_rsfMRI.git
  conda env create -f env.yml
  conda activate rs_env
  ```

### Running the Program
Modify the params.json:
  - data_path   : path to your data folder (where "se_rs" and "diff_rs" are stored)
  - savefig_path: path where you want to save the figures

Each script should be run as:
  ```shell
  cd ADC_rsfMRI/ADC_rsfMRI
  python name_of_the_script.py
  ```
   
## Files description

`Correlations_agreement.py` (Fig. 1): File that plots the ADC-to-BOLD agreement of mean functional connectivity (FC).

`FC_amplitude_per_regions.py` (Fig. 2): Function that separates FC edges amplitudes according to 3 connectivity regions:
- gray-to-gray matter (GM-GM)
- gray-to-white matter (GM-WM)
- white-to-white matter (WM-WM)

`FC_matrix.py` (Fig. S2): File that plots the ADC vs BOLD mean functional connectivity (FC).

`fig_style.py`: File that handles the plotting settings.

`graph_subject.py` (Fig. 3): File that calculates subjectwise weighted graph metrices in two connectivity regions:
- gray-to-gray matter (GM-GM)
- white-to-white matter (WM-WM)

`similarity.py` (Fig. 4): Function that calculates the inter-subject similarity by calculating the Pearson's correlation coefficient between the significant edges, excluding self-connections, split in 3 connectivity regions:
- gray-to-gray matter (GM-GM)
- gray-to-white matter (GM-WM)
- white-to-white matter (WM-WM)

`slab.py` (Fig. S1): File that plots the common slab between all ADC-fMRI volumes, in the MNI space, at 3T.

`utils.py`: File containing the utilitary functions.



