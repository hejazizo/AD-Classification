# Alzheimer's Disease Classification
Diagnosis of Alzheimer’s Disease Based on Structural MRI Images using Machine Learning Techniques

## Dataset
Open Access Series of Imaging Studies (OASIS) dataset which is publicly available <a href="http://www.oasis-brains.org/">here</a>.

## Requirements
1. Download and install Freesurfer for MRI images preprocessing from <a href="https://surfer.nmr.mgh.harvard.edu/fswiki/DownloadAndInstall">here</a> (Follow the instruction in Freesurfer folder to run it on your VM).
2. Download and install FSL for visualization from <a href="https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FslInstallation">here</a>.
3. <a href="https://pypi.python.org/pypi/nipype/">nipype</a> python library

## Data Preprocessing
1. Convert MRI images format to .nii file.
2. Run all preprocessing stages on .nii MRI files with `MRI_preprocess.py` file. <br />
**Note:** You have to provide `subject_id` and `nii_file_path` in the `.py` file.

## Feature Extraction
