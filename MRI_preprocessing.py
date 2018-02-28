from nipype.interfaces.freesurfer import ReconAll
import os
import shutil
import sys

##################################################
FREESURFER_HOME = os.environ.get('FREESURFER_HOME')
reconall = ReconAll()

# path to the subject T1 MRI image in .nii format
# this file must be in .nii format
nii_file_path = 'nii_file_path'

# Subject ID: this should be a string which defines the folder name 
# that will be created automatically for the subject in /usr/local/freesurfer/subjects/
subject_id = 'subject_id'

print('Data preprocessing started for {subject} in {path}'.format(subject=subject_id, path=nii_file_path))

reconall.inputs.subject_id = subject_id
reconall.inputs.directive = 'all'
reconall.inputs.subjects_dir = FREESURFER_HOME + '/subjects'
reconall.inputs.T1_files = nii_file_path
reconall.run()