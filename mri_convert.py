from nipype.interfaces.freesurfer import MRIConvert
import os

import shutil
import csv
import sys
import csv

from os import listdir
from os.path import isfile, join


############# MRI CONVERT ##############
sub_path = '/media/Ali/8A9E6F039E6EE6E3/freesurfer/subjects'
output_dir = '/media/Ali/8A9E6F039E6EE6E3/MRI/MRI-Converted'
dirs = [f for f in listdir(sub_path) if f.startswith('sub') == True]
mc = MRIConvert()
for sub in dirs:
    mgz_path = os.path.join(sub_path, sub, 'mri/brain.mgz')
    nii_path = os.path.join(output_dir, '{}-brain.nii'.format(sub))

    mc.inputs.in_file = mgz_path
    mc.inputs.out_file = nii_path
    mc.inputs.out_type = 'nii'
    mc.run()
