# Classification and Visualization
1. Converting preprocessing stats files to a `.csv` dataset file including features and labels.
After preprocessing stage, there are stats files for each subject in freesurfer subjects folder. Convert them into a `.csv` file using `mri_convert.py` file. 
     1. Run `python mri_convert stats2data` to convert stats files to features.
     2. Run `python mri_convert` to add labels and age feature for each subject. By running this command, `data.csv` file will be produced.
    
2. Visualization
Run `python script_visualization path` to produce images in the path specified by command line (for ex: `./image` folder in the current directory)

3. Classification
