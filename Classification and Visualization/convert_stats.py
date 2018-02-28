from nipype.interfaces.freesurfer import MRIConvert
import os
import shutil
import csv
import sys
import csv

### creating a dictionary of oasis dataset labels
with open('oasis_cross-sectional.csv', 'r') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',')
    cdr = {'0': [], '0.5': [], '1': [], '2': []}
    for row in spamreader:
        dict = {'id': row[0], 'sex': row[1], 'Hand': row[2], 'Age': row[3],
                'Educ': row[4], 'SES': row[5], 'MMSE': row[6], 'CDR': row[7],
                'eTIV': row[8], 'nWBV': row[9], 'ASF': row[10], 'Delay': row[11]}

        if dict['CDR'] == '0':
            cdr['0'].append(int(row[0].split('_')[1]))
        elif dict['CDR'] == '0.5':
            cdr['0.5'].append(int(row[0].split('_')[1]))
        elif dict['CDR'] == '1':
            cdr['1'].append(int(row[0].split('_')[1]))
        elif dict['CDR'] == '2':
            cdr['2'].append(int(row[0].split('_')[1]))

print('CDR = 0: \n', [int(i) for i in cdr['0']])
print('CDR = 0.5: \n', [int(i) for i in cdr['0.5']])
print('CDR = 1: \n', [int(i) for i in cdr['1']])
print('CDR = 2: \n', [int(i) for i in cdr['2']])
##########################################################

stats2data = None
if len(sys.argv) > 1:
    aseg = sys.argv[1]

path = '/media/Ali/8A9E6F039E6EE6E3/freesurfer/subjects'
SUBJECTS = sorted([d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d)) and d.startswith('sub')])

if stats2data == 'stats2data':

    # directories for output files
    outdirs = ['stats/aseg_stats', 'stats/r_aparc_stats', 'stats/l_aparc_stats']
    for out_dir in outdirs:
        if os.path.exists(out_dir):
            shutil.rmtree(out_dir)

        os.makedirs(out_dir)


    # extracting stats from subjects
    for sub in SUBJECTS:
        # meas can be in: [area, volume (ie, volume of gray matter), thickness, thicknessstd, or meancurv. thicknessstd]
        cmd_aseg = 'asegstats2table -s {subject} --meas {meas} --tablefile aseg_stats/aseg_stats_{subject}.txt'\
                        .format(subject=sub, meas='volume')
        cmd_l_aparc = 'aparcstats2table -s {subject} --hemi lh --meas {meas} --tablefile l_aparc_stats/l_aparc_stats_{subject}.txt'\
                        .format(subject=sub, meas='volume')
        cmd_r_aparc = 'aparcstats2table -s {subject} --hemi rh --meas {meas} --tablefile r_aparc_stats/r_aparc_stats_{subject}.txt'\
                        .format(subject=sub, meas='volume')

        os.system(cmd_aseg)
        os.system(cmd_l_aparc)
        os.system(cmd_r_aparc)

    print('\n\n{0} Subjects Processed!'.format(len(SUBJECTS)))

else:

    ## adding labels and age feature to the output dataset
    data = []
    for sub in SUBJECTS:

        ## reading stats files for segmentation, right parcellation, and left parcellation
        aseg_f = open('aseg_stats/aseg_stats_{subject}.txt'.format(subject=sub))
        aseg_reader = csv.reader(aseg_f, dialect='excel', delimiter='\t')

        l_aparc_f = open('l_aparc_stats/l_aparc_stats_{subject}.txt'.format(subject=sub))
        l_aparc_reader = csv.reader(l_aparc_f, dialect='excel', delimiter='\t')

        r_aparc_f = open('r_aparc_stats/r_aparc_stats_{subject}.txt'.format(subject=sub))
        r_aparc_reader = csv.reader(r_aparc_f, dialect='excel', delimiter='\t')

        counter = 0
        for aseg_row, l_aparc_row, r_aparc_row in zip(aseg_reader, l_aparc_reader, r_aparc_reader):

            ## extracting id from subject name (convention: sub + id (for ex: sub32 or sub95)
            id = int(sub[3:])

            ## adding label
            if id in cdr['0']:
                output = 0
            elif id in cdr['0.5']:
                output = 1
            elif id in cdr['1']:
                output = 1
            else:
                output = 1

            ## if the first item, we add the description of each feature in the first row
            ## in the next .txt files, we just add features, not the description like panda data format
            if counter == 1:
                datasheet_f = open('oasis_cross-sectional.csv')
                datasheet_reader = csv.reader(datasheet_f, dialect='excel', delimiter='\t')

                for row in datasheet_f:
                    if row.startswith('O'):
                        new_id = int(row.split(',')[0].split('_')[1])

                        if new_id == id:
                            age = int(row.split(',')[3])

                data.append(aseg_row + l_aparc_row[1:] + r_aparc_row[1:] + [age, output, ])
                counter = 0

            ## for the first time, we add the description of each feature
            elif len(data) == 0:
                data.append(aseg_row + l_aparc_row[1:] + r_aparc_row[1:] + ['age', 'output', ])
                counter += 1
            else:
                ## this is the line of description which is repeatitive in all .txt files, and we just skip!
                counter += 1

    ### writing dataset into a file
    with open("data.csv","w",newline="") as f:
        cw = csv.writer(f)
        cw.writerows(r+[""] for r in data)





