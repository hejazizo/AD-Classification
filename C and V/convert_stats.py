from nipype.interfaces.freesurfer import MRIConvert
import os
import shutil
import csv
import sys
import csv

### creating a dictionary of oasis dataset labels
with open('oasis_cross-sectional.csv', 'r') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',')
    counter = -1


    cdr = {'0': [], '0.5': [], '1': [], '2': []}
    for row in spamreader:
        counter = counter + 1
        # if counter == 0:
        #     continue

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

aseg = None
if len(sys.argv) > 1:
    aseg = sys.argv[1]

path = '/media/Ali/8A9E6F039E6EE6E3/freesurfer/subjects'

SUBJECTS = sorted([d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d)) and d.startswith('sub')])

# exclude = [130, 134, 179, 21, 226, 240, 243, 291, 308, 31, 65, 73, 86, 94, 98]
# SUBJECTS = [i for i in SUBJECTS if i not in ['sub{}'.format(i) for i in exclude]]

print(SUBJECTS)


if aseg == 'aseg':

    outdirs = ['aseg_stats', 'r_aparc_stats', 'l_aparc_stats']
    for out_dir in outdirs:
        if os.path.exists(out_dir):
            shutil.rmtree(out_dir)

        os.mkdir(out_dir)

    for sub in SUBJECTS:

        cmd_aseg = 'asegstats2table -s {subject} --meas {meas} --tablefile aseg_stats/aseg_stats_{subject}.txt'\
                        .format(subject=sub, meas='volume')

        # area, volume (ie, volume of gray matter), thickness, thicknessstd, or meancurv. thicknessstd
        cmd_l_aparc = 'aparcstats2table -s {subject} --hemi lh --meas {meas} --tablefile l_aparc_stats/l_aparc_stats_{subject}.txt'\
                        .format(subject=sub, meas='volume')
        cmd_r_aparc = 'aparcstats2table -s {subject} --hemi rh --meas {meas} --tablefile r_aparc_stats/r_aparc_stats_{subject}.txt'\
                        .format(subject=sub, meas='volume')

        os.system(cmd_aseg)
        os.system(cmd_l_aparc)
        os.system(cmd_r_aparc)

    print('\n\n{0} Subjects Processed!'.format(len(SUBJECTS)))

else:

    data = []
    for sub in SUBJECTS:

        aseg_f = open('aseg_stats/aseg_stats_{subject}.txt'.format(subject=sub))
        aseg_reader = csv.reader(aseg_f, dialect='excel', delimiter='\t')

        l_aparc_f = open('l_aparc_stats/l_aparc_stats_{subject}.txt'.format(subject=sub))
        l_aparc_reader = csv.reader(l_aparc_f, dialect='excel', delimiter='\t')

        r_aparc_f = open('r_aparc_stats/r_aparc_stats_{subject}.txt'.format(subject=sub))
        r_aparc_reader = csv.reader(r_aparc_f, dialect='excel', delimiter='\t')

        counter = 0
        for aseg_row, l_aparc_row, r_aparc_row in zip(aseg_reader, l_aparc_reader, r_aparc_reader):

            id = int(sub[3:])

            if id in cdr['0']:
                output = 0
            elif id in cdr['0.5']:
                output = 1
            elif id in cdr['1']:
                output = 1
            else:
                output = 1






            if counter == 1:
                datasheet_f = open('oasis_cross-sectional.csv')
                datasheet_reader = csv.reader(datasheet_f, dialect='excel', delimiter='\t')

                for row in datasheet_f:
                    if row.startswith('O'):
                        new_id = int(row.split(',')[0].split('_')[1])

                        if new_id == id:
                            age = int(row.split(',')[3])
                            # print(new_id, age)

                data.append(aseg_row + l_aparc_row[1:] + r_aparc_row[1:] + [age, output, ])
                counter = 0
            elif len(data) == 0:
                data.append(aseg_row + l_aparc_row[1:] + r_aparc_row[1:] + ['age', 'output', ])
                counter += 1
            else:
                counter += 1


    with open("data.csv","w",newline="") as f:
        cw = csv.writer(f)
        cw.writerows(r+[""] for r in data)





