import csv


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
        # print(type(dict['CDR']))
        # print(dict['CDR'] == 1)

        if dict['CDR'] == '0':
            cdr['0'].append(row[0].split('_')[1])
        elif dict['CDR'] == '0.5':
            cdr['0.5'].append(row[0].split('_')[1])
        elif dict['CDR'] == '1':
            cdr['1'].append(row[0].split('_')[1])
        elif dict['CDR'] == '2':
            cdr['2'].append(row[0].split('_')[1])



print('CDR = 0: \n', [int(i) for i in cdr['0']])
print('CDR = 0.5: \n', [int(i) for i in cdr['0.5']])
print('CDR = 1: \n', [int(i) for i in cdr['1']])
print('CDR = 2: \n', [int(i) for i in cdr['2']])


#######################################################
import os
import shutil
import re
from os import walk

des = '/media/Ali/8A9E6F039E6EE6E3/Academic/M.Sc (Computer Science)/CMPUT 697/project/MRI DATASET/hdr-img/'
if os.path.exists(des):
    shutil.rmtree(des)
if not os.path.exists(des):
    os.makedirs(des)
    os.chdir(des)
    os.makedirs('0')
    os.makedirs('0.5')
    os.makedirs('1')
    os.makedirs('2')

source_pattern = '/media/Ali/8A9E6F039E6EE6E3/Academic/M.Sc \(Computer Science\)/CMPUT 697/project/MRI DATASET/DISKs'

# ONLY MR1 ARE CONSIDERED
processed_files_pattern = r'/OAS1[_](\d+)[_]MR1/PROCESSED/MPRAGE/' #
pattern = re.compile(source_pattern + processed_files_pattern)

check_missing_files = list(range(1,458))
double_scan = []
#
# file_c = 0
# path = '/media/Ali/8A9E6F039E6EE6E3/Academic/M.Sc (Computer Science)/CMPUT 697/project/MRI DATASET/DISKs'
# for (dirpath, dirnames, filenames) in walk(path):
#
#     res = re.search(pattern, dirpath)
#     if res is not None:
#         id = res.group(1)
#         source = res.group(0) + 'T88_111/'
#
#
#         file_pattern = 'OAS1[_]\d+[_]MR1[_]mpr[_]n\d+[_]anon[_]111[_]t88_gfc(.hdr|.img)'
#
#         for file in filenames:
#             res_file = re.search(file_pattern, file)
#             # print(file)
#             if res_file is not None:
#                 file_c += 1
#                 file_name = res_file.group(0)
#                 source_file_path = source + file_name
#
#                 # if id in cdr['0']:
#                 #     shutil.copyfile(source_file_path, des + '0/' + file_name)
#                 #
#                 # elif id in cdr['0.5']:
#                 #     shutil.copyfile(source_file_path, des + '0.5/' + file_name)
#                 #
#                 # elif id in cdr['1']:
#                 #     shutil.copyfile(source_file_path, des + '1/' + file_name)
#                 #
#                 # elif id in cdr['2']:
#                 #     shutil.copyfile(source_file_path, des + '2/' + file_name)
#
#                 if file_c % 2:
#                     print(id, (file_c + 1)//2)
#                     try:
#                         check_missing_files.remove(int(id))
#                     except:
#                         double_scan.append(int(id))
#
# print('Missing Files: ', check_missing_files, ' ----- Total: ', len(check_missing_files))
# print('Double Scan (MR1 & MR2): ', double_scan, ' ----- Total: ', len(double_scan))
# print('Number of total files (hdr img pairs) processed: ', file_c/2)