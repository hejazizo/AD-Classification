from scipy import stats
import pandas as pd
import csv
import numpy as np

test = 't-test'


if test == 't-test':

    t_test_dic = {'Welch': {'RF - Logit': {}, 'RF - SVM': {}, 'Logit - SVM': {}},
              'Standard': {'RF - Logit': {}, 'RF - SVM': {}, 'Logit - SVM': {}}}

    flag = False
    for test_type in ['Welch', 'Standard']:
        if test_type == 'Welch':
            flag = True
        elif test_type == 'Standard':
            flag = False

        for criterion in ['Test Accuracy', 'AUC', 'Precision', 'Recall']:

            my_files = ['RF', 'Logit']
            for file in my_files:
                # --------------------------------- #
                ######    RF -> Logit, SVM   ########
                # --------------------------------- #
                if file == 'RF':
                    rf_df = pd.read_csv(file + '.csv')
                    rf_c = rf_df[criterion]

                    for sec_file in ['Logit', 'SVM']:
                        sec_file_df = pd.read_csv(sec_file + '.csv')
                        sec_file_c = sec_file_df[criterion]

                        t_stats = stats.ttest_ind(rf_c, sec_file_c, equal_var=flag)
                        t_test_dic[test_type]['{0} - {1}'.format(file, sec_file)][criterion]= t_stats[1]

                # --------------------------- #
                #####    Logit -> SVM    ######
                # --------------------------- #
                elif file == 'Logit':
                    logit_df = pd.read_csv(file + '.csv')
                    logit_c = logit_df[criterion]

                    for sec_file in ['SVM']:
                        sec_file_df = pd.read_csv(sec_file + '.csv')
                        sec_file_c = sec_file_df[criterion]

                        t_stats = stats.ttest_ind(logit_c, sec_file_c, equal_var=flag)
                        t_test_dic[test_type]['{0} - {1}'.format(file, sec_file)][criterion] = t_stats[1]


    welch_csvfile = open('Welch.csv', 'w')
    welch_writer = csv.writer(welch_csvfile)
    welch_writer.writerow(['Welch', 'Test Accuracy', 'AUC', 'Precision', 'Recall'])

    st_csvfile = open('Standard.csv', 'w')
    st_writer = csv.writer(st_csvfile)
    st_writer.writerow(['Standard', 'Test Accuracy', 'AUC', 'Precision', 'Recall'])

    print(t_test_dic)
    for test, test_values in t_test_dic.items():
        for alg_pair, test_values in test_values.items():
            # criterion, value = values.items()
            # print(alg_pair, test_values.keys(), test_values.values())
            print(test_values)

            if test == 'Welch':
                welch_writer.writerow([alg_pair, test_values['Test Accuracy'],
                    test_values['AUC'], test_values['Precision'], test_values['Recall']])
            else:
                st_writer.writerow([alg_pair, test_values['Test Accuracy'],
                    test_values['AUC'], test_values['Precision'], test_values['Recall']])

    st_csvfile.close()
    welch_csvfile.close()



### ANOVA TEST
elif test == 'ANOVA':

    std_csvfile = open('std.csv', 'w')
    std_writer = csv.writer(std_csvfile)
    std_writer.writerow(['std', 'Logit', 'RF', 'SVM'])

    ANOVA_csvfile = open('ANOVA.csv', 'w')
    ANOVA_writer = csv.writer(ANOVA_csvfile)
    ANOVA_writer.writerow(['Test/Criterion', 'ANOVA'])

    rf_df = pd.read_csv('RF.csv')
    svm_df = pd.read_csv('SVM.csv')
    logit_df = pd.read_csv('Logit.csv')

    for criterion in ['Test Accuracy', 'AUC', 'Precision', 'Recall']:

        rf_c = rf_df[criterion]
        svm_c = svm_df[criterion]
        logit_c = logit_df[criterion]

        rf_std = np.std(rf_c, ddof=1)
        svm_std = np.std(svm_c, ddof=1)
        logit_std = np.std(logit_c, ddof=1)

        std_writer.writerow([criterion, logit_std, rf_std, svm_std])
        ANOVA_writer.writerow([criterion, stats.f_oneway(rf_c, svm_c, rf_c)[1]])

        # print(stats.kruskal(nn_ta, svm_ta, nn_ta))

    std_csvfile.close()

