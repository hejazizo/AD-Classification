import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
# import statsmodels.api as sm
from statsmodels.nonparametric.kde import KDEUnivariate
from statsmodels.nonparametric import smoothers_lowess
from pandas import Series, DataFrame
from patsy import dmatrices
from sklearn import datasets, svm
# from KaggleAux import predict as ka # see github.com/agconti/kaggleaux for more details

import numpy as np
from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
### CLASSIFICATION algorithms
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn import linear_model
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve, auc
from sklearn import metrics

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler

import matplotlib.pylab as plt
import matplotlib.patches as patches
from scipy import interp

from sklearn.preprocessing import MinMaxScaler

import csv
from sklearn.model_selection import StratifiedKFold

#### LOADING DATA
alzheimer_df = pd.read_csv("data.csv")

#### PREPROCESSING ####
############################################

# we map each title
y = alzheimer_df.output
X = alzheimer_df.drop(['output'], axis=1)
X.info()


### CLASSIFICATION algorithm
n_classes = 2
classification_alg = {
                    # 'SVM': svm.SVC(C=1, kernel='linear', probability=True),
                    # 'NN': MLPClassifier(solver='lbfgs', alpha=1e-3, hidden_layer_sizes=(10, n_classes), activation= 'logistic'),
                    'RF': RandomForestClassifier(n_estimators=10, max_depth=3),
                    #  'Logistic': LogisticRegression(C=1.0, tol=1e-4, penalty='l2', solver='lbfgs')
                    }


#########################
scaler = MinMaxScaler()
results = {}

defected_set = []
numruns = 1
avg_accuracy = {'RF': {}, 'NN': {}, 'SVM': {}, 'Logistic': {}}
for i in range(numruns):

    for clf_name, clf in classification_alg.items():
        avg_accuracy[clf_name]['Run {}'.format(i)] = 0
        print('\t{1}\n\tClassification Algorithm: {0}\n\t{1}'.format(clf_name, '-' * 40))

        ### SPLITTING dataset to test and train data
        k_num = 10
        kf = StratifiedKFold(n_splits=k_num, shuffle=True)

        k_counter = 0

        csv_path = './CSV/'
        if clf_name == 'Logistic':
            csvfile = open(csv_path + 'Logistic.csv', 'w')
            log_writer = csv.writer(csvfile)
            log_writer.writerow(['Logit', 'Train Accuracy', 'Test Accuracy', 'TN', 'FP', 'FN', 'TP', 'AUC', 'Tol', 'C'])
        elif clf_name == 'SVM':
            csvfile = open(csv_path + 'SVM.csv', 'w')
            svm_writer = csv.writer(csvfile)
            svm_writer.writerow(['SVM', 'Train Accuracy', 'Test Accuracy', 'TN', 'FP', 'FN', 'TP', 'AUC', 'C', 'Gamma', 'Kernel Type'])
        elif clf_name == 'NN':
            csvfile = open(csv_path + 'NN.csv', 'w')
            nn_writer = csv.writer(csvfile)
            nn_writer.writerow(['NN', 'Train Accuracy', 'Test Accuracy', 'TN', 'FP', 'FN', 'TP', 'AUC', 'nh', 'alpha'])
        elif clf_name == 'RF':
            csvfile = open(csv_path + 'RF.csv', 'w')
            rf_writer = csv.writer(csvfile)
            rf_writer.writerow(['RandomFrest', 'Train Accuracy', 'Test Accuracy', 'TN', 'FP', 'FN', 'TP', 'AUC'])

        for train_index, test_index in kf.split(X, y):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            # print('Train/Test Size: {0} / {1}'.format(len(X_train), len(X_test)))



            sampleindices = X_test['Measure:volume']

            X_train = X_train.drop(['Measure:volume'], axis = 1)
            X_test = X_test.drop(['Measure:volume'], axis = 1)



            #####################
            # features selection
            clf_f_selection = RandomForestClassifier(n_estimators=50, max_features='sqrt')
            clf_f_selection.fit(X_train, y_train)
            print('Feature Selection Accuracy: ', clf_f_selection.score(X_test, y_test))

            features = pd.DataFrame()
            features['feature'] = X_train.columns
            features['importance'] = clf_f_selection.feature_importances_
            features.sort_values(by=['importance'], ascending=False, inplace=True)

            features = features[:5]
            features.sort_values(by=['importance'], ascending=True, inplace=True)

            features.set_index('feature', inplace=True)
            plt.yticks(size=5)

            features.plot(kind='barh', figsize=(20, 10))
            font = {
                # 'weight' : 'bold',
                'size': 15}

            plt.rc('font', **font)
            plt.show()
            print('Done')

            model = SelectFromModel(clf_f_selection, prefit=True)
            train_reduced = model.transform(X_train)
            test_reduced = model.transform(X_test)

            X_train = train_reduced
            X_test = test_reduced

            print(X_test.shape)
        #
        #
        #     #####################
        #     # feature scaling
        #     scaler.fit(X_train)
        #     scaler.transform(X_test)
        #
        #     print('Train/Test Size: {0} / {1}'.format(len(X_train), len(X_test)))
        #     # Parameter Tuning
        #     pt_k_num = 5  # number of folds for parameter tuning
        #
        #     if clf_name == 'RF':
        #         results['RF'] = {
        #             'Train_score':0,
        #             'Test_score': 0,
        #             # 'n_estimator': 0,
        #             # 'mex_depth': 0,
        #             'Confusion Matrix': [],
        #             # 'NN': 0
        #         }
        #         clf.fit(X_train, y_train)
        #         results['RF']['Train_score'] = clf.score(X_train, y_train)
        #         results['RF']['Test_score'] = clf.score(X_test, y_test)
        #
        #         # FIND defected images
        #         if results['RF']['Test_score'] < 0.7:
        #             defected_set = defected_set + list(sampleindices)
        #
        #
        #         pred = clf.predict(X_test)
        #         confusion_matrix = metrics.confusion_matrix(y_test, pred)
        #         fpr, tpr, thresholds = metrics.roc_curve(y_test, pred, pos_label=1)
        #         AUC = metrics.auc(fpr, tpr)
        #
        #         results['RF']['Confusion Matrix'] = confusion_matrix
        #         results['RF']['AUC'] = AUC
        #
        #         rf_writer.writerow(['$K_{}$'.format(k_counter),
        #                             results['RF']['Train_score'],
        #                             results['RF']['Test_score'],
        #                             confusion_matrix[0][0], confusion_matrix[0][1], confusion_matrix[1][0],
        #                             confusion_matrix[1][1], AUC])
        #
        #         print('Train Score: {} %'.format(clf.score(X_train, y_train) * 100))
        #         print('Test Score: {} %\n'.format((results['RF']['Test_score']) * 100))
        #
        #
        #     if clf_name == 'NN':
        #         results['NN'] = {
        #             # 'Train_score':0,
        #             'Test_score': 0,
        #             'Best_nh': 0,
        #             'Best_alpha': 0,
        #             'Confusion Matrix': [],
        #             'NN': 0
        #         }
        #         best_alpha = 0
        #         best_alpha_score = 0
        #         best_nh_glob_score = 0
        #         best_nh_glob = 0
        #         for alph in [1e-0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5]:
        #             clf.set_params(alpha=alph)
        #             best_NN_score = 0
        #             best_nh = 0
        #             for tuning_param in range(2, n_classes + 10):
        #                 clf.set_params(hidden_layer_sizes=(tuning_param, n_classes))
        #                 scores = cross_val_score(clf, X_train, y_train, cv=pt_k_num, scoring='accuracy')
        #
        #                 if scores.mean() >= best_NN_score:
        #                     best_NN_score = scores.mean()
        #                     best_nh = tuning_param
        #                     # print('Best Score: {0}, Best nh: {1}'.format(best_NN_score, best_nh))
        #                 else:
        #                     pass
        #                     # print('No improvement for nh: {0}, Score: {1}'.format(tuning_param, scores.mean()))
        #
        #             score = best_NN_score
        #             if best_NN_score >= best_nh_glob_score:
        #                 best_nh_glob_score = best_NN_score
        #                 best_nh_glob = best_nh
        #                 # print('Best Global Score: {0}, Best Global nh: {1}'.format(best_nh_glob_score, best_nh_glob))
        #             else:
        #                 pass
        #                 # print('No improvement for Global nh: {0}, Score: {1}'.format(best_nh_glob, scores))
        #
        #             if score >= best_alpha_score:
        #                 best_alpha_score = score
        #                 best_alpha = alph
        #                 results['NN']['Best_nh'] = best_nh_glob
        #                 results['NN']['Best_alpha'] = best_alpha
        #                 # print('Best Score: {0}, Best alpha: {1}'.format(best_alpha_score, best_alpha))
        #             else:
        #                 pass
        #                 # print('No improvement for alpha: {0}, Score: {1}'.format(alph, scores.mean()))
        #         clf.set_params(alpha=best_alpha)
        #         clf.set_params(hidden_layer_sizes=(best_nh_glob, n_classes))
        #
        #         clf.fit(X_train, y_train)
        #         results['NN']['Train_score'] = clf.score(X_train, y_train)
        #         results['NN']['Test_score'] = clf.score(X_test, y_test)
        #
        #         pred = clf.predict(X_test)
        #         confusion_matrix = metrics.confusion_matrix(y_test, pred)
        #         fpr, tpr, thresholds = metrics.roc_curve(y_test, pred, pos_label=1)
        #         AUC = metrics.auc(fpr, tpr)
        #
        #         results['NN']['Confusion Matrix'] = confusion_matrix
        #         results['NN']['AUC'] = AUC
        #
        #         nn_writer.writerow(['$K_{}$'.format(k_counter),
        #                          results['NN']['Train_score'],
        #                          results['NN']['Test_score'],
        #                          confusion_matrix[0][0], confusion_matrix[0][1], confusion_matrix[1][0],
        #                          confusion_matrix[1][1], AUC, best_nh_glob, best_alpha])
        #
        #         print('Train Score: {} %'.format(clf.score(X_train, y_train) * 100))
        #         print('Test Score: {} %\n'.format((results['NN']['Test_score']) * 100))
        #
        #     if clf_name == 'Logistic':
        #
        #         results['Logistic'] = {
        #             # 'Train_score':0,
        #             'Test_score': 0,
        #             'Best_C': 0,
        #             'Best_tol': 0,
        #             'Confusion Matrix': [],
        #             'AUC': 0
        #         }
        #         best_C_score = 0
        #         best_C = 0
        #         C_range = [0.1, 1, 10, 100, 1000]
        #         for tuning_C in C_range:
        #             clf.set_params(C=tuning_C)
        #
        #             best_tol_score = 0
        #             best_tol = 0
        #             tol_range = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
        #             for tuning_tol in tol_range:
        #                 clf.set_params(tol=tuning_tol)
        #                 scores = cross_val_score(clf, X_train, y_train, cv=pt_k_num, scoring='accuracy')
        #                 if scores.mean() >= best_tol_score:
        #                     best_tol_score = scores.mean()
        #                     best_tol = tuning_tol
        #                     # print('Best Score: {0}, Best tol: {1} for C: {2}'.format(best_tol_score, best_tol, tuning_C))
        #                 else:
        #                     pass
        #                     # print('No improvement for Score: {0} for tol: {1} and C: {2}'.format(scores.mean(), tuning_tol,
        #                     #                                                                      tuning_C))
        #
        #             clf.set_params(tol=best_tol)
        #
        #             score = best_tol_score
        #
        #             if score >= best_C_score:
        #                 best_C_score = score
        #                 best_C = tuning_C
        #                 results['Logistic']['Best_tol'] = best_tol
        #                 results['Logistic']['Best_C'] = best_C
        #                 # print('Best Score: {0}, Best C: {1}'.format(best_C_score, best_C))
        #             else:
        #                 pass
        #                 # print('No improvement for C: {0}, Score: {1}'.format(tuning_C, scores.mean()))
        #         clf.set_params(tol=results['Logistic']['Best_tol'])
        #         clf.set_params(C=results['Logistic']['Best_C'])
        #
        #         clf.fit(X_train, y_train)
        #
        #         results['Logistic']['Train_score'] = clf.score(X_train, y_train)
        #         results['Logistic']['Test_score'] = clf.score(X_test, y_test)
        #
        #         pred = clf.predict(X_test)
        #         confusion_matrix = metrics.confusion_matrix(y_test, pred)
        #         fpr, tpr, thresholds = metrics.roc_curve(y_test, pred, pos_label=1)
        #         AUC = metrics.auc(fpr, tpr)
        #
        #         results['Logistic']['Confusion Matrix'] = confusion_matrix
        #         results['Logistic']['AUC'] = AUC
        #
        #         log_writer.writerow(['$K_{}$'.format(k_counter),
        #                          results['Logistic']['Train_score'],
        #                          results['Logistic']['Test_score'],
        #                          confusion_matrix[0][0], confusion_matrix[0][1], confusion_matrix[1][0],
        #                          confusion_matrix[1][1], AUC, results['Logistic']['Best_tol'], results['Logistic']['Best_C']])
        #
        #         print('Train Score: {} %'.format(results['Logistic']['Train_score'] * 100))
        #         print('Test Score: {} %\n'.format(results['Logistic']['Test_score'] * 100))
        #
        #
        #     if clf_name == 'SVM':
        #         results['SVM'] = {
        #             'Train_score': 0,
        #             'Test_score': 0,
        #             'Best_C': 0,
        #             'Best_gamma': 0,
        #             'Confusion Matrix': [],
        #             'SVM': 0
        #         }
        #         best_C_score = 0
        #         best_C = 0
        #         C_range = [0.01, 0.1]  #, 1, 10, 100, 1000]
        #         for tuning_C in C_range:
        #             clf.set_params(C=tuning_C)
        #
        #             best_gamma_score = 0
        #             best_gamma = 0
        #             gamma_range = [10, 1, 1e-1, 1e-2, 1e-3, 1e-4]
        #             for tuning_gamma in gamma_range:
        #                 clf.set_params(gamma=tuning_gamma)
        #
        #                 scores = cross_val_score(clf, X_train, y_train, cv=pt_k_num, scoring='accuracy')
        #
        #                 if scores.mean() >= best_gamma_score:
        #                     best_gamma_score = scores.mean()
        #                     best_gamma = tuning_gamma
        #                     print('Best Score: {0}, Best gamma: {1} for C: {2}'.format(best_gamma_score, best_gamma,
        #                                                                                tuning_C))
        #
        #                 else:
        #                     print('No improvement for gamma: {0}, Score: {1} for C: {2}'.format(tuning_gamma, scores.mean(),
        #                                                                                         tuning_C))
        #             clf.set_params(gamma=best_gamma)
        #
        #             score = best_gamma_score
        #
        #             if score >= best_C_score:
        #                 best_C_score = score
        #                 best_C = tuning_C
        #                 results['SVM']['Best_gamma'] = best_gamma
        #                 results['SVM']['Best_C'] = best_C
        #                 print('Best Score: {0}, Best C: {1}'.format(best_C_score, best_C))
        #             else:
        #                 print('No improvement for C: {0}, Score: {1}'.format(tuning_C, scores.mean()))
        #
        #         clf.set_params(gamma=results['SVM']['Best_gamma'])
        #         clf.set_params(C=results['SVM']['Best_C'])
        #
        #
        #         clf.fit(X_train, y_train)
        #
        #         results['SVM']['Train_score'] = clf.score(X_train, y_train)
        #         results['SVM']['Test_score'] = clf.score(X_test, y_test)
        #
        #         pred = clf.predict(X_test)
        #         confusion_matrix = metrics.confusion_matrix(y_test, pred)
        #         fpr, tpr, thresholds = metrics.roc_curve(y_test, pred, pos_label=1)
        #         AUC = metrics.auc(fpr, tpr)
        #
        #         results['SVM']['Confusion Matrix'] = confusion_matrix
        #         results['SVM']['AUC'] = AUC
        #
        #         svm_writer.writerow(['$K_{}$'.format(k_counter),
        #                          results['SVM']['Train_score'],
        #                          results['SVM']['Test_score'],
        #                          confusion_matrix[0][0], confusion_matrix[0][1], confusion_matrix[1][0],
        #                          confusion_matrix[1][1], AUC, results['SVM']['Best_C'], results['SVM']['Best_gamma'], 'Linear'])
        #
        #         print('Train Score: {} %'.format(results['SVM']['Train_score'] * 100))
        #         print('Test Score: {} %\n'.format(results['SVM']['Test_score'] * 100))
        #
        #
        #     k_counter += 1
        #
        #     avg_accuracy[clf_name]['Run {}'.format(i)] = avg_accuracy[clf_name]['Run {}'.format(i)] + results[clf_name]['Test_score']
        #
        # csvfile.close()


# ### PRINTing the Results
# for clf_name, clf_accuracy in avg_accuracy.items():
#     print('Average accuracy for {}'.format(clf_name))
#     for run, accuracy in clf_accuracy.items():
#         print('{0}: {1}'.format(run, accuracy*10))
#     print(sum(avg_accuracy[clf_name].values())*10/10)
#
# # counter = [int(i[3:]) for i in set(defected_set) if defected_set.count(i) > 15]


