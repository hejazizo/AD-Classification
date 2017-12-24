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


from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve, auc
from sklearn import metrics


import matplotlib.pylab as plt
import matplotlib.patches as patches

from scipy import interp
### Loading dataset

dts_name, dts = 'OASIS', None


import csv

with open('data.csv', 'r') as f:
  reader = csv.reader(f)
  data = list(reader)


#### Digits Converstion ####
for row_ind, row in enumerate(data):
    if '' in row:
        row.remove('')

    for cell_ind, cell in enumerate(row):
        try:
            data[row_ind][cell_ind] = float(cell)
        except ValueError:
            pass

np_data = np.zeros((len(data)-1 ,  len(data[0]) - 1), dtype=float)

print(np_data.shape)

flag = True
for ind, row in enumerate(data):
    if not flag:
        np_data[ind-1, :] = row[1:]

    flag = False



print('{1}\nWorking on Database: {0}\n{1}'.format(dts_name, '='*40))

# shuffle and split training and test sets
X, y = np_data[:, :-1], np_data[:, -1]

print('dataset balance: {}'.format(sum(y)/y.shape[0]))
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5, random_state=0)

n_samples = X.shape[0]
n_features = X.shape[1]

n_classes = 4

print('Number of Samples: {0}'.format(n_samples))
print('Number of features: {0}'.format(n_features))
print('Number of classes: {0}'.format(n_classes))

### CLASSIFICATION algorithm
classification_alg = {
                    'SVM': OneVsRestClassifier(svm.SVC(kernel='poly', probability=True, random_state=0)), #kernel='linear', 'poly', 'rbf'
                    # 'NN': MLPClassifier(solver='lbfgs', alpha=1e-3, hidden_layer_sizes=(10, n_classes), random_state=0)
                    }

# avg = 0
# for clf_name, clf in classification_alg.items():
#     print('\t{1}\n\tClassification Algorithm: {0}\n\t{1}'.format(clf_name, '-' * 40))
#
#     ### SPLITTING dataset to test and train data
#     k_num = 6
#     kf = KFold(n_splits=k_num, shuffle=True)
#
#     k_counter = 0
#     for train_index, test_index in kf.split(X,):
#
#         X_train, X_test = X[train_index], X[test_index]
#         y_train, y_test = y[train_index], y[test_index]
#
#         print('Train/Test Size: {0} / {1}'.format(len(X_train), len(X_test)))
#
#         # Parameter Tuning
#         pt_k_num = 3 # number of folds for parameter tuning
#
#
#         if clf_name == 'NN':
#             best_score = 0
#             best_nh = 0
#             for tuning_param in range(2, n_classes + 4):
#                 clf.set_params(hidden_layer_sizes=(tuning_param, n_classes))
#                 scores = cross_val_score(clf, X_train, y_train, cv=pt_k_num, scoring='accuracy')
#
#                 if scores.mean() >= best_score:
#                     best_score = scores.mean()
#                     best_nh = tuning_param
#                     print('Best Score: {0}, Best nh: {1}'.format(best_score, best_nh))
#                 else:
#                     print('No improvement for nh: {0}, Score: {1}'.format(tuning_param, scores.mean()))
#
#             clf.set_params(hidden_layer_sizes=(best_nh, n_classes))
#
#
#
#         clf.fit(X_train, y_train)
#
#         print('Train Score: {} %'.format(clf.score(X_train, y_train)*100))
#         print('Test Score: {} %\n'.format(clf.score(X_test, y_test)*100))
#
#
#         # Confusion Matrix
#         y_predict = clf.predict(X_test)
#         # confusion_matrix = metrics.confusion_matrix(y_test, clf.predict(X_test))
#         # print(confusion_matrix)
#
#
#         avg = avg + clf.score(X_test, y_test)*100
#         #####################################################
#         # plot arrows
#         #####################################################
#         fig1 = plt.figure(figsize=[12, 12])
#         ax1 = fig1.add_subplot(111, aspect='equal')
#         ax1.add_patch(
#             patches.Arrow(0.45, 0.5, -0.25, 0.25, width=0.3, color='green', alpha=0.5)
#         )
#         ax1.add_patch(
#             patches.Arrow(0.5, 0.45, 0.25, -0.25, width=0.3, color='red', alpha=0.5)
#         )
#
#         #####################################################
#         # plot ROC
#         #####################################################
#
#         if n_classes == 2:
#             tprs = []
#             aucs = []
#             mean_fpr = np.linspace(0, 1, 100)
#
#             fpr, tpr, t = roc_curve(y_test, y_predict)
#
#             tprs.append(interp(mean_fpr, fpr, tpr))
#             roc_auc = auc(fpr, tpr)
#             aucs.append(roc_auc)
#             plt.plot(fpr, tpr, lw=2, alpha=0.3, label='ROC fold {0} (AUC = {1})'.format(k_counter, roc_auc))
#
#             plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='black')
#             mean_tpr = np.mean(tprs, axis=0)
#             mean_auc = auc(mean_fpr, mean_tpr)
#             plt.plot(mean_fpr, mean_tpr, color='blue',
#                      label=r'Mean ROC (AUC = %0.2f )' % (mean_auc), lw=2, alpha=1)
#
#             plt.xlabel('False Positive Rate')
#             plt.ylabel('True Positive Rate')
#             plt.title('ROC')
#             plt.legend(loc="lower right")
#             plt.text(0.32, 0.7, 'More accurate area', fontsize=12)
#             plt.text(0.63, 0.4, 'Less accurate area', fontsize=12)
#             plt.savefig('{}.pdf'.format(k_counter))
#             # plt.show()
#
#
#         k_counter += 1
#
#
# print(avg/6)










fig_flag = False
if fig_flag == True:
    #############################################
    images_path = '/media/Ali/8A9E6F039E6EE6E3/Academic/M.Sc (Computer Science)/CMPUT 551/Project/Initial Draft/images'
    df = titanic_df

    font = {
            # 'weight' : 'bold',
            'size'   : 20}

    plt.rc('font', **font)

    # specifies the parameters of our graphs
    fig = plt.figure(figsize=(15,6), dpi=1000)
    alpha=alpha_scatterplot = 0.2
    alpha_bar_chart = 0.55

    # lets us plot many diffrent shaped graphs together
    ax1 = plt.subplot2grid((1, 2),(0,0))
    # plots a bar graph of those who surived vs those who did not.

    print(df.Survived.value_counts())
    df.Survived.value_counts().plot(kind='bar', alpha=alpha_bar_chart)
    # this nicely sets the margins in matplotlib to deal with a recent bug 1.3.1
    ax1.set_xlim(-1, 2)
    # puts a title on our graph
    plt.title("Distribution of Survival, (1 = Survived)")

    plt.subplot2grid((1, 2),(0,1))
    plt.scatter(df.Survived, df.Age, alpha=alpha_scatterplot)
    # sets the y axis lable
    plt.ylabel("Age")
    plt.xticks((0, 1))
    plt.xlim((-0.5, 1.5))
    # formats the grid line style of our graphs
    plt.grid(b=True, which='major', axis='y')
    plt.title("Survival by Age,  (1 = Survived)")

    plt.savefig('{}/survival.pdf'.format(images_path))


    ###################################################

    ax3 = plt.subplot2grid((1, 1),(0,0))
    df.Pclass.value_counts().plot(kind="barh", alpha=alpha_bar_chart)
    ax3.set_ylim(-1, len(df.Pclass.value_counts()))
    plt.title("Class Distribution")

    plt.savefig('{}/class-dist.pdf'.format(images_path))

    ax5 = plt.subplot2grid((1, 1),(0,0))
    df.Embarked.value_counts().plot(kind='bar', alpha=alpha_bar_chart)
    ax5.set_xlim(-1, len(df.Embarked.value_counts()))
    # specifies the parameters of our graphs
    plt.title("Passengers per boarding location")

    plt.savefig('{}/board-loc.pdf'.format(images_path))




    plt.subplot2grid((1, 2),(0,0), colspan=2)
    # plots a kernel density estimate of the subset of the 1st class passangers's age
    df.Age[df.Pclass == 1].plot(kind='kde')
    df.Age[df.Pclass == 2].plot(kind='kde')
    df.Age[df.Pclass == 3].plot(kind='kde')
     # plots an axis lable
    plt.xlabel("Age")
    plt.title("Age Distribution within classes")
    # sets our legend for our graph.
    plt.legend(('1st Class', '2nd Class','3rd Class'),loc='best')

    plt.savefig('{}/age-class.pdf'.format(images_path))