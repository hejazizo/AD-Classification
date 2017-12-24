import pandas as pd
from sklearn.cluster import KMeans


import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor

#### LOADING DATA
alzheimer_df = pd.read_csv("data.csv")

#### PREPROCESSING ####
############################################

# we map each title
y = alzheimer_df.output
X = alzheimer_df.drop(['output', 'Unnamed: 140'], axis=1)

clf = LocalOutlierFactor()
y_pred = clf.fit_predict(X.drop(['Measure:volume'], axis=1))

print([int(X['Measure:volume'].iloc[sub][3:]) for sub in [ind for ind, val in enumerate(y_pred) if val == -1]])

