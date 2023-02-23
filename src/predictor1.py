import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from sklearn import metrics
from sklearn.metrics import r2_score, mean_squared_error, make_scorer

from sklearn.metrics import confusion_matrix,precision_score,recall_score,classification_report, accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler
import warnings

warnings.filterwarnings('ignore')

import read_data
from tools import smape

# following:
# https://www.kaggle.com/code/bibanh/baseline-82-train-inference-randomforest/comments


data = read_data.data()

x, y = data.get_x_y_data()


x['protein_mean']= x[data.protein_names].mean(axis=1)
x['protein_min']= x[data.protein_names].min(axis=1)
x['protein_max']= x[data.protein_names].max(axis=1)
x['peptide_mean']= x[data.peptide_names].mean(axis=1)
x['peptide_min']= x[data.peptide_names].min(axis=1)
x['peptide_max']= x[data.peptide_names].max(axis=1)


x = x[['protein_mean', 'protein_min', 'protein_max', 'peptide_mean', 'peptide_max', 'peptide_min']]
y = y['updrs_1']



model = {}
mms = MinMaxScaler()
n_estimators = [5]  # number of trees in the random forest
max_features = ['sqrt'] #['auto', 'sqrt']  # number of features in consideration at every split
max_depth = [int(x) for x in np.linspace(10, 50, num=12)]  # maximum number of levels allowed in each decision tree
min_samples_split = [2, 6, 10]  # minimum sample number to split a node
min_samples_leaf = [3, 4]  # minimum sample number that can be stored in a leaf node
bootstrap = [True, False]  # method used to sample data points

rfc = RandomForestRegressor()
forest_params = [{'n_estimators': n_estimators,

                  'max_features': max_features,

                  'max_depth': max_depth,

                  'min_samples_split': min_samples_split,

                  'min_samples_leaf': min_samples_leaf,

                  'bootstrap': bootstrap}]

cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=42)
clf = RandomizedSearchCV(rfc, forest_params, cv=cv, scoring=make_scorer(smape), verbose=1, error_score='raise')
clf.fit(x, y)

print(clf.best_params_)
print(clf.best_score_)
