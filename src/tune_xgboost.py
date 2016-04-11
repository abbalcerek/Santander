import numpy as np
from sklearn import cross_validation
from sklearn.decomposition import PCA, KernelPCA
import pandas as pd
from sklearn.decomposition import KernelPCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression

from src.utils import data_path

df = pd.read_csv(data_path('train.csv'))
df_test = pd.read_csv(data_path('test.csv'))

target = df['TARGET']
del df['TARGET']
id = df_test['ID']

from src.transfomations import remove_correlated
_, to_remove = remove_correlated(df, 0.99)

df_test.drop(to_remove, axis=1, inplace=True
             )
variance_threshold = VarianceThreshold(threshold=0.001)
df = variance_threshold.fit_transform(df)

df_test = variance_threshold.fit(df_test)

m2_xgb = XGBClassifier(n_estimators=110, nthread=1, max_depth=4, scale_pos_weight=.8)
m2_xgb.fit(df, target, eval_metric='auc')

param_dist = {
    "n_estimators": [80, 100, 110, 130],
    "max_depth": [3, 4, 5],
    "scale_pos_weight": [0.8, 1, 1.2],
    "learning_rate": [0.1, 0.05, 0.02],
}

randomizedSearch = RandomizedSearchCV(m2_xgb, n_iter=20, param_distributions=param_dist, verbose=2)
randomizedSearch.fit(df, target)

best = randomizedSearch.best_estimator_
print(randomizedSearch.best_params_)
scores = cross_validation.cross_val_score(best, df, target,
                                          cv=5, scoring='roc_auc')
print(scores.mean(), scores)


from src.submission import make_submission
# make_submission('tune_xgboost.csv', id, prediction)
