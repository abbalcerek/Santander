import numpy as np
from sklearn import cross_validation
from sklearn.decomposition import PCA, KernelPCA
import pandas as pd
from sklearn.decomposition import KernelPCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier


from src.utils import data_path

df = pd.read_csv(data_path('train.csv'))
df_test = pd.read_csv(data_path('test.csv'))

target = df['TARGET']
del df['TARGET']
del df['ID']
id = df_test['ID']
del df_test['ID']

from src.transfomations import remove_correlated
_, to_remove = remove_correlated(df, 0.99)

df_test.drop(to_remove, axis=1, inplace=True)

variance_threshold = VarianceThreshold(threshold=0.001)
df = variance_threshold.fit_transform(df)

df_test = variance_threshold.fit(df_test)

gbc = GradientBoostingClassifier()
gbc.fit(df, target)

# best = randomizedSearch.best_estimator_
# print(randomizedSearch.best_params_)
scores = cross_validation.cross_val_score(gbc, df, target, cv=5, scoring='roc_auc')
print(scores.mean(), scores)

# from src.submission import make_submission
# make_submission('gradient_boost1.csv', id, gbc.predict_proba(df_test))
