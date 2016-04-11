import numpy as np
from sklearn import cross_validation
from sklearn.decomposition import PCA, KernelPCA
import pandas as pd
from sklearn.decomposition import KernelPCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
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

df_test.drop(to_remove, axis=1, inplace=True)
variance_threshold = VarianceThreshold(threshold=0.001)

m2_xgb = XGBClassifier(n_estimators=110, nthread=1, max_depth=4)

pipe = Pipeline(steps=[
    ('variance_threshold', variance_threshold),
    ('m2_xgb', m2_xgb)
])

pipe.fit(df, target)
prediction = pipe.predict_proba(df_test)

scores = cross_validation.cross_val_score(pipe, df, target,
                                          cv=5, scoring='roc_auc')
print(scores.mean(), scores)


from src.submission import make_submission
make_submission('simple_xgboost.csv', id, prediction)
