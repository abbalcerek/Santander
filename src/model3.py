import numpy as np
from sklearn import cross_validation
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold
from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing import StandardScaler


from src.utils import data_path

df = pd.read_csv(data_path('train.csv'))
df_test = pd.read_csv(data_path('test.csv'))

target = df['TARGET']
del df['TARGET']

from src.transfomations import remove_correlated

_, to_remove = remove_correlated(df, 0.99)

df_test.drop(to_remove, axis=1, inplace=True)

variance_threshold = VarianceThreshold(threshold=0.001)

scaler = StandardScaler()
pca = PCA(n_components=100)
random_forest = RandomForestClassifier(n_estimators=30, max_depth=5, max_features=20)

pipe = Pipeline(steps=[
    ('variance_threshold', variance_threshold),
    ('scaler', scaler),
    ('pca', pca),
    ('random_forest', random_forest)
])

# pca.fit(df, target)
pipe.fit(df, target)
# scores = cross_validation.cross_val_score(pipe, df, target,
#                                               cv=5, scoring='log_loss')


# n_components =[20, 40, 64, 100, 150, 200]
n_components = [50, 70, 100, 120, 140, 160]
v_treshold = [0.1, 0.01, 0.001]
depth = [5, 7, 10]
estimator = GridSearchCV(pipe,
                         dict(
                             pca__n_components=n_components,
                             variance_threshold__threshold=v_treshold,
                             random_forest__max_depth=depth
                         ),
                         scoring='log_loss')
estimator.fit(df, target)
#
print(estimator.best_params_)
scores = cross_validation.cross_val_score(estimator.best_estimator_, df, target,
                                              cv=5, scoring='log_loss')
print(scores.mean(), scores)
#
# train_transformed = pca.fit_transform(df)
# print(train_transformed.shape)

# pipe.fit(df, target)

# scores = cross_validation.cross_val_score(pipe, df, target,
#                                               cv=5, scoring='roc_auc')
# print(scores.mean(), scores)

# from src.submission import make_submission
# make_submission('clean_data.csv', df_test['ID'], pipe.predict_proba(df_test))
