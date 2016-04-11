import numpy as np
from sklearn import cross_validation
from sklearn.decomposition import PCA, KernelPCA
import pandas as pd
from sklearn.decomposition import KernelPCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression


from src.utils import data_path

df = pd.read_csv(data_path('train.csv'))
df_test = pd.read_csv(data_path('test.csv'))

target = df['TARGET']
del df['TARGET']
# del df['ID']
id = df_test['ID']
# del df_test['ID']

pca = PCA(n_components=250)
train_pcaed = pca.fit_transform(df, target)

random_forest = RandomForestClassifier(n_estimators=30, max_depth=5, max_features=20)
random_forest.fit(train_pcaed, target)
forested = random_forest.predict_proba(train_pcaed)
# pipe = Pipeline(steps=[('pca', pca), ('random_forest', random_forest)])

m2_xgb = XGBClassifier(n_estimators=110, nthread=1, max_depth=4)
m2_xgb.fit(train_pcaed, target)
m2_xgbed = m2_xgb.predict_proba(train_pcaed)

logistic_regression = LogisticRegression(penalty='l1')
logistic_regression.fit(train_pcaed, target)
logistic_regressioned = logistic_regression.predict_proba(train_pcaed)

combined = np.concatenate([forested, m2_xgbed, logistic_regressioned], axis=1)


log_reg = LogisticRegression()
log_reg.fit(combined, target)

scores = cross_validation.cross_val_score(log_reg, combined, target,
                                              cv=5, scoring='roc_auc')
print(scores.mean(), scores)

test_pcaed = pca.transform(df_test)
combined_test = np.concatenate([
    random_forest.predict_proba(test_pcaed),
    m2_xgb.predict_proba(test_pcaed),
    logistic_regression.predict_proba(test_pcaed)
  ],
  axis=1
)

from src.submission import make_submission
make_submission('xgboost_rforests.csv', id, log_reg.predict_proba(combined_test))
