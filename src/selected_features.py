import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import cross_validation
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
from src.utils import data_path


df = pd.read_csv(data_path('train.csv'))
df_test = pd.read_csv(data_path('test.csv'))

colsToRemove = []
for col in df.columns:
    if df[col].std() == 0:
        colsToRemove.append(col)

df.drop(colsToRemove, axis=1, inplace=True)
df_test.drop(colsToRemove, axis=1, inplace=True)

# remove duplicate columns
colsToRemove = []
columns = df.columns
for i in range(len(columns)-1):
    v = df[columns[i]].values
    for j in range(i+1,len(columns)):
        if np.array_equal(v, df[columns[j]].values):
            colsToRemove.append(columns[j])

df.drop(colsToRemove, axis=1, inplace=True)
df_test.drop(colsToRemove, axis=1, inplace=True)

labels = df['TARGET']
df_test_id = df_test['ID']
df = df.drop(['ID', 'TARGET'], axis=1)
df_test = df_test.drop(['ID'], axis=1)


# df = pd.read_csv(data_path('train.csv'))
# df_test = pd.read_csv(data_path('test.csv'))
#
# labels = df['TARGET']
# df_test_id = df_test['ID']
#
# colls = ['saldo_var30', 'var15', 'saldo_var5', 'ind_var30', 'var38', 'saldo_medio_var5_ult3', 'num_meses_var5_ult3', 'saldo_medio_var5_hace3', 'var36', 'num_meses_var39_vig_ult3', 'num_var30', 'num_var5', 'num_var4', 'num_var45_hace2']
# print(sorted(colls))
#
# df = df[colls]
# df_test = df_test[colls]

# poly = PolynomialFeatures(2)
# df = poly.fit_transform(df)
# df_test = poly.transform(df_test)

clf = GradientBoostingClassifier(verbose=3)


# clf = RandomForestClassifier()
clf.fit(df, labels)

scores = cross_validation.cross_val_score(clf, df, labels,
                                          cv=5, scoring='roc_auc')
print(scores.mean(), scores)

from src.submission import make_submission
make_submission('gradient_boosting.csv', df_test_id, clf.predict_proba(df_test))
