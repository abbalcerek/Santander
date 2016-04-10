from sklearn.linear_model import LogisticRegression
import pandas as pd

from src.submission import make_submission
from src.utils import data_path
from sklearn import metrics, cross_validation

df = pd.read_csv(data_path('train.csv'))
df_test = pd.read_csv(data_path('test.csv'))

clf = LogisticRegression()

target = df['TARGET']
del df['TARGET']
scores = cross_validation.cross_val_score(clf, df, target,
                                          cv=5, scoring='log_loss')
print(scores.all())
print(scores)
print(scores.mean())
clf.fit(df, target)

print(len(df_test))
print(len(clf.predict_proba(df_test)))

prediction = [pred for _, pred in clf.predict_proba(df_test)]

# make_submission('baseline.csv', df_test['ID'], prediction)