from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

from src.submission import make_submission
from src.utils import data_path
from sklearn import metrics, cross_validation

df = pd.read_csv(data_path('train.csv'))
df_test = pd.read_csv(data_path('test.csv'))

target = df['TARGET']
del df['TARGET']


clf = RandomForestClassifier(n_estimators=30, max_depth=5, max_features=20)
clf.fit(df, target)

print(len(df_test))
print(len(clf.predict_proba(df_test)))

prediction = [pred for _, pred in clf.predict_proba(df_test)]

make_submission('rf30_5_20.csv', df_test['ID'], prediction)