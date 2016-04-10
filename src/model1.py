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


def run_classifier(est_num, depth, n_f='auto'):
    clf = RandomForestClassifier(n_estimators=est_num, max_depth=depth, max_features=n_f)
    scores = cross_validation.cross_val_score(clf, df, target,
                                              cv=5, scoring='log_loss')
    print('est_num={}, depth={}, n_f={}'.format(str(est_num), str(depth), str(n_f)), scores.mean(), scores)
    (scores.mean(), scores, est_num, depth, n_f)


n_fs = [20, 40]
result = []


print(['auto'] + n_fs)

for depth in [5, 7, 8, 10, 15]:
    for n_f in ['auto'] + n_fs:
        for num in [10, 20, 30, 40, 50, 80, 100]:
            result.append(run_classifier(num, depth, n_f))


print("=======================================")
result.sort()
for r in result:
    mean, scores, est_num, depth, n_f = r
    print('est_num={}, depth={}, n_f={}'.format(str(est_num), str(depth), str(n_f)), mean, scores)


# clf.fit(df, target)

# print(len(df_test))
# print(len(clf.predict_proba(df_test)))
#
# prediction = [pred for _, pred in clf.predict_proba(df_test)]

# make_submission('baseline.csv', df_test['ID'], prediction)