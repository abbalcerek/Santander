import numpy as np
from sklearn import cross_validation
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold
from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing import StandardScaler
from src.transfomations import remove_correlated

from unbalanced_dataset import *

from src.utils import data_path

if __name__ == '__main__':
    df = pd.read_csv(data_path('train.csv'))
    df_test = pd.read_csv(data_path('test.csv'))

    target = df['TARGET']
    del df['TARGET']


    _, to_remove = remove_correlated(df, 0.99)

    df_test.drop(to_remove, axis=1, inplace=True)

    variance_threshold = VarianceThreshold(threshold=0.001)

    scaler = StandardScaler()
    pca = PCA(n_components=120)
    random_forest = RandomForestClassifier(n_estimators=30, max_depth=7, max_features=20, class_weight='auto')

    ratio = float(sum(target)) / float(len(target) - sum(target))
    print(ratio)

    smote = OverSampler(ratio=ratio * 2, verbose=True)
    print(type(target))
    # df, target = smote.fit_transform(df.as_matrix(), target.as_matrix())

    print(df.shape, target.shape)

    pipe = Pipeline(steps=[
        ('variance_threshold', variance_threshold),
        ('scaler', scaler),
        ('pca', pca),
        ('random_forest', random_forest)
    ])

    # pca.fit(df, target)
    # scores = cross_validation.cross_val_score(pipe, df, target,
    #                                               cv=5, scoring='log_loss')


    # n_components =[20, 40, 64, 100, 150, 200]
    # n_components = [50, 100, 140, 160]
    # v_treshold = [0.1, 0.01, 0.001]
    # depth = [5, 7, 10]
    # estimator = GridSearchCV(pipe,
    #                          dict(
    #                              pca__n_components=n_components,
    #                              variance_threshold__threshold=v_treshold,
    #                              random_forest__max_depth=depth
    #                          ),
    #                          scoring='roc_auc')
    # estimator.fit(df, target)
    # #
    # print(estimator.best_params_)
    # scores = cross_validation.cross_val_score(estimator.best_estimator_, df, target,
    #                                               cv=5, scoring='roc_auc')
    # print(scores.mean(), scores)
    #
    # train_transformed = pca.fit_transform(df)
    # print(train_transformed.shape)

    # pipe.fit(df, target)

    scores = cross_validation.cross_val_score(pipe, df, target, n_jobs=4, verbose=2,
                                              cv=5, scoring='roc_auc')
    print(scores.mean(), scores)

    # from src.submission import make_submission
    # make_submission('clean_data.csv', df_test['ID'], pipe.predict_proba(df_test))
