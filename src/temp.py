import pandas as pd
from src.utils import data_path
from collections import Counter

df = pd.read_csv(data_path('train.csv'))


# columns = list(df.columns)
#
# from itertools import groupby
# import re
#
# grouped = groupby(columns, key=lambda name: re.split('_|[0-9]', name)[0])
#
# for key, group in grouped:
#     print(key, list(group))

def desc(name):
    var = df[[name]]
    varV = df[name]

    print(set(varV))
    print(var.describe())
    print(len(var))
    print(len(var[varV == -999999]))
    counter = Counter(varV)
    print(counter.most_common())


# desc('var3')
# desc('var36')

# print(df.dtypes)
#
# print(len(df.columns))

# desc('imp_ent_var16_ult1')

def remove_corr(df):
    to_remove = []
    corr = df.corr()




d = {'one': pd.Series([1., 2., 3., 3.], index=['a', 'b', 'c', 'd']),
     'two': pd.Series([1., 2., 3., 4.], index=['a', 'b', 'c', 'd']),
     'tree': pd.Series([1., 2., 3., 4.], index=['a', 'b', 'c', 'd'])
     }


df1 = pd.DataFrame(d)

# print(d)
# print(df1.corr())
# df1 = df1.corr()
# print(df1)

# print(df1.at['a', 'one'])


# def remove_correlated(df, th):
#     to_remove = set()
#     corr = df.corr()
#     col_pairs = ((row, col) for row in corr.index
#                             for col in corr.columns if row < col)
#     for row, col in col_pairs:
#         if (row in to_remove) or (col in to_remove):
#             pass
#         else:
#             if corr.at[row, col] > th:
#                 to_remove.add(col)
#     df.drop(to_remove, axis=1, inplace=True)
#     return df, to_remove


# pairs = ((row, col) for row in df1.index for col in df1.columns if row < col)

# print(pairsf(df1, 0.9))


# transformed, removed = remove_correlated(df, 0.9)
# print(removed)
# print(len(removed))

# print(list(pairs))
# for index in df1.itertuples():
#
#     print(index)
#     # print(row)
#     print()


print df['TARGET'].mean()
