

def remove_correlated(df, th):
    to_remove = set()
    corr = df.corr()
    col_pairs = ((row, col) for row in corr.index
                            for col in corr.columns if row < col)
    for row, col in col_pairs:
        if (row in to_remove) or (col in to_remove):
            pass
        else:
            if corr.at[row, col] > th:
                to_remove.add(col)
    df.drop(to_remove, axis=1, inplace=True)
    return df, to_remove