import numpy as np

# TODO: comment funcitons
# TODO: maybe there is a way to use k-nearest neighbours and their distances to remove outliers (specles)


def remove_percentile_above(df, features, percentile=95):
    for col_name in features:
        df = df[df[col_name] < np.percentile(df[col_name], percentile)]
    return df

def remove_percentile_below(df, features, percentile=95):
    for col_name in features:
        df = df[df[col_name] > np.percentile(df[col_name], percentile)]
    return df

def remove_percentile(df, features, percentile=95, above=True, below=True):
    if above:
        df = remove_percentile_above(df, features, percentile)
    if below:
        df = remove_percentile_below(df, features, 100-percentile)
    return df