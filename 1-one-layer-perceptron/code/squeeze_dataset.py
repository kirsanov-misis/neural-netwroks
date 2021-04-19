import pandas as pd

dataframe = pd.read_csv('../avila.txt', sep=',', header=None)
allowed_targets = ['A', 'E', 'F', 'H', 'I', 'X']
dataframe = dataframe.drop(
    dataframe[~dataframe.iloc[:, -1].isin(allowed_targets)].index)
dataframe.to_csv('../avila_squeezed.txt', header=None, index=None)
