import numpy as np

def transform_with_log1p(dataframe, column_name):
    dataframe[column_name] = np.log1p(dataframe[column_name])