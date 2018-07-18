import pandas as pd

def remove_outliers(data_frame):
    data_frame.drop(data_frame[(data_frame['GrLivArea']>4000) & (data_frame['SalePrice']<300000)].index)

def concat_dataframes(df1, df2):
    return pd.concat((df1, df2)).reset_index(drop=True)

def fill_na_of_df_with_none_for_columns(df, columns):
    for col in columns:
        df[col] = df[col].fillna('None')
    return df