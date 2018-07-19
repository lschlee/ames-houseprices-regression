import pandas as pd
from sklearn.preprocessing import LabelEncoder

def remove_outliers(data_frame):
    data_frame.drop(data_frame[(data_frame['GrLivArea']>4000) & (data_frame['SalePrice']<300000)].index)

def concat_dataframes(df1, df2):
    return pd.concat((df1, df2)).reset_index(drop=True)

def fill_na_with_none(df, columns):
    for col in columns:
        df[col] = df[col].fillna('None')
    return df

def fill_group_column_with_media(df, group_column, column_to_fill):
    return df.groupby(group_column)[column_to_fill].transform(
    lambda x: x.fillna(x.median()))

def treat_missing_data(all_data):
    categorical_columns = ["PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu", "MasVnrType", "GarageType", "GarageFinish", "GarageQual", "GarageCond", "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2", "MSSubClass"]
    all_data = fill_na_with_none(all_data, categorical_columns)

    all_data["LotFrontage"] = all_data.groupby("Neighborhood")["LotFrontage"].transform(
    lambda x: x.fillna(x.median()))

    for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
       all_data[col] = all_data[col].fillna(0)

    for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
        all_data[col] = all_data[col].fillna(0)

    all_data["MasVnrArea"] = all_data["MasVnrArea"].fillna(0)
    all_data['MSZoning'] = all_data['MSZoning'].fillna(all_data['MSZoning'].mode()[0])
    all_data = all_data.drop(['Utilities'], axis=1)
    all_data["Functional"] = all_data["Functional"].fillna("Typ")
    all_data['Electrical'] = all_data['Electrical'].fillna(all_data['Electrical'].mode()[0])
    all_data['KitchenQual'] = all_data['KitchenQual'].fillna(all_data['KitchenQual'].mode()[0])
    all_data['Exterior1st'] = all_data['Exterior1st'].fillna(all_data['Exterior1st'].mode()[0])
    all_data['Exterior2nd'] = all_data['Exterior2nd'].fillna(all_data['Exterior2nd'].mode()[0])
    all_data['SaleType'] = all_data['SaleType'].fillna(all_data['SaleType'].mode()[0])

    return all_data

def transform_numeric_to_categorical(all_data):
    all_data['MSSubClass'] = all_data['MSSubClass'].apply(str)
    all_data['OverallCond'] = all_data['OverallCond'].astype(str)
    all_data['YrSold'] = all_data['YrSold'].astype(str)
    all_data['MoSold'] = all_data['MoSold'].astype(str)

    return all_data

def label_encoding(all_data, cols):
    for c in cols:
       lbl = LabelEncoder() 
       lbl.fit(list(all_data[c].values)) 
       all_data[c] = lbl.transform(list(all_data[c].values))
    return all_data

