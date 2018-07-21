import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt  # Matlab-style plotting
import seaborn as sns
color = sns.color_palette()
sns.set_style('darkgrid')
import warnings
def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn #ignore annoying warning (from sklearn and seaborn)
from scipy import stats
from scipy.stats import norm, skew #for some statistics
# Models
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
import df_helper as dh




pd.set_option('display.float_format', lambda x: '{:.3f}'.format(x)) #Limiting floats output to 3 decimal points

train = pd.read_csv('.\\data\\train.csv')
test = pd.read_csv('.\\data\\test.csv')

#Save the 'Id' column
train_ID = train['Id']
test_ID = test['Id']

#Now drop the  'Id' colum since it's unnecessary for  the prediction process.
train.drop("Id", axis = 1, inplace = True)
test.drop("Id", axis = 1, inplace = True)

# Removendo outliers
dh.remove_outliers(train)


# Normalizando SalePrice
np.log1p(train["SalePrice"])

y_train = train.SalePrice.values

ntrain = train.shape[0]
ntest = test.shape[0]

all_data = dh.concat_dataframes(train, test)

all_data.drop(['SalePrice'], axis=1, inplace=True)

# Tratando missing values
all_data = dh.treat_missing_data(all_data)

# Adequando valores numéricos para string pois são praticamente categóricos
all_data = dh.transform_numeric_to_categorical(all_data)

# Label encoding
cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 
        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 
        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 
        'YrSold', 'MoSold')
# process columns, apply LabelEncoder to categorical features
all_data = dh.label_encoding(all_data, cols)

# Adicionando feature
all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']

#Encontrando valores enviesados 
numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index


# Check the skew of all numerical features
skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
skewness_upper = pd.DataFrame({'Skew' :skewed_feats[skewed_feats > 0.75]})
skewness_under = pd.DataFrame({'Skew' :skewed_feats[skewed_feats < -0.75]})
skewness = pd.concat((skewness_upper, skewness_under))

# Normalizando valores enviesados
from scipy.special import boxcox1p
skewed_features = skewness.index
lam = 0.15
for feat in skewed_features:
    all_data[feat] = boxcox1p(all_data[feat], lam)

# Transformando variáveis cateegóricas em booleanas
all_data = pd.get_dummies(all_data)

# Repartindo datasets iniciais
train = all_data[:ntrain]
test = all_data[ntrain:]

#Validation function
n_folds = 5

def rmsle_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train.values)
    rmse= np.sqrt(-cross_val_score(model, train.values, y_train, scoring="neg_mean_squared_error", cv = kf))
    return(rmse)

def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))

ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))
score = rmsle_cv(ENet)
print("ElasticNet score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

ENet.fit(train, y_train)
ENet_prediction = ENet.predict(train.values)
print(rmsle(y_train, ENet_prediction))

ENet_test_prediction = np.expm1(ENet.predict(test.values))

sub = pd.DataFrame()
sub['Id'] = test_ID
sub['SalePrice'] = ENet_test_prediction
sub.to_csv('data\\submission.csv',index=False)