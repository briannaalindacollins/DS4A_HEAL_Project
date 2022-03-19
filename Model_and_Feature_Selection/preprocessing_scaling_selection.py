"""
This script brute forces through different preprocessing and model combinations to find the best pairing
"""

import seaborn as sns
import pandas as pd
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor, ExtraTreesRegressor, RandomForestRegressor
from tpot import TPOTRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from mlxtend.regressor import StackingCVRegressor
from sklearn.linear_model import Ridge, Lasso
import numpy as np
import pandas as pd
import category_encoders as ce
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.style.use('ggplot')
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, Normalizer, MinMaxScaler, MaxAbsScaler, RobustScaler, QuantileTransformer, PowerTransformer, PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor, ExtraTreesRegressor, RandomForestRegressor, HistGradientBoostingRegressor, VotingRegressor
from tpot import TPOTRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from mlxtend.regressor import StackingCVRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.compose import make_column_selector as selector
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_validate
from sklearn.kernel_ridge import KernelRidge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.neighbors import RadiusNeighborsRegressor, KNeighborsRegressor
from sklearn.cross_decomposition import PLSCanonical, PLSRegression
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn import tree
from sklearn.preprocessing import OneHotEncoder
from sklearn.neural_network import MLPRegressor, BernoulliRBM
from sklearn.linear_model import RANSACRegressor, LinearRegression, HuberRegressor, BayesianRidge, Lasso, ElasticNet, ARDRegression, TheilSenRegressor, SGDRegressor
class DenseTransformer():

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None, **fit_params):
        return X.todense()

df = pd.read_csv('./df_texas_cleaned.csv')
#print(df.columns)
df_feats = df[['MHLTH_CrudePrev', 'LILATracts_1And10', 'LILATracts_halfAnd10','LILATracts_1And20', 'LILATracts_Vehicle', 'LowIncomeTracts', 'PovertyRate', 'MedianFamilyIncome', 'lasnaphalfshare','TractAsian', 'TractOMultir', 'TractHispanic']]
#print(df_feats.head())
df_feats['LILATracts_1And10'] = df_feats['LILATracts_1And10'].astype(object)
df_feats['LILATracts_halfAnd10'] = df_feats['LILATracts_halfAnd10'].astype(object)
df_feats['LILATracts_1And20'] = df_feats['LILATracts_1And20'].astype(object)
df_feats['LILATracts_Vehicle'] = df_feats['LILATracts_Vehicle'].astype(object)
df_feats['LowIncomeTracts'] = df_feats['LowIncomeTracts'].astype(object)
x = df_feats[['LILATracts_1And10', 'LILATracts_halfAnd10','LILATracts_1And20', 'LILATracts_Vehicle', 'LowIncomeTracts', 'PovertyRate', 'MedianFamilyIncome', 'lasnaphalfshare','TractAsian', 'TractOMultir', 'TractHispanic']]
y = df_feats['MHLTH_CrudePrev']
###mix model
#numerical_columns_selector = selector(dtype_exclude=object)
#categorical_columns_selector = selector(dtype_include=object)
#
#numerical_columns = numerical_columns_selector(x)
#categorical_columns = categorical_columns_selector(x)
#
#categorical_preprocessor = OneHotEncoder(handle_unknown="ignore")
#numerical_preprocessor = MinMaxScaler()
#preprocessor = ColumnTransformer([
#    ('one-hot-encoder', categorical_preprocessor, categorical_columns),
#    ('standard_scaler', numerical_preprocessor, numerical_columns)])
#model = make_pipeline(preprocessor)
#x = model.fit_transform(x)
#x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)
list = [GradientBoostingRegressor(), KernelRidge(), SVR(), GaussianProcessRegressor(), tree.DecisionTreeRegressor(), MLPRegressor(), RandomForestRegressor(), ExtraTreesRegressor(), HistGradientBoostingRegressor() ]
scaling_list = [StandardScaler(), Normalizer(), MinMaxScaler(), MaxAbsScaler(), RobustScaler(), QuantileTransformer(), PowerTransformer(), PolynomialFeatures()]
df_rmse = pd.DataFrame()
df_r2_train = pd.DataFrame()
df_r2_test = pd.DataFrame()
for i in range(len(scaling_list)):
    ###mix model
    numerical_columns_selector = selector(dtype_exclude=object)
    categorical_columns_selector = selector(dtype_include=object)
    numerical_columns = numerical_columns_selector(x)
    categorical_columns = categorical_columns_selector(x)
    categorical_preprocessor = OneHotEncoder(handle_unknown="ignore")
    numerical_preprocessor = scaling_list[i]
    preprocessor = ColumnTransformer([
    ('one-hot-encoder', categorical_preprocessor, categorical_columns),
    ('standard_scaler', numerical_preprocessor, numerical_columns)])
    pre = make_pipeline(preprocessor)
    x_t = pre.fit_transform(x)
    x_train, x_test, y_train, y_test = train_test_split(x_t, y, test_size = 0.2)
    for j in range(len(list)):
        model = list[j]
        name = str(list[j])
        scale = str(scaling_list[i])
        _ = model.fit(x_train, y_train)
        r2_train = model.score(x_train, y_train)
        r2_test = model.score(x_test, y_test)
        rmse = np.sqrt(mean_squared_error(y_test, model.predict(x_test)))
        df_rmse.loc[scale, name] = rmse
        df_r2_test.loc[scale, name] = r2_test
        df_r2_train.loc[scale, name] = r2_train
        print('------', name, scale, '--------')
        print(model.score(x_train, y_train), 'r2 train')
        print(model.score(x_test, y_test), 'r2 test')
        print(np.sqrt(mean_squared_error(y_test, model.predict(x_test))), "test root mean squared error")
        print('------', name, scale, '--------')
        del model
    del x_t
df_rmse.to_csv('./rmse.csv')
df_r2_test.to_csv('./r2_test.csv')
df_r2_train.to_csv('./r2_train.csv')


