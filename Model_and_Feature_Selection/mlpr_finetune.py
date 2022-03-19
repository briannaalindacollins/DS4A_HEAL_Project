"""
This script uses GridSearchCV to find the optimal hyperparameters with the r2 as a metric for the MLPRegressor
"""

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
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
from matplotlib import pyplot
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

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
numerical_columns_selector = selector(dtype_exclude=object)
categorical_columns_selector = selector(dtype_include=object)
#
numerical_columns = numerical_columns_selector(x)
categorical_columns = categorical_columns_selector(x)
#
categorical_preprocessor = OneHotEncoder(handle_unknown="ignore")
numerical_preprocessor = StandardScaler()
preprocessor = ColumnTransformer([
    ('one-hot-encoder', categorical_preprocessor, categorical_columns),
    ('standard_scaler', numerical_preprocessor, numerical_columns)])
model = make_pipeline(preprocessor)
x = model.fit_transform(x)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)
mlp = MLPRegressor()
param_grid = {'hidden_layer_sizes': [i for i in range(1,200)],
              'solver': ["lbfgs", "sgd", "adam"],
              'activation': ['identity', 'logistic', 'tanh', 'relu'],
              'learning_rate': ['constant', 'adaptive'],
              'learning_rate_init': [0.0001, 0.1],
              'alpha': [0.00005,0.0005],
              'max_iter': [10000],
              'early_stopping': [False],
              'warm_start': [False]}
GS = GridSearchCV(mlp, param_grid=param_grid, scoring='r2', verbose=True)
GS.fit(x_train, y_train)
print(GS.score(x_train, y_train), 'r2 train')
print(GS.score(x_test, y_test), 'r2 test')
print(np.sqrt(mean_squared_error(y_test, GS.predict(x_test))), "test root mean squared error")
print(GS.best_params_)
