"""
This script contains the model we gained from the model selection portion, it also contains the needed code to calculate the shap values which we used for our model analysis
"""

import numpy as np
import pandas as pd
import category_encoders as ce
import re
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
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
import shap
class DenseTransformer():

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None, **fit_params):
        return X.todense()

df = pd.read_csv('./df_texas_cleaned.csv')
#print(df.columns)
df_feats = df[['MHLTH_CrudePrev', 'LILATracts_1And10', 'LILATracts_halfAnd10','LILATracts_1And20', 'LILATracts_Vehicle', 'LowIncomeTracts', 'PovertyRate', 'MedianFamilyIncome', 'lasnaphalfshare','TractAsian', 'TractOMultir', 'TractHispanic']]
#print(df_feats.head())
df_feats['LILATracts_1And10'] = df_feats['LILATracts_1And10'].astype("category")
df_feats['LILATracts_halfAnd10'] = df_feats['LILATracts_halfAnd10'].astype("category")
df_feats['LILATracts_1And20'] = df_feats['LILATracts_1And20'].astype("category")
df_feats['LILATracts_Vehicle'] = df_feats['LILATracts_Vehicle'].astype("category")
df_feats['LowIncomeTracts'] = df_feats['LowIncomeTracts'].astype("category")
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
#preprocessor = ColumnTransformer([
#    ('one-hot-encoder', categorical_preprocessor, categorical_columns),
#    ('standard_scaler', numerical_preprocessor, numerical_columns)])
preprocessor = ColumnTransformer([('standard_scaler', numerical_preprocessor, numerical_columns)])
model = make_pipeline(preprocessor)
x = model.fit_transform(x)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)
GS = MLPRegressor(activation='relu', early_stopping=False, learning_rate='adaptive', max_iter=10000, warm_start=False)
GS.fit(x_train, y_train)
print(GS.score(x_train, y_train), 'r2 train')
print(GS.score(x_test, y_test), 'r2 test')
print(np.sqrt(mean_squared_error(y_test, GS.predict(x_test))), "test root mean squared error")
#print(GS.best_params_)
#shap.plots.partial_dependence(
#    "RM", GS.predict, X100, ice=False,
#    model_expected_value=True, feature_expected_value=True
#)

# make a standard partial dependence plot with a single SHAP value overlaid
#fig,ax = shap.partial_dependence_plot(
#    "RM", GS.predict, x, model_expected_value=True,
#    feature_expected_value=True, show=False, ice=False,
#   shap_values=shap_values_ebm[sample_ind:sample_ind+1,:]
#)
features = ['LILATracts_1And10', 'LILATracts_halfAnd10','LILATracts_1And20', 'LILATracts_Vehicle', 'LowIncomeTracts', 'PovertyRate', 'MedianFamilyIncome', 'lasnaphalfshare','TractAsian', 'TractOMultir', 'TractHispanic']
explainer = shap.KernelExplainer(GS.predict,np.array(x_train), feature_names=features)
shap_values = explainer.shap_values(np.array(x_test))
shap.summary_plot(shap_values, np.array(x_train), plot_type="bar", feature_names=features)
shap.summary_plot(shap_values, np.array(x_train), feature_names=features)
#'LILATracts_1And10', 'LILATracts_halfAnd10','LILATracts_1And20', 'LILATracts_Vehicle', 'LowIncomeTracts', 'PovertyRate', 'MedianFamilyIncome', 'lasnaphalfshare','TractAsian', 'TractOMultir', 'TractHispanic'
shap.dependence_plot('LILATracts_1And10', shap_values, x_train, feature_names=features)
shap.dependence_plot('LILATracts_halfAnd10', shap_values, x_train, feature_names=features)
shap.dependence_plot('LILATracts_1And20', shap_values, x_train, feature_names=features)
shap.dependence_plot('LILATracts_Vehicle', shap_values, x_train, feature_names=features)
shap.dependence_plot('LowIncomeTracts', shap_values, x_train, feature_names=features)
shap.dependence_plot('PovertyRate', shap_values, x_train, feature_names=features)
shap.dependence_plot('MedianFamilyIncome', shap_values, x_train, feature_names=features)
shap.dependence_plot('lasnaphalfshare', shap_values, x_train, feature_names=features)
shap.dependence_plot('TractAsian', shap_values, x_train, feature_names=features)
shap.dependence_plot('TractOMultir', shap_values, x_train, feature_names=features)
shap.dependence_plot('TractHispanic', shap_values, x_train, feature_names=features)
shap.initjs()
shap.force_plot(explainer.expected_value, shap_values[0,:]  , x_test[0,:],feature_names=features)
#def shap_plot(j):
#    explainerModel = shap.TreeExplainer(model)
#    shap_values_Model = explainerModel.shap_values(S)
#    p = shap.force_plot(explainerModel.expected_value, shap_values_Model[j], S.iloc[[j]])
#    return(p)
