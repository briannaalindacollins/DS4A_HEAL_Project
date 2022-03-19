"""
This script uses RFE with the gradient boosting regressor for feature selection
"""

import seaborn as sns
import pandas as pd
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from tabulate import tabulate
import numpy as np

df = pd.read_csv('./df_texas_cleaned.csv')
#print(df.columns)
df_r30 = df[['LILATracts_1And10', 'LILATracts_halfAnd10', 'LILATracts_1And20', 'LILATracts_Vehicle', 'LowIncomeTracts', 'PovertyRate', 'MedianFamilyIncome', 'LALOWI05_10', 'lalowihalf', 'lalowihalfshare', 'lawhitehalfshare', 'laomultirhalfshare', 'lahisphalfshare', 'lahunvhalf', 'lahunvhalfshare', 'lasnaphalf', 'lasnaphalfshare', 'TractAsian', 'TractOMultir', 'TractHispanic', 'TractHUNV'   ,'TractSNAP', 'MHLTH_CrudePrev']]
x = df_r30[['LILATracts_1And10', 'LILATracts_halfAnd10', 'LILATracts_1And20', 'LILATracts_Vehicle', 'LowIncomeTracts', 'PovertyRate', 'MedianFamilyIncome', 'LALOWI05_10', 'lalowihalf', 'lalowihalfshare', 'lawhitehalfshare', 'laomultirhalfshare', 'lahisphalfshare', 'lahunvhalf', 'lahunvhalfshare', 'lasnaphalf', 'lasnaphalfshare', 'TractAsian', 'TractOMultir', 'TractHispanic', 'TractHUNV'   ,'TractSNAP']]
y = df_r30['MHLTH_CrudePrev']
##RFE Feature selection
# explore the algorithm wrapped by RFE
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from matplotlib import pyplot
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor
#def get_models():
#	models = dict()
#	# lr
#	rfe = RFE(estimator=RandomForestRegressor(), n_features_to_select=12)
#	model = DecisionTreeRegressor()
#	models['RandomForest'] = Pipeline(steps=[('s',rfe),('m',model)])
#	# perceptron
#	rfe = RFE(estimator=ExtraTreesRegressor(), n_features_to_select=12)
#	model = DecisionTreeRegressor()
#	models['ExtraTrees'] = Pipeline(steps=[('s',rfe),('m',model)])
#	# cart
#	rfe = RFE(estimator=GradientBoostingRegressor(), n_features_to_select=12)
#	model = DecisionTreeRegressor()
#	models['GradientBoosting'] = Pipeline(steps=[('s',rfe),('m',model)])
#	# rf
#	rfe = RFE(estimator=AdaBoostRegressor(), n_features_to_select=12)
#	model = DecisionTreeRegressor()
#	models['AdaBoost'] = Pipeline(steps=[('s',rfe),('m',model)])
#	# gbm
#	rfe = RFE(estimator=DecisionTreeRegressor(), n_features_to_select=12)
#	model = DecisionTreeRegressor()
#	models['DecisionTree'] = Pipeline(steps=[('s',rfe),('m',model)])
#	return models
# 
## evaluate a give model using cross-validation
#def evaluate_model(model, X, y):
#	scores = cross_val_score(model, X, y, scoring='r2', cv=5, n_jobs=-1)
#	return scores
# 
## define dataset
## get the models to evaluate
#models = get_models()
## evaluate the models and store results
#results, names = list(), list()
#for name, model in models.items():
#	scores = evaluate_model(model, x, y)
#	results.append(scores)
#	names.append(name)
#	print('>%s %.3f (%.3f)' % (name, mean(scores), std(scores)))
## plot model performance for comparison
#pyplot.boxplot(results, labels=names, showmeans=True)
#pyplot.show()
rfe = RFE(estimator=GradientBoostingRegressor(), n_features_to_select=7)
# fit RFE
rfe.fit(x, y)
# summarize all features
for i in range(x.shape[1]):
	print('Column: %d, %s, Rank: %.3f' % (i, rfe.support_[i], rfe.ranking_[i]))

