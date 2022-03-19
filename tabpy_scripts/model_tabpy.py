"""
This is the model written in a way that tabpy can read. Used in the backend of the application
"""
SCRIPT_REAL("
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.compose import make_column_selector as selector
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import OneHotEncoder
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import numpy.ma as ma

sc= StandardScaler()
LILATracts_1And10 = np.array(_arg1).reshape(-1,1)
LILATracts_halfAnd10 = np.array(_arg2).reshape(-1,1)
LILATracts_1And20 = np.array(_arg3).reshape(-1,1)
LILATracts_Vehicle = np.array(_arg4).reshape(-1,1)
LowIncomeTracts = np.array(_arg5).reshape(-1,1)
PovertyRate=sc.fit_transform(np.array(_arg6).reshape(-1,1))
MedianFamilyIncome=sc.fit_transform(np.array(_arg7).reshape(-1,1))
lasnaphalfshare=sc.fit_transform(np.array(_arg8).reshape(-1,1))
TractAsian=sc.fit_transform(np.array(_arg9).reshape(-1,1))
TractOMultir=sc.fit_transform(np.array(_arg10).reshape(-1,1))
TractHispanic=sc.fit_transform(np.array(_arg11).reshape(-1,1))

list = [LILATracts_1And10, LILATracts_halfAnd10, LILATracts_1And20, LILATracts_Vehicle, LowIncomeTracts, PovertyRate, MedianFamilyIncome, lasnaphalfshare, TractAsian, TractOMultir, TractHispanic]
x_all = np.array(list).reshape(-1,3)
y = np.array(_arg12)

x = np.where(np.isnan(x_all), ma.array(x_all, mask=np.isnan(x_all)).mean(axis=0), x_all)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)
GS = MLPRegressor(activation='relu', early_stopping=False, learning_rate='adaptive', max_iter=10000, warm_start=False)
GS.fit(x_train, y_train)
predicted_category = GS.predict(x)
return predicted_category.tolist()"

,
ATTR([LILATracts 1And10]),
ATTR([LILATracts halfAnd10]),
ATTR([LILATracts 1And20]),
ATTR([LILATracts Vehicle]),
ATTR([Low Income Tracts]), ATTR([Poverty Rate]), ATTR([Median Family Income]), ATTR([Lasnaphalfshare]), ATTR([Tract Asian]), ATTR([Tract O Multir]), ATTR([Tract Hispanic]), ATTR([MHLTH CrudePrev]))
