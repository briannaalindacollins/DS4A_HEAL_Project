"""
This is the simpler model written in a way that tabpy can read. Used in the backend of the application
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

sc = StandardScaler()

povertyrate = sc.fit_transform(np.array(_arg2).reshape(1, -1))
MedianFamilyIncome=sc.fit_transform(np.array(_arg3).reshape(1, -1))
food_access = sc.fit_transform(np.array(_arg1).reshape(1, -1))

#povertyrate = np.array(_arg2)
#MedianFamilyIncome=np.array(_arg3)
#food_access = np.array(_arg1)

#calc input

calc_1 = sc.fit_transform(np.array(_arg5).reshape(1,-1))
calc_2 = sc.fit_transform(np.array(_arg6).reshape(1, -1))
calc_3 = sc.fit_transform(np.array(_arg7).reshape(1, -1))

calc_input = [calc_1, calc_2, calc_3]

list = [povertyrate, food_access, MedianFamilyIncome]

x_all = np.array(list).reshape(-1,3)
x_calc = np.array(calc_input).reshape(-1,3)

y = np.array(_arg4)

x = np.where(np.isnan(x_all), ma.array(x_all, mask=np.isnan(x_all)).mean(axis=0), x_all)

#print(x.shape)
#print(y.shape)

x_calc =  np.where(np.isnan(x_calc), ma.array(x_calc, mask=np.isnan(x_calc)).mean(axis=0), x_calc)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)
GS = MLPRegressor(activation='relu', early_stopping=False, learning_rate='adaptive', max_iter=10000, warm_start=False)
GS.fit(x_train, y_train)
predicted_category = GS.predict(x_calc)
predicted_category = predicted_category.tolist()
out = predicted_category[0]

return out"
,
ATTR([Food_Access]),
ATTR([Poverty Rate]), ATTR([Median Family Income]), ATTR([MHLTH CrudePrev]), ATTR([What is the poverty rate of the area?]), ATTR([Is there access to food?]), ATTR([What is the median family income in the area?]))
