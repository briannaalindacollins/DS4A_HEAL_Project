"""
We used this script to calculate the Shapiro scores which we used in the EDA
"""


import seaborn as sns
import pandas as pd
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor
#from statsmodels.stats import shapiro
from scipy.stats import shapiro

df = pd.read_csv('./df_texas_cleaned.csv')
#print(df.columns)
df_r30 = df[['LILATracts_1And10', 'LILATracts_halfAnd10', 'LILATracts_1And20', 'LILATracts_Vehicle', 'LowIncomeTracts', 'PovertyRate', 'MedianFamilyIncome', 'LALOWI05_10', 'lalowihalf', 'lalowihalfshare', 'lawhitehalfshare', 'laomultirhalfshare', 'lahisphalfshare', 'lahunvhalf', 'lahunvhalfshare', 'lasnaphalf', 'lasnaphalfshare', 'TractAsian', 'TractOMultir', 'TractHispanic', 'TractHUNV'   ,'TractSNAP', 'MHLTH_CrudePrev']]
vif_data = pd.DataFrame()
#vif_data["feature"] = df_r30.columns
  
# calculating VIF for each feature
vif_data = df_r30.apply(lambda x: pd.Series(shapiro(x), index=['W','P']))
vif_data = vif_data.T
print(vif_data)  
vif_data.to_csv('./shapiro.csv')
