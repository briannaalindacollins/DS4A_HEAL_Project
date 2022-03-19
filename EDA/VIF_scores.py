"""
This script calculates VIF scores which we used in our EDA
"""

import seaborn as sns
import pandas as pd
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor

df = pd.read_csv('./df_texas_cleaned.csv')
#print(df.columns)
df_r30 = df[['LILATracts_1And10', 'LILATracts_halfAnd10', 'LILATracts_1And20', 'LILATracts_Vehicle', 'LowIncomeTracts', 'PovertyRate', 'MedianFamilyIncome', 'LALOWI05_10', 'lalowihalf', 'lalowihalfshare', 'lawhitehalfshare', 'laomultirhalfshare', 'lahisphalfshare', 'lahunvhalf', 'lahunvhalfshare', 'lasnaphalf', 'lasnaphalfshare', 'TractAsian', 'TractOMultir', 'TractHispanic', 'TractHUNV'   ,'TractSNAP', 'MHLTH_CrudePrev']]
vif_data = pd.DataFrame()
vif_data["feature"] = df_r30.columns
  
# calculating VIF for each feature
vif_data["VIF"] = [variance_inflation_factor(df_r30.values, i)
                          for i in range(len(df_r30.columns))]
  
vif_data.to_csv('./vif_data.csv')
