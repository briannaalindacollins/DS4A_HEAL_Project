"""
We used this script to plot most of the graphs for our basic EDA
"""

import seaborn as sns
import pandas as pd
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from tabulate import tabulate

df = pd.read_csv('./df_texas_cleaned.csv')
#print(df.columns)
df_r30 = df[['LILATracts_1And10', 'LILATracts_halfAnd10', 'LILATracts_1And20', 'LILATracts_Vehicle', 'LowIncomeTracts', 'PovertyRate', 'MedianFamilyIncome', 'LALOWI05_10', 'lalowihalf', 'lalowihalfshare', 'lawhitehalfshare', 'laomultirhalfshare', 'lahisphalfshare', 'lahunvhalf', 'lahunvhalfshare', 'lasnaphalf', 'lasnaphalfshare', 'TractAsian', 'TractOMultir', 'TractHispanic', 'TractHUNV'   ,'TractSNAP', 'MHLTH_CrudePrev']]
#df_r30 = df_r30.drop_duplicates(subset=['Census_tract'])
#print(df_r30.info(verbose=True))
#hist = df_r30.hist(column=['laomultirhalfshare', 'lahisphalfshare', 'lahunvhalf', 'lahunvhalfshare', 'lasnaphalf', 'lasnaphalfshare', 'TractAsian', 'TractOMultir', 'TractHispanic', 'TractHUNV'   ,'TractSNAP', 'MHLTH_CrudePrev'])
#hist = df_r30.hist()
df_split_1 = df[['PovertyRate', 'MedianFamilyIncome', 'LALOWI05_10', 'lalowihalf', 'lalowihalfshare', 'lawhitehalfshare', 'MHLTH_CrudePrev']]
df_split_2 = df[['lahisphalfshare', 'lahunvhalf', 'lahunvhalfshare', 'lasnaphalf', 'lasnaphalfshare', 'MHLTH_CrudePrev']]
df_split_3 = df[['TractAsian', 'TractOMultir', 'TractHispanic', 'TractHUNV','TractSNAP', 'MHLTH_CrudePrev', 'laomultirhalfshare']]
df_split_4 = df[['LILATracts_1And10', 'LILATracts_halfAnd10', 'LILATracts_1And20', 'LILATracts_Vehicle', 'LowIncomeTracts', 'MHLTH_CrudePrev']]
#stats_1 = df_split_1.describe()
#stats_1.to_html('df_split_1.html')
#stats_2 = df_split_2.describe()
#stats_2.to_html('df_split_2.html')
#stats_3 = df_split_3.describe()
#stats_3.to_html('df_split_3.html')
#corr = df_r30.corr() 
#sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, cmap='coolwarm')
fig, axes = plt.subplots(1,len(df_split_2.columns.values)-1, sharey=True)

for i, col in enumerate(df_split_2.columns.values[:-1]):
    df_split_2.plot(x=[col], y=['MHLTH_CrudePrev'], kind="scatter", ax=axes[i])

plt.show()
#plt.show()
#plt.show()

#print(tabulate(stats, headers = 'keys', tablefmt = 'psql'))
#plt.show()
