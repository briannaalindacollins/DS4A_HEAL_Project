"""
This was used to plot some of the graphs in the Categorical EDA, here most of these graphs are commented out, this script was used to produce several different graphs
"""

import seaborn as sns
import pandas as pd
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from tabulate import tabulate
import numpy as np
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

df = pd.read_csv('./df_texas_cleaned.csv')
#print(df.columns)
#df_r30 = df[['LILATracts_1And10', 'LILATracts_halfAnd10', 'LILATracts_1And20', 'LILATracts_Vehicle', 'LowIncomeTracts', 'PovertyRate', 'MedianFamilyIncome', 'LALOWI05_10', 'lalowihalf', 'lalowihalfshare', 'lawhitehalfshare', 'laomultirhalfshare', 'lahisphalfshare', 'lahunvhalf', 'lahunvhalfshare', 'lasnaphalf', 'lasnaphalfshare', 'TractAsian', 'TractOMultir', 'TractHispanic', 'TractHUNV'   ,'TractSNAP', 'MHLTH_CrudePrev']]
#df_r30 = df_r30.drop_duplicates(subset=['Census_tract'])
#print(df_r30.info(verbose=True))
#hist = df_r30.hist(column=['laomultirhalfshare', 'lahisphalfshare', 'lahunvhalf', 'lahunvhalfshare', 'lasnaphalf', 'lasnaphalfshare', 'TractAsian', 'TractOMultir', 'TractHispanic', 'TractHUNV'   ,'TractSNAP', 'MHLTH_CrudePrev'])
#hist = df_r30.hist()
#df_split_1 = df[['PovertyRate', 'MedianFamilyIncome', 'LALOWI05_10', 'lalowihalf', 'lalowihalfshare', 'lawhitehalfshare', 'MHLTH_CrudePrev']]
#df_split_2 = df[['lahisphalfshare', 'lahunvhalf', 'lahunvhalfshare', 'lasnaphalf', 'lasnaphalfshare', 'MHLTH_CrudePrev']]
#df_split_3 = df[['TractAsian', 'TractOMultir', 'TractHispanic', 'TractHUNV','TractSNAP', 'MHLTH_CrudePrev', 'laomultirhalfshare']]
#df_split_4 = df[['LILATracts_1And10', 'LILATracts_halfAnd10', 'LILATracts_1And20', 'LILATracts_Vehicle', 'LowIncomeTracts', 'MHLTH_CrudePrev']]
#stats_1 = df_split_1.describe()
#stats_1.to_html('df_split_1.html')
#stats_2 = df_split_2.describe()
#stats_2.to_html('df_split_2.html')
#stats_3 = df_split_3.describe()
#stats_3.to_html('df_split_3.html')
#corr = df_r30.corr() 
#sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, cmap='coolwarm')
#fig, axes = plt.subplots(1,len(df_split_2.columns.values)-1, sharey=True)

#for i, col in enumerate(df_split_2.columns.values[:-1]):
#    df_split_2.plot(x=[col], y=['MHLTH_CrudePrev'], kind="scatter", ax=axes[i])

#plt.show()
#plt.show()
#plt.show()

#print(tabulate(stats, headers = 'keys', tablefmt = 'psql'))
#plt.show()
df_cat = df[['LILATracts_1And10', 'LILATracts_halfAnd10', 'LILATracts_1And20', 'LILATracts_Vehicle', 'LowIncomeTracts', 'MHLTH_CrudePrev', 'PovertyRate']]
###Numeric vs. Categorical EDA; Multiple Histograms
#fig, ax = plt.subplots()
#ax.hist(df_cat[df_cat["LILATracts_1And10"]==1]["MHLTH_CrudePrev"], bins=15, alpha=0.5, color="blue", label="Yes")
#ax.hist(df_cat[df_cat["LILATracts_1And10"]==0]["MHLTH_CrudePrev"], bins=15, alpha=0.5, color="green", label="No")
#ax.set_xlabel("MHLTH_CrudePrev")
#ax.set_ylabel("LILATracts_1And10")
#fig.suptitle("Flag for low income and low access tract versus mental health score")
#ax.legend()
#plt.show()
###Numeric vs. Categorical EDA; Multiple Density Estimate Plots
#fig, ax = plt.subplots()
#
#sns.kdeplot(df_cat[df_cat["LowIncomeTracts"]==1]["MHLTH_CrudePrev"], shade=True, color="blue", label="Yes", ax=ax)
#sns.kdeplot(df_cat[df_cat["LowIncomeTracts"]==0]["MHLTH_CrudePrev"], shade=True, color="green", label="No", ax=ax)
#
#ax.set_xlabel("MHLTH_CrudePrev")
#ax.set_ylabel("LowIncomeTracts")
#
#fig.suptitle("MHLTH_CrudePrev vs. LowIncomeTracts")
#ax.legend()
#plt.show()
#######Numeric vs. Categorical EDA; Multiple Box Plots
#fig, ax = plt.subplots()
#
#sns.boxplot(x="MHLTH_CrudePrev", y="LowIncomeTracts", data=df_cat, orient="h", palette={1:"blue", 0:"green"}, ax=ax)
#
#ax.get_yaxis().set_visible(False)
#
#fig.suptitle("MHLTH_CrudePrev vs. LowIncomeTracts")
#
#color_patches = [
#    Patch(facecolor="blue", label="Yes"),
#    Patch(facecolor="green", label="No")
#]
#ax.legend(handles=color_patches);
#plt.show()
####Categorical vs. Categorical; Grouped Bar Charts 
#df_cat = df[['LILATracts_1And10', 'LILATracts_halfAnd10', 'LILATracts_1And20', 'LILATracts_Vehicle', 'LowIncomeTracts', 'MHLTH_CrudePrev']]
#sns.catplot(x='LILATracts_Vehicle', hue='LowIncomeTracts', data=df_cat, kind="count")
#plt.show()
####Categorical vs. Categorical; Stacked Bar Charts
#pd.crosstab(df_cat['LILATracts_Vehicle'],df_cat['LowIncomeTracts']).plot.bar(stacked=True)
#plt.show()
####Numeric vs. Numeric vs. Categorical EDA; Color-coded scatter plots

fig, ax = plt.subplots(figsize=(10, 5))

ax.scatter(df_cat[df_cat["LowIncomeTracts"]==1]["MHLTH_CrudePrev"], df_cat[df_cat["LowIncomeTracts"]==1]["PovertyRate"], c="blue", alpha=0.5)
ax.scatter(df_cat[df_cat["LowIncomeTracts"]==0]["MHLTH_CrudePrev"], df_cat[df_cat["LowIncomeTracts"]==0]["PovertyRate"], c="green", alpha=0.5)

ax.set_xlabel("MHLTH_CrudePrev")
ax.set_ylabel("PovertyRate")

color_patches = [
    Line2D([0], [0], marker='o', color='w', label='Yes', markerfacecolor='b', markersize=10),
    Line2D([0], [0], marker='o', color='w', label='No', markerfacecolor='g', markersize=10)
]
ax.legend(handles=color_patches)
fig.suptitle("LowIncomeTracts vs. MHLTH_CrudePrev vs. PovertyRate")
plt.show()
