"""
This script did chi2 testing which we used in out categorical Eda
""


import seaborn as sns
import pandas as pd
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from tabulate import tabulate

df = pd.read_csv('./df_texas_cleaned.csv')
#print(df.columns)
df_r30 = df[['LILATracts_1And10', 'LILATracts_halfAnd10', 'LILATracts_1And20', 'LILATracts_Vehicle', 'LowIncomeTracts', 'PovertyRate', 'MedianFamilyIncome', 'LALOWI05_10', 'lalowihalf', 'lalowihalfshare', 'lawhitehalfshare', 'laomultirhalfshare', 'lahisphalfshare', 'lahunvhalf', 'lahunvhalfshare', 'lasnaphalf', 'lasnaphalfshare', 'TractAsian', 'TractOMultir', 'TractHispanic', 'TractHUNV'   ,'TractSNAP', 'MHLTH_CrudePrev']]
# chi-squared test with similar proportions
from scipy.stats import chi2_contingency
from scipy.stats import chi2
# contingency table
table = pd.crosstab(df_r30.LILATracts_halfAnd10, df.LILATracts_1And20)
stat, p, dof, expected = chi2_contingency(table)
#print('dof=%d' % dof)
#print(expected)
print(float(p), 'float(p)')
# interpret test-statistic
prob = 0.95
critical = chi2.ppf(prob, dof)
print('probability=%.3f, critical=%.3f, stat=%.3f' % (prob, critical, stat))
if abs(stat) >= critical:
	print('Dependent (reject H0)')
else:
	print('Independent (fail to reject H0)')
# interpret p-value
alpha = 1.0 - prob
print(alpha, p)
if p <= alpha:
	print('Dependent (reject H0)')
else:
	print('Independent (fail to reject H0)')
