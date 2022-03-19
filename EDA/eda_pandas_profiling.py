"""
This script calls pandas profiling for a complimentary EDA
"""

import numpy as np
import pandas as pd
import pandas_profiling as pp

df = pd.read_csv('./df_texas_cleaned.csv')
df_r30 = df[['LILATracts_1And10', 'LILATracts_halfAnd10', 'LILATracts_1And20', 'LILATracts_Vehicle', 'LowIncomeTracts', 'PovertyRate', 'MedianFamilyIncome', 'LALOWI05_10', 'lalowihalf', 'lalowihalfshare', 'lawhitehalfshare', 'laomultirhalfshare', 'lahisphalfshare', 'lahunvhalf', 'lahunvhalfshare', 'lasnaphalf', 'lasnaphalfshare', 'TractAsian', 'TractOMultir', 'TractHispanic', 'TractHUNV'	 ,'TractSNAP', 'MHLTH_CrudePrev']]
report = pp.ProfileReport(df_r30, explorative=True, correlations={"pearson": {"calculate": True}}, interactions={"continuous": True})
report.to_file("report_basic_texas_cleaned_r30.html")
