"""


This script takes the results from the concate_data script and filters out data that is not needed

"""

import pandas as pd


df = pd.read_csv('./total.csv')
#print(df.info(verbose=True))
#print(df.head(20))
df_texas = df.loc[df['StateAbbr'] == 'TX']
#print(df_texas.describe())
#print(df_texas.info())
#save_des = df_texas.describe()
df_texas['Census_tract'] = df_texas['CensusTract_x']
df_texas = df_texas.set_index('Census_tract')
list_drop = ['CensusTract_x', 'CensusTract_y', 'COLON_SCREEN_CrudePrev', 'COREM_CrudePrev', 'COREW_CrudePrev', 'MAMMOUSE_CrudePrev', 'PAPTEST_CrudePrev', 'TEETHLOST_CrudePrev', 'COLON_SCREEN_Crude95CI', 'COREM_Crude95CI', 'COREW_Crude95CI', 'MAMMOUSE_Crude95CI', 'PAPTEST_Crude95CI', 'TEETHLOST_Crude95CI', 'CensusTract_new', 'Unnamed: 0', 'PlaceFIPS', 'TractFIPS', 'POP2010', 'len(censustract)']
for i in range(len(list_drop)):
    drop = list_drop[i]
    df_texas = df_texas.drop(drop, 1)  

print(df_texas.isna().sum().sum())


df_texas.to_csv('df_texas_cleaned.csv')
