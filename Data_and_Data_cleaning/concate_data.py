
"""
This script concats the two original data sets together
"""

import pandas as pd
import numpy as np
#import censusgeocode as cg

df_PLACES = pd.read_csv('./500_Cities__Census_Tract-level_Data__GIS_Friendly_Format___2018_release.csv')
df_FOODATLAS = pd.read_csv('./FoodAccessResearchAtlasData2015_data.csv')

#print(df_PLACES.info())
#print(df_FOODATLAS.info(verbose=True))
#print(df_PLACES['Place_TractID'].head())
#print(df_FOODATLAS['CensusTract'].head())

df_PLACES = df_PLACES.dropna(subset=['Place_TractID'])
df_PLACES['Place_TractID_cp'] = df_PLACES['Place_TractID'].apply(lambda x: x.split('-'))
df2 = pd.DataFrame(df_PLACES['Place_TractID_cp'])
df2[['Place_TractID_2','CENSUS_TRACT']] = pd.DataFrame(df2.Place_TractID_cp.tolist(), index= df2.index)
df_PLACES['CensusTract'] = df2['CENSUS_TRACT']
#print(df_PLACES['CensusTract'].head())
df_PLACES['CensusTract'] = df_PLACES['CensusTract'].astype(object)
df_FOODATLAS['CensusTract'] = df_FOODATLAS['CensusTract'].astype(object)
df_PLACES = df_PLACES.sort_values(by=['StateAbbr'])
df_FOODATLAS = df_FOODATLAS.sort_values(by=['State'])
df_FOODATLAS['CensusTract'] = df_FOODATLAS['CensusTract'].astype(str)
df_FOODATLAS['len(censustract)'] = df_FOODATLAS['CensusTract'].apply(lambda x: len(x))
df_FOODATLAS['CensusTract_new'] = np.nan
for index, row in df_FOODATLAS.iterrows():
    census = str(row['CensusTract'])
    len_num = row['len(censustract)']
    if len_num == 10:
       census = '0' + census
    df_FOODATLAS.loc[index, 'CensusTract_new'] = census


#print(df_FOODATLAS['CensusTract_new'].tail())
#print(df_PLACES.head(100))
#print(df_FOODATLAS.head(100))
df_total = pd.merge(df_PLACES, df_FOODATLAS, left_on='CensusTract', right_on='CensusTract_new')
df_total.to_csv('total.csv')
#print(df_total.head())









#df_PLACES['census_tract'] = df_PLACES['Place_TractID_cp'].apply(lambda x: np.array(x))
#df_PLACES['census_tract'] = df_PLACES['census_tract'].apply(lambda x: x[1])
#print(df_PLACES['census_tract'].tail())
#df_PLACES['Geolocation_cp'] = df_PLACES['Geolocation']
#df_PLACES['Geolocation_cp'] = df_PLACES['Geolocation_cp'].apply(lambda x: x.replace("POINT", ""))
#df_PLACES['Geolocation_cp'] = df_PLACES['Geolocation_cp'].apply(lambda x: x.replace(")", ""))
#df_PLACES['Geolocation_cp'] = df_PLACES['Geolocation_cp'].apply(lambda x: x.replace("(", ""))
#df_PLACES['Geolocation_cp'] = df_PLACES['Geolocation_cp'].apply(lambda x: x.split())
#print(df_PLACES['Geolocation_cp'].head())
#df_PLACES['Geolocation_lat'] = df_PLACES['Geolocation_cp'].apply(lambda x: x[0])
#df_PLACES['Geolocation_long'] = df_PLACES['Geolocation_cp'].apply(lambda x: x[1])
#print(df_PLACES['Geolocation_lat'].head())
#print(df_PLACES['Geolocation_long'].head())
#result = cg.coordinates(x=-76, y=41)
#print(result)

