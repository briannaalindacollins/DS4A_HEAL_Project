"""
Calculated correlation matrix
"""
import pandas as pd

df = pd.read_csv('./df_texas_cleaned.csv')
corr = df.corr() 
corr.to_csv('df_texas_cleaned_corr_matrix.csv')
