#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 27 16:43:59 2018

@author: andrewcady
"""

import pandas as pd
import numpy as np
from statsmodels.formula.api import ols


#ingest data, ensure no duplicates exist, set index
df = pd.read_csv(r"Model1_Dataframe.csv")
df = df.drop_duplicates(subset="HANDLE")
df = df.set_index("HANDLE") #unique parcel identifier

df['v_fosho'] = (df.xVB_Final>1).astype(int) #indicator for definitely vacant buildings

#a few summary stats about each neighborhood -- remove outliers
nbrhd_pt = df.groupby('Nbrhd').agg({'AsdTotal':[np.median,np.mean,np.std,len],'v_fosho':sum})
nbrhd_pt.columns = ['median_assd_value','mean_assd_value','assd_value_std','num_parcels','num_vacant']
nbrhd_pt.sort_values('mean_assd_value',inplace=True,ascending=True) #sort smallest to largest
nbrhd_pt = nbrhd_pt.loc[nbrhd_pt.num_parcels>100] #remove outlier neighborhoods with very few parcels 
nbrhd_pt = nbrhd_pt.loc[nbrhd_pt.assd_value_std<50000] #also remove a neighborhood with huge variation in housing prices
nbrhd_pt['pct_vacant'] = nbrhd_pt.num_vacant/nbrhd_pt.num_parcels

#pick out a few representative neighborhoods for testing purposes
med_nbrhd = nbrhd_pt.iloc[len(nbrhd_pt)//2].name
lowVal_nbrhd = nbrhd_pt.iloc[len(nbrhd_pt)//10].name
hiVal_nbrhd = nbrhd_pt.iloc[int(len(nbrhd_pt)*0.9)].name
print('neighborhoods:')
print('low: {} avg assessed price: {}'.format(lowVal_nbrhd,round(nbrhd_pt.loc[lowVal_nbrhd].mean_assd_value)))
print('median: {} avg assessed price: {}'.format(med_nbrhd,round(nbrhd_pt.loc[med_nbrhd].mean_assd_value)))
print('high: {} avg assessed price: {}'.format(hiVal_nbrhd,round(nbrhd_pt.loc[hiVal_nbrhd].mean_assd_value)))

#subset main data frame
df2 = df.loc[df.Nbrhd.isin([med_nbrhd,lowVal_nbrhd,hiVal_nbrhd])]

df = df.loc[df.Nbrhd==med_nbrhd]
df_model = ols("AsdTotalLog ~  xResGarage + v_fosho + xResYrBlt + x250ftCounts + xIs_0_Story + xIs_2Stories  + xIs_3Stories", dfl).fit()
print(df_model.summary())



###############################################################################

df_all = pd.read_csv('buildgs_all.csv')
df_all = df_all.drop_duplicates(subset="HANDLE")
df_all = df_all.set_index("HANDLE") #unique parcel identifier

df_all['v_fosho'] = (df_all.VB_Final>1).astype(int) #indicator for definitely vacant buildings

#a few summary stats about each neighborhood -- remove outliers
df_all.dropna(subset=['AsdTotal','v_fosho'],inplace=True)

#also remove non-residential buildings, using same procedure sa in mvr.py
df_all = df_all.loc[df_all.ResYrBlt>0]

# remove dollar signs, commas, and such from monetary columns in the dataset
for col in ['AsdTotal']:
    strCol = df_all[col].str.replace(',','')
    df_all[col] = pd.to_numeric(strCol.str.replace('$',''),'coerce','integer')

df_all.pivot_table(index='ResUnits',columns='ResOccType',values='AsdTotal',aggfunc=len)

nbrhd_pt = df_all.groupby('Nbrhd').agg({'AsdTotal':[np.median,np.mean,np.std,len],'v_fosho':sum})
nbrhd_pt.columns = ['median_assd_value','mean_assd_value','assd_value_std','num_parcels','num_vacant']
nbrhd_pt.sort_values('mean_assd_value',inplace=True,ascending=True) #sort smallest to largest
nbrhd_pt = nbrhd_pt.loc[nbrhd_pt.num_parcels>100] #remove outlier neighborhoods with very few parcels 
nbrhd_pt = nbrhd_pt.loc[nbrhd_pt.assd_value_std<50000] #also remove a neighborhood with huge variation in housing prices
nbrhd_pt['pct_vacant'] = nbrhd_pt.num_vacant/nbrhd_pt.num_parcels

#pick out a few representative neighborhoods for testing purposes
med_nbrhd = nbrhd_pt.iloc[len(nbrhd_pt)//2].name
lowVal_nbrhd = nbrhd_pt.iloc[len(nbrhd_pt)//10].name
hiVal_nbrhd = nbrhd_pt.iloc[int(len(nbrhd_pt)*0.9)].name
print('neighborhoods:')
print('low: {} avg assessed price: {}'.format(lowVal_nbrhd,round(nbrhd_pt.loc[lowVal_nbrhd].mean_assd_value)))
print('median: {} avg assessed price: {}'.format(med_nbrhd,round(nbrhd_pt.loc[med_nbrhd].mean_assd_value)))
print('high: {} avg assessed price: {}'.format(hiVal_nbrhd,round(nbrhd_pt.loc[hiVal_nbrhd].mean_assd_value)))

#subset main data frame
#df_all = df_all.loc[df.Nbrhd.isin([med_nbrhd,lowVal_nbrhd,hiVal_nbrhd])]
