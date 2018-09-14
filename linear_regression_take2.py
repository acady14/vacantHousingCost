#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 14 18:52:04 2018

@author: andrew cady
"""

import pandas as pd
import numpy as np
from statsmodels.formula.api import ols

dfold = pd.read_csv(r"Model1_Dataframe.csv")
dfold = dfold.drop_duplicates(subset="HANDLE")
dfold = dfold.set_index("HANDLE") #unique parcel identifier

df0 = pd.read_csv('buildgs_all.csv')
df0 = df0.drop_duplicates(subset="HANDLE")
df0 = df0.set_index("HANDLE") #unique parcel identifier

for col in ['AsdTotal']:
    strCol = df0[col].str.replace(',','')
    df0[col] = pd.to_numeric(strCol.str.replace('$',''),'coerce','integer')


otherCols = ['OnFloodBlo',
             'ResLivArea',
             'ResUnits',
             'ResFullBat',
             'ResHlfBath',
             'ResAC',
             'ResGarage',
             'ResYrBlt',
             'Acres_1',
             'v_fosho'
             ]

#a few summary stats about each neighborhood -- remove outliers


#also remove non-residential buildings, using same procedure sa in mvr.py
df1 = df0.loc[df0.ResYrBlt>0] #negative year built doesn't make sense.  remove.
df1 = df1.loc[(df1.ResUnits>=1) & (df1.ResUnits<=7)] #also if building not recorded to have any residential units or has a ton of them
df1 = df1.loc[df1.ResLivArea>0] # a few properties don't have square footage--remove
df2 = df1.loc[:,otherCols] #filter down to only the relevant coluns

#interesting view on the number of residentail units versus the residential occupancy type
df1.pivot_table(index='ResUnits',columns='ResOccType',values='AsdTotal',aggfunc=len).astype(int)

#merge the two datasets together and ensure no duplicates exist
df = dfold.merge(df2,how='inner',left_index=True,right_index=True)
df['v_fosho'] = (df.xVB_Final>1).astype(int) #indicator for definitely vacant buildings
df.dropna(subset=['AsdTotal','v_fosho'],inplace=True)

out_col  = 'AsdTotalLog'

#add in neighborhood means (fixed effects panel regression)
means = df.groupby('Nbrhd').agg({out_col:np.mean})
means.columns=['nbrhd_mean']
df = df.merge(means,how='left',left_on='Nbrhd',right_index=True)

#any other columns we think may be interesting for our regressions
df['logLivArea'] = np.log(df['ResLivArea'])


for col in ['x400ftCounts', 'x450ftCounts','x350ftCounts',
            'x300ftCounts', 'x250ftCounts','x200ftCounts',
            'x150ftCounts',  'x100ftCounts','x50ftCounts',]:
    #col = 'x450ftCounts'
    df['vacant_x_nbrhd'] = df[col] * df.nbrhd_mean
    reg_string = "{} ~ {} + nbrhd_mean + vacant_x_nbrhd + v_fosho + ResGarage + logLivArea + ResFullBat + ResHlfBath + ResAC + ResGarage + Acres_1".format(out_col,col)
    dfl = df.loc[:]
    df_model = ols(reg_string, dfl).fit()
    #print(df_model.summary())
    print('col: {}, aic: {}, pval for dist: {}'.format(col,df_model.aic,df_model.pvalues[col]))
  
    
#now that we've decided on a distance column to use, let's determine the total value of all homes in the Lou
col = 'x450ftCounts'    
df['vacant_x_nbrhd'] = df[col] * df.nbrhd_mean
reg_string = "{} ~ {} + nbrhd_mean + vacant_x_nbrhd + v_fosho + ResGarage + logLivArea + ResFullBat + ResHlfBath + ResAC + ResGarage + Acres_1".format(out_col,col)
df_model = ols(reg_string, df).fit()
pred = np.exp(df_model.predict(df))
df['all_housing_included'] = pred
EstTotalValue = df['all_housing_included'].sum() 
ActTotalValue = df.AsdTotal.sum()

# this gets us to within 5% of the actual Assessed value of all housing in St Louis
PctDiffInAsdValue = round((ActTotalValue-EstTotalValue)/ActTotalValue*100,2)
print('percent difference of estimated assessed total value against actual') 
print('assessed total value: {}%'.format(PctDiffInAsdValue))

dfz1 = df.copy()
dfz1['v_fosho'] = 0
pred = np.exp(df_model.predict(dfz1))
df['sole vacancy effect'] = pred
PrimaryCostOfVacancy = df['sole vacancy effect'].sum() - EstTotalValue

dfz2 = df.copy()
dfz2[col] = 0
dfz2['vacant_x_nbrhd'] = 0
pred = np.exp(df_model.predict(dfz2))
df['total vacancy effect'] = pred
TotalCostOfVacancy = df['total vacancy effect'].sum() - EstTotalValue
PctReduction = round((TotalCostOfVacancy/EstTotalValue)*100,2)


print('Assessed total value of residential property:    ${:,.2f}'.format(EstTotalValue))
print('Cost of vacancy on value of vacant properties:       ${:,.2f}'.format(PrimaryCostOfVacancy))
print('Total cost of vacancy on value of all properties:   ${:,.2f}'.format(TotalCostOfVacancy))
print('Percent reduction in value due to vacancy:                    {}%'.format(PctReduction))

