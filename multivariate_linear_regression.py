# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
import pandas
from numpy import exp

#cd "C:/Documents/Daugherty/Analytics/Vacancy Project/"

df = pandas.read_csv(r"Model1_Dataframe.csv")
df = df.drop_duplicates(subset="HANDLE")
df = df.set_index("HANDLE")

#df_all = pandas.read_csv(r"Buildgs_All.csv")
#df_all = df_all.drop_duplicates(subset="HANDLE")
#df_all = df_all.set_index("HANDLE")

df = df.loc[df['xResYrBlt'] > 0]
#df_all = df_all.loc[df_all['xResYrBlt'] > 0]

df_model = ols("AsdTotalLog ~ xComGrdFlr + xResGarage + xVB_Final + xResYrBlt + x300ftCounts + x250ftCounts + x150ftCounts + x400ftCounts + x450ftCounts + x100ftCounts + x50ftCounts + x350ftCounts + x200ftCounts + xIsBrick + xIsStone + xIs_0_Story + xIs_2Stories + xIs_23StoriesStories + xIs_3Stories + xIs_1Story + xIs_13StoriesStories + xIs_BiLevel + xIs_TriLevel + xIsFrame", df).fit()
print(df_model.summary())

df_model = ols("AsdTotalLog ~ xResGarage + xVB_Final + xResYrBlt + x150ftCounts + x100ftCounts + x50ftCounts + x200ftCounts + xIsStone + xIs_2Stories + xIs_3Stories + xIs_1Story + xIsFrame", df).fit()
print(df_model.summary())

#df_all_model = ols("AsdTotalLog ~ xResGarage + xVB_Final + xResYrBlt + x150ftCounts + x100ftCounts + x50ftCounts + x200ftCounts + xIsStone + xIs_2Stories + xIs_3Stories + xIs_1Story + xIsFrame", df_all).fit()
#print(df_all_model.summary())

#pandas.get_dummies(df_all.Nbrhd)

df_novacancy = df.copy()
df_novacancy['xVB_Final'] = 0
df_novacancy['x150ftCounts'] = 0
df_novacancy['x100ftCounts'] = 0
df_novacancy['x50ftCounts'] = 0
df_novacancy['x200ftCounts'] = 0

orig = df['AsdTotalLog']
unchanged = df_model.predict(df)
novacancy = df_model.predict(df_novacancy)

totalloss = sum(exp(novacancy) - exp(orig))
print(totalloss)