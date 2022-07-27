#################### Financial Turbulence, Correlation Surprise, and Absorption Ratio ####################
#
#  (C) 2021, Yoshimasa (Yoshi) Satoh, CFA 
#
#  All rights reserved.
#
# Created:      2021/10/12
# Last Updated: 2021/10/15
#
# Github:
# https://github.com/yoshisatoh/Stats/tree/main/FTI_AR/py00generatedatay.py
# https://github.com/yoshisatoh/Stats/blob/main/FTI_AR/py00generatedatay.py
#
#
########## Output Data Files
#
#df_Date.csv
#
########## Output Data Files
#
#df.csv
#df2.csv
#y.csv
#
#
####################


########## import Python libraries

import numpy as np
import pandas as pd




########## Load Day.csv

df_Date = pd.read_csv('df_Date.csv')
print(df_Date.head())




########## df and df2 data generation

# time-series data A
mu    = 0.10/(250)**0.50 # daily return: mean
sigma = 0.20/(250)**0.50 # daily return: standard deviation

np.random.seed(1)       # set 1 to fix the result of random sampling

A     = np.random.normal(mu, sigma, size=(250*5, 1))
B     = A * 1.05 + np.random.normal(0, 0.01/(250)**0.50, size=(250*5, 1))
C     = B * 0.95 + np.random.normal(0, 0.01/(250)**0.50, size=(250*5, 1))

#print(len(A))
#print(len(B))
#print(len(C))

#print(A)

df_A = pd.DataFrame(A, columns = ['A'])
df_B = pd.DataFrame(B, columns = ['B'])
df_C = pd.DataFrame(C, columns = ['C'])

df_Day = pd.DataFrame(df_A.index, columns = ['Day'])

'''
print(df_A.head())
print(df_B.head())
print(df_C.head())
print(df_Day.head())
'''
#print(df_Day.join(df_A))

df   = df_Day.join(df_A)
df   = df.join(df_B)
df   = df.join(df_C)

'''
print(df.head())
print(df['Day'].head())
print(df['A'].head())
print(df['B'].head())
print(df['C'].head())
'''

print(df.describe())
'''
               Day            A            B            C
count  1250.000000  1250.000000  1250.000000  1250.000000
mean    624.500000     0.006095     0.007254     0.005685
std     360.988227     0.012363     0.015282     0.012734
min       0.000000    -0.032206    -0.044068    -0.031762
25%     312.250000    -0.002326    -0.003033    -0.002839
50%     624.500000     0.005981     0.007636     0.005962
75%     936.750000     0.014126     0.017422     0.014215
max    1249.000000     0.046435     0.057181     0.043778
'''

#df2 = df    # This is a refernce. If you update df2, then df is updated as well.
df2 = df.copy(deep=True)    # If you update df2, then df is NOT updated.

# Update df2['A'] for the Days from 500 to 519
df2['A'][(250*2):(250*2)+20] = df2['A'][(250*2):(250*2)+20] *(+1) - (6*sigma)

# Update df2['A'] for the Days from 750 to 769
df2['A'][(250*3):(250*3)+20] = df2['A'][(250*3):(250*3)+20] *(-1)

# Update df2['A'] for the Days from 1000 to 1019
#print(df2[(250*4):(250*4)+20])
#df2['A'][(250*4):(250*4)+20] = df2['A'][(250*4):(250*4)+20] *(-1)
#df2['A'][(250*4):(250*4)+20] = df2['A'][(250*4):(250*4)+20] - (pd.Series(np.sign(df2['A'][(250*4):(250*4)+20]))) * (6*sigma)
df2['A'][(250*4):(250*4)+20] = df2['A'][(250*4):(250*4)+20] *(-1) - (6*sigma)
#print(df2[(250*4):(250*4)+20])

'''
print(df2[(250*2):(250*2)+20])
print(df2[(250*3):(250*3)+20])
print(df2[(250*4):(250*4)+20])
'''



print(df2.describe())
'''
               Day            A            B            C
count  1250.000000  1250.000000  1250.000000  1250.000000
mean    624.500000     0.005037     0.007254     0.005685
std     360.988227     0.017552     0.015282     0.012734
min       0.000000    -0.094471    -0.044068    -0.031762
25%     312.250000    -0.003257    -0.003033    -0.002839
50%     624.500000     0.005564     0.007636     0.005962
75%     936.750000     0.014126     0.017422     0.014215
max    1249.000000     0.074912     0.057181     0.043778
'''


########## df3 data generation

'''
df3   = df_Date.join(df_Day)
df3   = df3.join(df_A)
df3   = df3.join(df_B)
df3   = df3.join(df_C)
'''

df3 = df2.copy(deep=True)
df3 = df_Date.join(df3)

#print(df3[(250*4):(250*4)+20])


########## df4 data generation

df4 = df3.copy(deep=True)    # If you update df2, then df is NOT updated.
print(df4.head())
print(df4.columns)
df4 = df4.drop('Day', axis=1)


########## y data generation

#y = df4
y = df4.copy(deep=True)


########## Output results

df.to_csv('df.csv', index=False)
df2.to_csv('df2.csv', index=False)
df3.to_csv('df3.csv', index=False)
df4.to_csv('df4.csv', index=False)

y.to_csv('y.csv', index=False) 
