##### import libraries

import pandas as pd
#import numpy as np




##### Load all files

y          = pd.read_csv('y.csv', header=0)

ymmu       = pd.read_csv('ymmu.csv', header=0)

sigma      = pd.read_csv('sigma.csv', header=0)

FTI        = pd.read_csv('FTI.csv', header=0)

FTIdma     = pd.read_csv('FTIdma.csv', header=0)

FTIpctrank = pd.read_csv('FTIpctrank.csv', header=0)

MSI        = pd.read_csv('MSI.csv', header=0)

MSIdma     = pd.read_csv('MSIdma.csv', header=0)

CSI        = pd.read_csv('CSI.csv', header=0)

CSIdma     = pd.read_csv('CSIdma.csv', header=0)

AR         = pd.read_csv('AR.csv', header=0)


##### Change column names

print(y.columns)
y.rename(columns = lambda x: 'y_' + x, inplace=True)
print(y.columns)
y.rename(columns = {'y_Date':'Date'}, inplace=True)
print(y.columns)


ymmu.rename(columns = lambda x: 'ymmu_' + x, inplace=True)
ymmu.rename(columns = {'ymmu_Date':'Date'}, inplace=True)


sigma.rename(columns = lambda x: 'sigma_' + x, inplace=True)
sigma.rename(columns = {'sigma_Date':'Date'}, inplace=True)


#FTI.rename(columns = lambda x: 'FTI_' + x, inplace=True)
#FTI.rename(columns = {'FTI_Date':'Date'}, inplace=True)


#FTIdma.rename(columns = lambda x: 'FTIdma_' + x, inplace=True)
#FTIdma.rename(columns = {'FTIdma_Date':'Date'}, inplace=True)
FTIdma.rename(columns = {'FTI':'FTIdma'}, inplace=True)


#FTIpctrank.rename(columns = lambda x: 'FTIpctrank_' + x, inplace=True)
#FTIpctrank.rename(columns = {'FTIpctrank_Date':'Date'}, inplace=True)
FTIpctrank.rename(columns = {'FTI':'FTIpctrank'}, inplace=True)


#MSI.rename(columns = lambda x: 'MSI_' + x, inplace=True)
#MSI.rename(columns = {'MSI_Date':'Date'}, inplace=True)


#MSIdma.rename(columns = lambda x: 'MSIdma_' + x, inplace=True)
#MSIdma.rename(columns = {'MSIdma_Date':'Date'}, inplace=True)
MSIdma.rename(columns = {'MSI':'MSIdma'}, inplace=True)


#CSI.rename(columns = lambda x: 'CSI_' + x, inplace=True)
#CSI.rename(columns = {'CSI_Date':'Date'}, inplace=True)


#CSIdma.rename(columns = lambda x: 'CSIdma_' + x, inplace=True)
#CSIdma.rename(columns = {'CSIdma_Date':'Date'}, inplace=True)
CSIdma.rename(columns = {'CSI':'CSIdma'}, inplace=True)


#AR.rename(columns = lambda x: 'AR_' + x, inplace=True)
#AR.rename(columns = {'AR_Date':'Date'}, inplace=True)


##### Merge all files

results = pd.merge(y, ymmu, on='Date', how='outer')

results = pd.merge(results, sigma, on='Date', how='outer')

results = pd.merge(results, FTI, on='Date', how='outer')

results = pd.merge(results, FTIdma, on='Date', how='outer')

results = pd.merge(results, FTIpctrank, on='Date', how='outer')

results = pd.merge(results, MSI, on='Date', how='outer')

results = pd.merge(results, MSIdma, on='Date', how='outer')

results = pd.merge(results, CSI, on='Date', how='outer')

results = pd.merge(results, CSIdma, on='Date', how='outer')

results = pd.merge(results, AR, on='Date', how='outer')


##### Save all files

results.to_csv("results.csv", index=False)