##### import libraries
import pandas as pd
import numpy as np

from py02param import *

FTI = pd.read_csv('FTI.csv', index_col=0)
MSI = pd.read_csv('MSI.csv', index_col=0)
#print(MSI)



CSI = pd.concat([FTI, MSI], axis=1, join='inner')
print(CSI)
CSI = CSI.dropna(how="any")
print(CSI)
CSI['CSI'] = CSI['FTI'] / CSI['MSI']
print(CSI)

CSI = CSI.drop('FTI', axis=1)
CSI = CSI.drop('MSI', axis=1)

CSI.to_csv("CSI.csv")

#print(FTI_pct_rank_dma)
#CSIdma = CSI.rolling(window=FTI_pct_rank_dma).mean()
#print(CSIdma)

#CSIdma = CSIdma.dropna(how="any")
#print(CSIdma)

#CSIdma.to_csv("CSIdma.csv")