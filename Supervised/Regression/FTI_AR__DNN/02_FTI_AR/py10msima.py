##### import libraries
import pandas as pd
import numpy as np

from py02param import *


MSI = pd.read_csv('MSI.csv', index_col=0)
#print(MSI)



#print(FTI_pct_rank_dma)
MSIdma = MSI.rolling(window=FTI_pct_rank_dma).mean()
#print(MSIdma)

MSIdma = MSIdma.dropna(how="any")
print(MSIdma)

MSIdma.to_csv("MSIdma.csv")