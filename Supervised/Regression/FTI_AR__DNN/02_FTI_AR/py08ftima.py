##### import libraries
import pandas as pd
import numpy as np

from py02param import *


FTI = pd.read_csv('FTI.csv', index_col=0)
#print(FTI)



print(FTI_pct_rank_dma)
FTIdma = FTI.rolling(window=FTI_pct_rank_dma).mean()
#print(FTIdma)

FTIdma = FTIdma.dropna(how="any")
print(FTIdma)

FTIdma.to_csv("FTIdma.csv")