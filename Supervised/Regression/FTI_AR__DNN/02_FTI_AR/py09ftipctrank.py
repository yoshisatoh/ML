

##### import libraries
import pandas as pd
import numpy as np

from py02param import *


FTIdma = pd.read_csv('FTIdma.csv', index_col=0)
#print(FTIdma)

#type(FTIdma)
#<class 'pandas.core.frame.DataFrame'>

#print(FTI_pct_rank_dtp)

FTIpctrank = FTIdma.rolling(FTI_pct_rank_dtp).apply(lambda x: pd.Series(x).rank().values[-1])/FTI_pct_rank_dtp

#print(FTIpctrank)

FTIpctrank = FTIpctrank.dropna(how="any")
#print(FTIpctrank)

FTIpctrank.to_csv("FTIpctrank.csv")