

##### import libraries
import pandas as pd
import numpy as np

from py02param import *


CSIdma = pd.read_csv('CSIdma.csv', index_col=0)
#print(FTIdma)

#type(FTIdma)
#<class 'pandas.core.frame.DataFrame'>

#print(FTI_pct_rank_dtp)

CSIpctrank = CSIdma.rolling(FTI_pct_rank_dtp).apply(lambda x: pd.Series(x).rank().values[-1])/FTI_pct_rank_dtp

#print(FTIpctrank)

CSIpctrank = CSIpctrank.dropna(how="any")
#print(FTIpctrank)

CSIpctrank.to_csv("CSIpctrank.csv")