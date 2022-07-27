##### import libraries
import pandas as pd
import numpy as np

from py02param import *


y = pd.read_csv('y.csv', index_col=0)
print(y)


print(dma)

corr = y.rolling(window=dma).corr()
print(corr)

corr = corr.dropna(how="any")
print(corr)

corr.to_csv("corr.csv")