##### import libraries
import pandas as pd
import numpy as np

from py02param import *


y = pd.read_csv('y.csv', index_col=0)
print(y)


print(dma)

sigma = y.rolling(window=dma).std()
print(sigma)

sigma = sigma.dropna(how="any")
print(sigma)

sigma.to_csv("sigma.csv")