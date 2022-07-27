##### import libraries
import pandas as pd
import numpy as np

from py02param import *


y = pd.read_csv('y.csv')
print(y)


print(dma)

mu = pd.concat([y.iloc[:,0], y.rolling(window=dma).mean()], axis=1, join='inner')
print(mu)

mu = mu.dropna(how="any")
print(mu)

mu.to_csv("mu.csv", index=False)