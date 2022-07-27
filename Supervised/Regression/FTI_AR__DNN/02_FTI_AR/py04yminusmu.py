##### import libraries
import pandas as pd
import numpy as np

from py02param import *


y = pd.read_csv('y.csv', index_col=0)
mu = pd.read_csv('mu.csv', index_col=0)

print(y)
print(mu)


ymmu = y - mu
print(ymmu)

ymmu = ymmu.dropna(how="any")
print(ymmu)

ymmu.to_csv("ymmu.csv")
