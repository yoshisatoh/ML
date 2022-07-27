

##### import libraries
import pandas as pd
import numpy as np

#You have to run the following code on Terminal of MacOS before import.
#pip3 uninstall scikit-learn 
#pip3 install -U scikit-learn
#pip3 install sklearn
#pip3 install -U numpy
#pip3 install -U scipy
#
import sklearn # machine learning library
from sklearn.decomposition import PCA #PCA: Principal Component Analysis

from py02param import *


y = pd.read_csv('y.csv', index_col=0)
ypca = y

#do PCA
pca = PCA()
pca.fit(ypca)
feature = pca.transform(ypca)

# window size
#print(dma)

# N = 20% of the number of principal components in integer
N = int(round(len(ypca.columns) * 0.20, 0))

ypca2 = ypca.index.values

ypca2 = pd.Series(ypca2)

i = 0

#ypca3 = pd.concat([pd.Series(ypca2[dma+i-1:dma+i].iloc[0]), pd.Series(pca.explained_variance_ratio_[N-1])], axis=1)

for i in range(int(len(ypca)-dma+1)):
    pca.fit(ypca[i:dma+i])
    pca.explained_variance_ratio_
    if i == 0:
        #ypca3 = pd.concat([pd.Series(ypca2[dma+i-1:dma+i].iloc[0]), pd.Series(pca.explained_variance_ratio_[N-1])], axis=1)
        #ypca3 = pd.concat([pd.Series(ypca2[(dma+i-1):(dma+i)].iloc[0]), pd.Series(pca.explained_variance_ratio_[N-1])], axis=1)
        #ypca3 = pd.concat([pd.Series(ypca2[(dma+i-1):(dma+i)].iloc[0]), pd.Series(pca.explained_variance_ratio_[N-1])], axis=1)
        ypca3 = pd.concat([pd.Series(ypca2[(dma+i-1):(dma+i)].iloc[0]), pd.Series(np.cumsum(pca.explained_variance_ratio_)[N-1])], axis=1)
    else:
        #ypca3 = pd.concat([ypca3, pd.concat([pd.Series(ypca[dma+i-1:dma+i].index.values), pd.Series(pca.explained_variance_ratio_[N-1])], axis=1)], axis=0)
        #ypca3 = pd.concat([ypca3, pd.concat([pd.Series(ypca[(dma+i-1):(dma+i)].index.values), pd.Series(pca.explained_variance_ratio_[N-1])], axis=1)], axis=0)
        ypca3 = pd.concat([ypca3, pd.concat([pd.Series(ypca[(dma+i-1):(dma+i)].index.values), pd.Series(np.cumsum(pca.explained_variance_ratio_)[N-1])], axis=1)], axis=0)

i = 0

#ypca3 = ypca3.rename(columns = {ypca3.columns[0]: "Trade Date", ypca3.columns[1]: "AR"})
#ypca3 = ypca3.rename(columns = {ypca3.columns[0]: "Day", ypca3.columns[1]: "AR"})
ypca3 = ypca3.rename(columns = {ypca3.columns[0]: "Date", ypca3.columns[1]: "AR"})

#ypca3.set_index("Trade Date",inplace=True)
#ypca3.set_index("Day",inplace=True)
ypca3.set_index("Date",inplace=True)

#print(ypca3[['Trade Date', 'AR']])
#print(ypca3[['Day', 'AR']])
print(ypca3['AR'])

AR = ypca3

AR.to_csv("AR.csv")
