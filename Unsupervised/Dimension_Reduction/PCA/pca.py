#################### Unsupervised Learning: Principal Component Analysis (PCA) for Dimension Reduction ####################
#
#  (C) 2021, Yoshimasa (Yoshi) Satoh, CFA 
#
#  All rights reserved.
#
# Created:      2021/10/02
# Last Updated: 2021/10/02
#
# Github:
# https://github.com/yoshisatoh/ML/tree/main/Unsupervised/Dimension_Reduction/PCA/pca.py
# https://github.com/yoshisatoh/ML/blob/main/Unsupervised/Dimension_Reduction/PCA/pca.py
#
#
########## Input Data File(s)
#
#X.csv
#
#
########## Usage Instructions
#
#Run this script on Terminal of MacOS (or Command Prompt of Windows) as follows:
#
#python pca.py X.csv 2 x1 x2
#
#Generally,
#python pca.py (X file that includes columns x1 and x2) (number of PCA components) (column x1) (column x2)
#
#
########## References
#
#In Depth: Principal Component Analysis
#https://jakevdp.github.io/PythonDataScienceHandbook/05.09-principal-component-analysis.html
#
#
####################




########## import Python libraries

import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

from sklearn.decomposition import PCA




########## arguments

#print(sys.argv[0])
#pca.py
#
Xfilename = sys.argv[1]
n         = int(sys.argv[2])
x1name    = sys.argv[3]
x2name    = sys.argv[4]




########## raw data plotting

#rng = np.random.RandomState(1)
#
#X = np.dot(rng.rand(2, 2), rng.randn(2, 200)).T
#
#
#print(type(X))
#<class 'numpy.ndarray'>
#print(X.shape)
#
#colnames=['x1', 'x2'] 
#pd.DataFrame(X).to_csv('X.csv', header=False, index=False)
#
#add x1, x2 to the columns of X.csv

X = pd.read_csv(Xfilename, header=0)
#
#print(X)
#print(X[x1name])
#print(X[x2name])

#plt.scatter(X[:, 0], X[:, 1])
plt.scatter(X[x1name], X[x2name])
plt.xlabel(x1name)
plt.ylabel(x2name)
plt.axis('equal')
plt.savefig('Fig_1.png')
plt.show()
plt.close()




########## PCA fitting

pca = PCA(n_components=n)
pca.fit(X)

print(pca.components_)
'''
[[-0.94446029 -0.32862557]
 [-0.32862557  0.94446029]]
'''
'''
       x1           x2
pc1    -0.94446029 -0.32862557
pc2    -0.32862557  0.94446029

Namely,
pc1 (calculated) = ((-0.94446029) * x1) + ((-0.32862557) * x2))
pc2 (calculated) = ((-0.32862557) * x1) + (( 0.94446029) * x2))
'''

print(pca.explained_variance_)
#[0.7625315 0.0184779]

print(pca.explained_variance_ratio_)
#[0.97634101 0.02365899]

print(np.cumsum(pca.explained_variance_ratio_))
#[0.97634101 1.        ]




########## vector drawing function

def draw_vector(v0, v1, ax=None):
    ax = ax or plt.gca()
    arrowprops=dict(arrowstyle='->',
                    linewidth=2,
                    color='k',
                    shrinkA=0, shrinkB=0)
    ax.annotate('', v1, v0, arrowprops=arrowprops)




########## plot data x1 & x2 and principal components PC1 & PC2

fig, ax = plt.subplots(1, 2, figsize=(16, 6))
fig.subplots_adjust(left=0.0625, right=0.95, wspace=0.1)


##### PCA
X_pca = pca.transform(X)

#print(type(X_pca[:, 0]))
#<class 'numpy.ndarray'>
#
#print(max(X_pca[:, 0]))
#2.6580358349697173
#
#print(np.argmax(X_pca[:, 0]))
#50
#
#print(X_pca[:, 0][np.argmax(X_pca[:, 0])])
#2.6580358349697173
#
#print(max(X_pca[:, 1]))
#0.393203001154575
#
#print(X_pca[:, 1][np.argmax(X_pca[:, 1])])
#0.393203001154575


##### raw data

#X[:, 0]
#print(X[x1name][np.argmax(X_pca[:, 0])])
#-2.488382767219812
#
#print(X[x2name][np.argmax(X_pca[:, 0])])
#-0.844571248983918

#X[:, 1]
#print(X[x1name][np.argmax(X_pca[:, 1])])
#-0.3526694737890958
#
#print(X[x2name][np.argmax(X_pca[:, 1])])
#0.2778729045016903


##### validation: PCA data and raw data

###pc1
#
#print(((X[x1name][np.argmax(X_pca[:, 0])]) ** 2) + ((X[x2name][np.argmax(X_pca[:, 0])]) ** 2))
#6.905349390806785
#(-2.488382767219812) ** 2 + (-0.844571248983918) ** 2
#
#print(max(X_pca[:, 0]) ** 2)
#7.065154499983162
#(2.6580358349697173) ** 2

###pc2
#
#print(((X[x1name][np.argmax(X_pca[:, 1])]) ** 2) + ((X[x2name][np.argmax(X_pca[:, 1])]) ** 2))
#0.2015891087988832
#(-0.3526694737890958) ** 2 + (0.2778729045016903) ** 2
#
#print((max(X_pca[:, 1])) ** 2)
#0.15460860011696473
#(0.393203001154575) ** 2


##### plot raw data

#ax[0].scatter(X[:, 0], X[:, 1], alpha=0.2)
ax[0].scatter(X[x1name], X[x2name], alpha=0.2)
for length, vector in zip(pca.explained_variance_, pca.components_):
    v = vector * 3 * np.sqrt(length)
    draw_vector(pca.mean_, pca.mean_ + v, ax=ax[0])
#
ax[0].axis('equal')
#ax[0].set(xlabel='x', ylabel='y', title='input')
ax[0].set(xlabel=x1name, ylabel=x2name, title='Raw Data')
#
ax[0].text(X[x1name][np.argmax(X_pca[:, 0])] * 1.00, X[x2name][np.argmax(X_pca[:, 0])] * 1.15, '(' + str(X[x1name][np.argmax(X_pca[:, 0])]) + ', ' + str(X[x2name][np.argmax(X_pca[:, 0])]) + ')')
ax[0].text(X[x1name][np.argmax(X_pca[:, 1])] * 1.10, X[x2name][np.argmax(X_pca[:, 1])] * 1.30, '(' + str(X[x1name][np.argmax(X_pca[:, 1])]) + ', ' + str(X[x2name][np.argmax(X_pca[:, 1])]) + ')')
#


##### plot principal components

ax[1].scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.2)
#
#draw_vector([0, 0], [0, 3], ax=ax[1])
draw_vector([0, 0], [0, max(X_pca[:, 1])], ax=ax[1])
#
#draw_vector([0, 0], [3, 0], ax=ax[1])
draw_vector([0, 0], [max(X_pca[:, 0]),0], ax=ax[1])
#
ax[1].axis('equal')
'''
ax[1].set(xlabel='PC1', ylabel='PC2',
          title='Principal Components',
          xlim=(-5, 5), ylim=(-3, 3.1))
'''
#print(max(X_pca[:, 0]))
#print(min(X_pca[:, 0]))
#print(max(X_pca[:, 1]))
#print(min(X_pca[:, 1]))
#
ax[1].set(
          xlabel='PC1',
          ylabel='PC2',
          title='Principal Components',
          xlim=(min(X_pca[:, 0]), max(X_pca[:, 0])),
          ylim=(min(X_pca[:, 1]), max(X_pca[:, 1]))
          )
#
ax[1].text(max(X_pca[:, 0])/2, 0.05, '(' + str(max(X_pca[:, 0])) + ', 0)')
ax[1].text(0.05, max(X_pca[:, 1])/2, '(0, ' + str(max(X_pca[:, 1])) + ')')
#
plt.savefig('Fig_2.png')
plt.show()
plt.close()




########## output

##### output principal components

pc1 = pd.DataFrame(X_pca[:, 0])
pc1.rename({0:'pc1'},axis=1,inplace=True)

pc2 = pd.DataFrame(X_pca[:, 1])
pc2.rename({0:'pc2'},axis=1,inplace=True)

Xpca = pd.concat(
    [
        pc1,
        pc2
    ],
    axis=1
)

#print(Xpca)
Xpca.to_csv('Xpca.csv', header=True, index=False)


##### output raw data and principal components

XXpca = pd.concat(
    [
        X,
        Xpca
    ],
    axis=1
)

#print(XXpca)
XXpca.to_csv('XXpca.csv', header=True, index=False)
