#################### Gaussian Naive Bayes (GNB) for Single-class Classification with a Non-linear Boundary (generative classification that uses a generative model to probabilistically determine labels for new points) ####################
#
#  (C) 2021, Yoshimasa (Yoshi) Satoh, CFA 
#
#  All rights reserved.
#
# Created:      2021/10/19
# Last Updated: 2021/10/19
#
# Github:
# https://github.com/yoshisatoh/ML/tree/main/Supervised/Classification/Naive_Bayes/Gaussian/non-linear/single_class/gnbnlsc.py
#
#
########## Input Data File(s)
#
#df_yX.csv       # training data with columns y, x0, and x1 
#df_Xtest.csv    # test data with columns x0 and x1 
#
#
########## Usage Instructions
#
# Run this script on Terminal of MacOS (or Command Prompt of Windows) as follows:
#
#python gnbnlsc.py df_yX.csv y x0 x1 df_Xtest.csv
#
# Generally,
#python gnbnlsc.py (df_name: training data file with y, x0, and x1) (y_name: column y with a value 0 or 1) (x0_name: column x0) (x1_name: column x1) (df_Xtest_name: test data file with x0 and x1)
#
# x0 and x1 will be used to draw a graph with a non-linear boundary line for classification of y.
#
#
########## Output Data File(s)
#
#df_yXtest.csv    # test data with columns y (predicted), x0, and x1 
#df_yXtestyprob.csv    # test data with columns y (predicted), x0, x1, probability (label=0), and probability (label=1)
#
#
########## References
#
#https://jakevdp.github.io/PythonDataScienceHandbook/05.05-naive-bayes.html
#
#
####################




########## import Python libraries

import sys

import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

from sklearn.naive_bayes import GaussianNB


########## arguments
for i in range(len(sys.argv)):
    print(str(sys.argv[i]))

#sys.argv[0]    #nbnlsc.py

df_name = str(sys.argv[1])    #e.g., df_yX.csv

y_name  = str(sys.argv[2])    #e.g., y (0 or 1)
x0_name = str(sys.argv[3])    #e.g., x0
x1_name = str(sys.argv[4])    #e.g., x1

df_Xtest_name = str(sys.argv[5])    #e.g., 'df_Xtest.csv'




########## Load input dataset


# df_yX.csv data generation for the first time
'''
from sklearn.datasets import make_blobs

X, y = make_blobs(100, 2, centers=2, random_state=2, cluster_std=1.5)

plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='RdBu');

print(type(X))    #<class 'numpy.ndarray'>
print(type(y))    #<class 'numpy.ndarray'>
print(X)
print(y)

df_X  = pd.DataFrame(X, columns = ['x0','x1'])
df_y  = pd.DataFrame(y, columns = ['y'])

df_yX = pd.merge(df_y, df_X, left_index=True, right_index=True)

df_yX.to_csv('df_yX.csv', index=False, header=True)

exit()
'''


df_yX = pd.read_csv(df_name, index_col=None, header=0)

ds_y  = df_yX[y_name]
ds_x0 = df_yX[x0_name]
ds_x1 = df_yX[x1_name]

df_x0x1 = df_yX[[x0_name, x1_name]]
'''
print(type(ds_y))     #<class 'pandas.core.series.Series'>
print(type(ds_x0))    #<class 'pandas.core.series.Series'>
print(type(ds_x1))    #<class 'pandas.core.series.Series'>

print(ds_y.name)      #y
print(ds_x0.name)     #x0
print(ds_x1.name)     #x1
'''

np_y    = df_yX[y_name].values
np_x0   = df_yX[x0_name].values
np_x1   = df_yX[x1_name].values
np_x0x1 = df_yX[[x0_name, x1_name]].values
'''
print(type(np_y))     #<class 'numpy.ndarray'>
print(type(np_x0))    #<class 'numpy.ndarray'>
print(type(np_x1))    #<class 'numpy.ndarray'>
print(type(np_x0x1))    #<class 'numpy.ndarray'>
'''




########## Draw a graph #1

plt.figure(num=1, figsize=(12, 8))

plt.scatter(df_yX[x0_name][df_yX[y_name] == 0], df_yX[x1_name][df_yX[y_name] == 0], label='0', s=50, cmap='rainbow');
plt.scatter(df_yX[x0_name][df_yX[y_name] == 1], df_yX[x1_name][df_yX[y_name] == 1], label='1', s=50, cmap='rainbow');

plt.xlabel(ds_x0.name)
plt.ylabel(ds_x1.name)

plt.title('Training Data')

plt.legend()

plt.savefig('Fig_1.png')
plt.show()

plt.close('all')




########## Draw a graph #2


'''
X, y = make_blobs(100, 2, centers=2, random_state=2, cluster_std=1.5)

print(type(X))    #<class 'numpy.ndarray'>
print(type(y))    #<class 'numpy.ndarray'>
print(X)    #<class 'numpy.ndarray'>
print(y)    #<class 'numpy.ndarray'>
'''


X = np_x0x1
y = np_y



fig, ax = plt.subplots()

ax.scatter(df_yX[x0_name][df_yX[y_name] == 0], df_yX[x1_name][df_yX[y_name] == 0], label='0', s=50, cmap='rainbow')
ax.scatter(df_yX[x0_name][df_yX[y_name] == 1], df_yX[x1_name][df_yX[y_name] == 1], label='1', s=50, cmap='rainbow')

ax.set_title('Gaussian Naive Bayes Model: Training Data', size=12)

xmax = max(df_yX[x0_name][df_yX[y_name] == 0].max(), df_yX[x0_name][df_yX[y_name] == 1].max())
xmin = min(df_yX[x0_name][df_yX[y_name] == 0].min(), df_yX[x0_name][df_yX[y_name] == 1].min())
ymax = max(df_yX[x1_name][df_yX[y_name] == 0].max(), df_yX[x1_name][df_yX[y_name] == 1].max())
ymin = min(df_yX[x1_name][df_yX[y_name] == 0].min(), df_yX[x1_name][df_yX[y_name] == 1].min())

xlim = (xmin, xmax)

ylim = (ymin, ymax)

xg = np.linspace(xlim[0], xlim[1], 60)
yg = np.linspace(ylim[0], ylim[1], 40)
xx, yy = np.meshgrid(xg, yg)
Xgrid = np.vstack([xx.ravel(), yy.ravel()]).T

for label, color in enumerate(['blue', 'red']):
    mask = (y == label)
    mu, std = X[mask].mean(0), X[mask].std(0)
    P = np.exp(-0.5 * (Xgrid - mu) ** 2 / std ** 2).prod(1)
    Pm = np.ma.masked_array(P, P < 0.03)
    ax.pcolorfast(xg, yg, Pm.reshape(xx.shape), alpha=0.5,
                  cmap=color.title() + 's')
    ax.contour(xx, yy, P.reshape(xx.shape),
               levels=[0.01, 0.1, 0.5, 0.9],
               colors=color, alpha=0.2)
    
ax.set(xlim=xlim, ylim=ylim)

ax.set_xlabel(ds_x0.name)    #'x0'

ax.set_ylabel(ds_x1.name)    #'x1'
 
fig.legend()

fig.savefig('Fig_2.png')




########## Fitting a Naive Bayes Model (NBM)

model = GaussianNB()
model.fit(X, y);




########## Random new data and prediction of the label

#Now let's generate some new data and predict the label:
'''
rng = np.random.RandomState(0)

Xnew = [-6, -14] + [14, 18] * rng.rand(2000, 2)

df_Xnew = pd.DataFrame(Xnew, columns = ['x0','x1'])
df_Xnew.to_csv('df_Xnew.csv', index=False, header=True)
'''
df_Xtest = pd.read_csv(df_Xtest_name, index_col=None, header=0)
np_Xtest = df_Xtest.to_numpy()


ytest = model.predict(np_Xtest)


df_ytest  = pd.DataFrame(ytest, columns = [y_name])

df_yXtest = pd.merge(df_ytest, df_Xtest, left_index=True, right_index=True)

df_yXtest.to_csv('df_yXtest.csv', index=False, header=True)




########## Draw a graph #3

plt.figure(num=3, figsize=(12, 8))

plt.scatter(df_yX[x0_name][df_yX[y_name] == 0], df_yX[x1_name][df_yX[y_name] == 0], label='0', s=50, cmap='rainbow')
plt.scatter(df_yX[x0_name][df_yX[y_name] == 1], df_yX[x1_name][df_yX[y_name] == 1], label='1', s=50, cmap='rainbow')

lim = plt.axis()

plt.scatter(df_yXtest[x0_name][df_yXtest[y_name] == 0], df_yXtest[x1_name][df_yXtest[y_name] == 0], label='0', s=5, cmap='rainbow')
plt.scatter(df_yXtest[x0_name][df_yXtest[y_name] == 1], df_yXtest[x1_name][df_yXtest[y_name] == 1], label='1', s=5, cmap='rainbow')

plt.axis(lim);

plt.title('Gaussian Naive Bayes Model: Training Data and Test Data')

plt.legend()

plt.savefig('Fig_3.png')

plt.show()

plt.close('all')




########## probabilistic classification

yprob = model.predict_proba(np_Xtest)
#print(yprob.round(2))

df_yprob  = pd.DataFrame(yprob, columns = [0, 1])    # columns - label: 0 or 1
#print(df_yprob.head())

df_yXtestyprob = pd.merge(df_yXtest, df_yprob, left_index=True, right_index=True)

df_yXtestyprob.to_csv('df_yXtestyprob.csv', index=False, header=True)



'''
The columns give the posterior probabilities of the first and second label, respectively. If you are looking for estimates of uncertainty in your classification, Bayesian approaches like this can be a useful approach.
'''