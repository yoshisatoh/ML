#################### Support Vector Machines (SVMs) for Single-class Classification with a Linear Boundary (discriminative classification) ####################
#
#  (C) 2021, Yoshimasa (Yoshi) Satoh, CFA 
#
#  All rights reserved.
#
# Created:      2021/10/19
# Last Updated: 2021/10/19
#
# Github:
# https://github.com/yoshisatoh/ML/tree/main/Supervised/Classification/SVM/linear/binary_class/svmlbc.py
#
#
########## Input Data File(s)
#
#df_yX.csv
#
#
########## Usage Instructions
#
# Run this script on Terminal of MacOS (or Command Prompt of Windows) as follows:
#
#python svmlbc.py df_yX.csv y x0 x1 1E10
#
# Generally,
#python svmlbc.py (df_name: data file with y, x0, and x1) (y_name: column y) (x0_name: column x0) (x1_name: column x1) (c_param: For very large c_param, the margin of is hard, and points cannot lie in it.)
#
# x0 and x1 will be used to draw a graph with a linear boundary line for classification of y.
#
#
########## References
#
#https://jakevdp.github.io/PythonDataScienceHandbook/05.07-support-vector-machines.html
#
#
####################




########## import Python libraries

import sys

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from scipy import stats

# use seaborn plotting defaults
import seaborn as sns; sns.set()

##from sklearn.datasets.samples_generator import make_blobs
from sklearn.datasets import make_blobs

from sklearn.svm import SVC # "Support vector classifier"




########## arguments
for i in range(len(sys.argv)):
    print(str(sys.argv[i]))

#sys.argv[0]    #svmlbc.py

df_name = str(sys.argv[1])    #e.g., df_yX.csv

y_name  = str(sys.argv[2])    #e.g., y (0 or 1)
x0_name = str(sys.argv[3])    #e.g., x0
x1_name = str(sys.argv[4])    #e.g., x1

c_param = float(sys.argv[5])    #e.g., 1E10
'''
To handle a case when some data points with different labels overlap, the SVM implementation has a bit of a fudge-factor which "softens" the margin: that is, it allows some of the points to creep into the margin if that allows a better fit. The hardness of the margin is controlled by a tuning parameter, most often known as C. For very large C, the margin is hard, and points cannot lie in it. For smaller C, the margin is softer, and can grow to encompass some points.
'''

########## Load input dataset


# df_yX.csv data generation for the first time
'''
X, y = make_blobs(n_samples=50, centers=2,
                  random_state=0, cluster_std=0.60)

print(type(X))    #<class 'numpy.ndarray'>
print(type(y))    #<class 'numpy.ndarray'>
print(X)
print(y)

plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn');

df_X  = pd.DataFrame(X, columns = ['x0','x1'])
df_y  = pd.DataFrame(y, columns = ['y'])
df_yX = pd.merge(df_y, df_X, left_index=True, right_index=True)
df_yX.to_csv('df_yX.csv', index=False, header=True)
'''


df_yX = pd.read_csv(df_name, index_col=None, header=0)

ds_y  = df_yX[y_name]
ds_x0 = df_yX[x0_name]
ds_x1 = df_yX[x1_name]
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

plt.figure(1)

#print(df_yX[df_yX[y_name] == 0])
#print(df_yX[x0_name][df_yX[y_name] == 0])

plt.scatter(df_yX[x0_name][df_yX[y_name] == 0], df_yX[x1_name][df_yX[y_name] == 0], label='0', s=50, cmap='rainbow');
plt.scatter(df_yX[x0_name][df_yX[y_name] == 1], df_yX[x1_name][df_yX[y_name] == 1], label='1', s=50, cmap='rainbow');

plt.xlabel(ds_x0.name)
plt.ylabel(ds_x1.name)

plt.legend()

plt.savefig('Fig_1.png')
plt.show()

plt.close('all')




########## Fitting a support vector machine (SVM)

#model = SVC(kernel='linear', C=1E10)
model = SVC(kernel='linear', C=c_param)

model.fit(np_x0x1, np_y)




##### a quick convenience function that will plot SVM decision boundaries for us

def plot_svc_decision_function(model, ax=None, plot_support=True):
    """Plot the decision function for a 2D SVC"""
    if ax is None:
        ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    #
    # create grid to evaluate model
    x = np.linspace(xlim[0], xlim[1], 30)
    y = np.linspace(ylim[0], ylim[1], 30)
    Y, X = np.meshgrid(y, x)
    xy = np.vstack([X.ravel(), Y.ravel()]).T
    P = model.decision_function(xy).reshape(X.shape)
    #
    # plot decision boundary and margins
    ax.contour(X, Y, P, colors='k',
               levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])
    #
    # plot support vectors
    if plot_support:
        ax.scatter(model.support_vectors_[:, 0],
                   model.support_vectors_[:, 1],
                   s=300, linewidth=1, facecolors='none');
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)




########## Draw a graph #2

plt.figure(2)

plt.scatter(df_yX[x0_name][df_yX[y_name] == 0], df_yX[x1_name][df_yX[y_name] == 0], label='0', s=50, cmap='rainbow');
plt.scatter(df_yX[x0_name][df_yX[y_name] == 1], df_yX[x1_name][df_yX[y_name] == 1], label='1', s=50, cmap='rainbow');

#Show the dividing line that maximizes the margin between the two sets of points. 
#Notice that a few of the training points just touch the margin: they are indicated by the black circles in this figure.
#These points are the pivotal elements of this fit, and are known as the support vectors, and give the algorithm its name.
plot_svc_decision_function(model);

plt.xlabel(ds_x0.name)
plt.ylabel(ds_x1.name)

plt.legend()

plt.savefig('Fig_2.png')
plt.show()

plt.close('all')




########## Show positions of support vactors

print(model.support_vectors_)
'''
array([[ 0.44359863,  3.11530945],
       [ 2.33812285,  3.43116792],
       [ 2.06156753,  1.96918596]])
'''
'''
There are 3 rows; that means there are 3 points of the support vectors.
There are 2 columns; the first column is a position of x(horizontal)-axis while the second column is a position of y(vertical)-axis.

A key to this classifier's success is that for the fit, only the position of the support vectors matter; any points further from the margin which are on the correct side do not modify the fit! Technically, this is because these points do not contribute to the loss function used to fit the model, so their position and number do not matter so long as they do not cross the margin.
'''

