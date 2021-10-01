#################### KNN (K nearest neighbors) for Multi-class Classification with Non-linear Boundaries ####################
#
#  (C) 2021, Yoshimasa (Yoshi) Satoh, CFA 
#
#  All rights reserved.
#
# Created:      2021/10/01
# Last Updated: 2021/10/01
#
# Github:
# https://github.com/yoshisatoh/ML/tree/main/Supervised/Classification/KNN/non-linear/multi_class/knnnlmc.py
# https://github.com/yoshisatoh/ML/blob/main/Supervised/Classification/KNN/non-linear/multi_class/knnnlmc.py
#
#
########## Input Data File(s)
#
#df_yX.csv
#
#
########## Usage Instructions
#
# Run this script on Terminal of MacOS as follows:
#
#python knnnlmc.py df_yX.csv target sepal_length sepal_width 1
#python knnnlmc.py df_yX.csv target sepal_length sepal_width 3
#
# Generally,
#python knnnlmc.py (df_yX: data file with y, x1, and x2) (y: category values for classification) (x1: explanatory variable 1) (x2: explanatory variable 2) (n_neighbors)
#
# x1 and x2 will be used to draw a graph with non-linear boundary lines for classification of y.
#
#
########## References
#https://scipy-lectures.org/packages/scikit-learn/index.html
#https://scipy-lectures.org/packages/scikit-learn/auto_examples/plot_iris_knn.html
#https://archive.ics.uci.edu/ml/datasets/iris
#https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
#
#
####################




########## import Python libraries

import sys
import numpy as np
from matplotlib import pyplot as plt
from sklearn import neighbors, datasets
from sklearn import metrics
from matplotlib.colors import ListedColormap
import pandas as pd




########## arguments
for i in range(len(sys.argv)):
    print(str(sys.argv[i]))

#sys.argv[0]    #knnnlmc.py
dfname = str(sys.argv[1])    #e.g., df_yX.csv
yname  = str(sys.argv[2])    #e.g., target
x1name = str(sys.argv[3])    #e.g., sepal_length
x2name = str(sys.argv[4])    #e.g., sepal_width
nn     = int(sys.argv[5])    #n_neighbors: e.g., 1, 3, or any other integers




########## Create color maps for a classification problem

##### If you need more than 3 classes, then add 4+ th colors for 4+ th classes.
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold  = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])




########## Load input dataset


'''
iris = datasets.load_iris()

X = iris.data[:, :2]  # we only take the first two features. We could avoid this ugly slicing by using a two-dim dataset
y = iris.target

print(X)
print(type(X))
np.savetxt('X.csv', X, delimiter=',', fmt='%.8f')

print(y)
print(type(y))
np.savetxt('y.csv', y, delimiter=',', fmt='%i')

df_X = pd.DataFrame(X)
df_X.rename(columns={0: "sepal_length", 1: "sepal_width"}, inplace=True)
print(df_X)
#
df_y = pd.DataFrame(y)
df_y.rename(columns={0: "target"}, inplace=True)
print(df_y)

print(df_y.join(df_X))
df_yX = df_y.join(df_X)
df_yX.to_csv('df_yX.csv', index=False)
'''


#df_yX = pd.read_csv('df_yX.csv', sep=',', index_col=None, header=0)
df_yX = pd.read_csv(dfname, sep=',', index_col=None, header=0)
#print(df_yX)

#print(df_yX['target'])
#print(df_yX[['sepal_length', 'sepal_width']])

#y = df_yX['target'].to_numpy()
y = df_yX[yname].to_numpy()

#X = df_yX[['sepal_length', 'sepal_width']].to_numpy()
X = df_yX[[x1name, x2name]].to_numpy()




########## KNN (fitting by training data)

#knn = neighbors.KNeighborsClassifier(n_neighbors=1)
#knn = neighbors.KNeighborsClassifier(n_neighbors=3)
knn = neighbors.KNeighborsClassifier(n_neighbors=nn)

#knn.fit(X, y)
clf = knn.fit(X, y)

# set max and min of horizontal axis (x-axis) and vertical axis (y-axis)
#
x_min, x_max = X[:, 0].min() - .1, X[:, 0].max() + .1
y_min, y_max = X[:, 1].min() - .1, X[:, 1].max() + .1
#
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                     np.linspace(y_min, y_max, 100))


Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)

plt.figure(1)

plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

# Plot also the training points
for i in range(len(df_yX[yname].unique())):
    print(i)
    #plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold, label=i)
    #plt.scatter(df_yX[:, x1name][(df_yX[:, yname] == i)], df_yX[:, x2name][(df_yX[:, yname] == i)], c=y, cmap=cmap_bold, label=i)
    #plt.scatter(X[:, 0][(y[:] == i)], X[:, 1][(y[:] == i)], c=y[:][(y[:] == i)], cmap=cmap_bold, label=i)
#
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold)
#
#print(X[:, 0])
#print(X[:, 0][(y[:] == 0)])

plt.xlabel(x1name)
plt.ylabel(x2name)
plt.axis('tight')

#plt.legend()
#plt.legend(['red', 'green', 'blue'])
#plt.legend(loc='best')

plt.savefig('Fig_KNN_' + str(nn) + '.png')

plt.show()
plt.close()




########## KNN (evaluation)

#clf = neighbors.KNeighborsClassifier().fit(X, y)

y_pred = clf.predict(X)

##### F1 score
print('%s: %s' %
          (neighbors.KNeighborsClassifier.__name__, metrics.f1_score(y, y_pred, average="macro")))
'''
Compute the F1 score, also known as balanced F-score or F-measure.
The F1 score can be interpreted as a weighted average of the precision and recall, where an F1 score reaches its best value at 1 and worst score at 0. The relative contribution of precision and recall to the F1 score are equal. The formula for the F1 score is:
F1 = 2 * (precision * recall) / (precision + recall)
'''


##### confusion_matrix
print(metrics.confusion_matrix(y, y_pred))


##### classification_report
print(metrics.classification_report(y, y_pred))
#
# for instance, when nn = 1
'''
[[50  0  0]
 [ 0 45  5]
 [ 0  6 44]]
'''
'''
predicted
[[50  0  0]
 [ 0 45  5]
 [ 0  6 44]]
'''