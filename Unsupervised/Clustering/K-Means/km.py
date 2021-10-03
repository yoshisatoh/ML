#################### Unsupervised Learning: K-means Clustering ####################
#
#  (C) 2021, Yoshimasa (Yoshi) Satoh, CFA 
#
#  All rights reserved.
#
# Created:      2021/10/03
# Last Updated: 2021/10/03
#
# Github:
# https://github.com/yoshisatoh/ML/tree/main/Unsupervised/Clustering/K-Means/km.py
# https://github.com/yoshisatoh/ML/blob/main/Unsupervised/Clustering/K-Means/km.py
#
#
########## Input Data File(s)
#
#df_train.csv
#df_test.csv
#
#
########## Usage Instructions
#
#Run this script on Terminal of MacOS (or Command Prompt of Windows) as follows:
#
#python km.py df_train.csv df_test.csv 2 x y
#
#Generally,
#python km.py (training data file) (test data file) (k) (column name of data for x-axis) (column name of data for y-axis)
#
#
########## Output Data File(s)
#
#df_train_labels_out.csv
#df_test_labels_out.csv
#
#
########## References
#
#Understanding K-means Clustering in Machine Learning
#https://towardsdatascience.com/understanding-k-means-clustering-in-machine-learning-6a6e67336aa1
#
#
####################


'''
“The objective of K-means is simple:
To group similar data points together and discover underlying patterns.
To achieve this objective, K-means looks for a fixed number (k) of clusters in a dataset.
k, number of categories has to be known and provided by a human.
”
'''

########## import Python libraries

import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans




########## arguments

#print(sys.argv[0])
#km.py
#

df_train_name = str(sys.argv[1])

df_test_name  = str(sys.argv[2])

k      = int(sys.argv[3])

xname  = str(sys.argv[4])

yname  = str(sys.argv[5])



########## scatter plot color setting for centroids

# If you set k = 2, this can be used as it is. (raw data) + k = 3.
# If you use k >=3, then add 4th and more colors.
scatterc = ['b', 'g', 'r']




########## Figure 1: training data plotting

#####generate df.csv from scratch
#
'''
X  = -2 * np.random.rand(100,2)
X1 = 1 + 2 * np.random.rand(50,2)
X[50:100, :] = X1

df_train = pd.DataFrame(X, columns=['x', 'y'], index=None)
print(df_train.columns)
df_train = df_train.rename(index={0:'x', 1:'y'})

df_train.to_csv('df_train.csv', header=True, index=False)
'''
#####


df_train = pd.read_csv(df_train_name, header=0)


plt.figure(1, figsize=(8, 8))

plt.scatter(df_train[xname], df_train[yname], s = 50, c = scatterc[0], marker='o')

plt.title('Training Data')
plt.xlabel(xname)
plt.ylabel(yname)

plt.savefig('Fig_1_Training_Data.png')

plt.show()
plt.close()




########## K-Means Clustering

Kmean = KMeans(n_clusters = k)

#print(df_train.values)
#print(type(df_train.values))
#<class 'numpy.ndarray'>

#print(df_train[[xname, yname]].values)
#print(type(df_train[[xname, yname]].values))
#<class 'numpy.ndarray'>


Kmean.fit(df_train[[xname, yname]].values)




########## Figure 2: Training Data and Centroids (the center of the clusters)

print(Kmean.cluster_centers_)
'''
[[-0.91226973 -0.94803347]
 [ 2.01194855  2.01694867]]
'''
'''
First centroid:
[-0.91226973 -0.94803347] = [x-axis, y-axis]

Second centroid:
 [ 2.01194855  2.01694867] = [x-axis, y-axis]
'''

#print(type(Kmean.cluster_centers_))
#<class 'numpy.ndarray'>


plt.figure(2, figsize=(8, 8))

plt.scatter(df_train[xname], df_train[yname], s = 50, c = scatterc[0], label='Training', marker='o')

plt.title('K-means Clustering (Training Data)')
plt.xlabel(xname)
plt.ylabel(yname)


for i in range(k):
    #
    #print(i)
    #print(str(i+1) + ' st/nd/th centroid --- ' + 'x: ' + str(Kmean.cluster_centers_[i, 0]) + ', y: ' + str(Kmean.cluster_centers_[i, 1]))
    plt.scatter(Kmean.cluster_centers_[i, 0], Kmean.cluster_centers_[i, 1], s=100, c=scatterc[i+1], marker='s', label=i+1)

plt.legend()

plt.savefig('Fig_2_Training_Data_and_Centroids.png')

plt.show()
plt.close()




########## Output Training Data

#print(Kmean.labels_ + 1)

df_train_labels_ = pd.DataFrame(Kmean.labels_ + 1, columns=['labels_'], index=None)
#df_train_labels_.to_csv('df_train_labels_.csv', header=True, index=False)

df_train_labels_out = pd.concat([df_train[xname], df_train[yname], df_train_labels_], axis=1)
df_train_labels_out.to_csv('df_train_labels_out.csv', header=True, index=False)




########## Testing

df_test = pd.read_csv(df_test_name, header=0)

'''
for l in range(len(np.array(df_test)[:, 0])):
    print(df_test[xname][l])
    print(df_test[yname][l])
    print(Kmean.predict(np.array(df_test))[l] + 1)
'''


##### Training Data

plt.figure(3, figsize=(8, 8))

plt.scatter(df_train[xname], df_train[yname], s = 50, c = scatterc[0], marker='o', label='Training')

##### Training Data
for m in range(k):
    #
    #print(m)
    #print(str(m+1) + ' st/nd/th centroid --- ' + 'x: ' + str(Kmean.cluster_centers_[m, 0]) + ', y: ' + str(Kmean.cluster_centers_[m, 1]))
    plt.scatter(Kmean.cluster_centers_[m, 0], Kmean.cluster_centers_[m, 1], s=100, c=scatterc[m+1], marker='s', label=('Training: '+ str(m+1)))


##### Test Data
for n in range(len(np.array(df_test)[:, 0])):
    #
    plt.scatter(df_test[xname][n], df_test[yname][n], s=100, c=scatterc[Kmean.predict(np.array(df_test))[n] + 1], marker='x', label=('Test:       '+ str(Kmean.predict(np.array(df_test))[n] + 1))) 


##### Graph

plt.title('K-means Clustering (Training Data and Test Data)')
plt.xlabel(xname)
plt.ylabel(yname)

plt.legend()

plt.savefig('Fig_3_Training_Data_and_Centroids_plus_Prediction_by_Test_Data.png')

plt.show()
plt.close()


##### Output Test Data

#print(Kmean.predict(np.array(df_test)) + 1)

df_test_labels_ = pd.DataFrame(Kmean.predict(np.array(df_test)) + 1, columns=['labels_'], index=None)
#df_test_labels_.to_csv('df_test_labels_.csv', header=True, index=False)

df_test_labels_out = pd.concat([df_test[xname], df_test[yname], df_test_labels_], axis=1)
df_test_labels_out.to_csv('df_test_labels_out.csv', header=True, index=False)
