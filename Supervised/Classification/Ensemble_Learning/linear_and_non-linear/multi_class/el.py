#################### ML (Supervised Learning) multi-class Classification: Ensemble Learning by multiple models ####################
#
#  (C) 2021, Yoshimasa (Yoshi) Satoh, CFA 
#
#  All rights reserved.
#
# Created:      2021/10/10
# Last Updated: 2021/10/10
#
# Github:
# https://github.com/yoshisatoh/ML/tree/main/Supervised/Ensemble_Learning/el.py
#
#
########## The ultimate purpose of this script
#
#Create 6 machine learning models, pick the best and build confidence that the accuracy is reliable.
#
#
########## Input Data File(s)
#
#df.csv
#This file is originally from https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv. 
#The following columns were added to this file iris.csv
#sepal-length,sepal-width,petal-length,petal-width,class
#and then iris.csv was saved as df.csv
#
#
########## Usage Instructions
#
#Run this script on Terminal of MacOS (or Command Prompt of Windows) as follows:
#python el.py df.csv class r 0.20 SVM
#
#Generally,
#python el.py (arg_df_file_name) (arg_class_name) (arg_data_type: r:raw data, s:standadized data, n: normalized data) (arg_test_size) (arg_model)
#
#
########## Output Data File(s)
#
#df2.csv    #pre-processed dataset including the class column
#
#df2_X_train.csv
#df2_y_train.csv
#df2_X_valid.csv
#df2_y_valid.csv
#df2_X_valid_pred.csv
#
#df3_train.csv
#df3_valid.csv
#df3_valid_pred.csv
#
#
########## References
#
#Your First Machine Learning Project in Python Step-By-Step
#https://machinelearningmastery.com/machine-learning-in-python-step-by-step/
#
#
#
####################




########## How a small end-to-end machine learning project is structured
'''
A number of well known steps for a small end-to-end machine learning project:

1. Define a problem to be solved.
2. Prepare (alternative/big) data.
3. Evaluate machine learning algorithms.
4. Improve the outcome by reviewing and updating 2. data and 3. algorithms; re-defining a problem as in 1 might be needed.
5. Present final results.

This is a good mixture of simple linear (LR and LDA), nonlinear (KNN, CART, NB and SVM) algorithms.
'''




########## 1. Installing and importing Python libraries

import sys
print('Python(sys): {}'.format(sys.version))

import scipy

#import numpy
import numpy as np

#import matplotlib
#from matplotlib import pyplot
import matplotlib.pyplot as plt

#import pandas
import pandas as pd
#from pandas import read_csv
from pandas.plotting import scatter_matrix

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

import seaborn as sns




##### If you haven't installed any of those, you'd see an issue. If that's the case, install the package as follows.
#
#For instance, if you haven't installed scipy, then try
#
#pip install scipy --upgrade
#pip install scipy --U
#python -m pip install scipy
#python -m pip install scipy==1.7.1
#
#If any of the above does not work in your environment, then try:
#pip install --upgrade scipy --trusted-host pypi.org --trusted-host files.pythonhosted.org




########## arguments

#sys.argv[0]    #el.py
arg_df_file_name = str(sys.argv[1])    # e.g., df.csv
arg_class_name   = str(sys.argv[2])    # e.g., class
arg_data_type    = str(sys.argv[3])    # r:raw data, s:standadized data, n: normalized data
arg_test_size    = float(sys.argv[4])    # e.g., 0.20 (if arg_test_size is 0.20, then training dataset is 0.80 = 1 - 0.20 of the entire dataset df/df2)
arg_model        = str(sys.argv[5])    # e.g., LR, LDA, KNN, CART, NB, or SVM - a specified and trained model to test X_valid and y_valid datasets 


for i in range(len(sys.argv)):
    print(str(sys.argv[i]))




########## pre-defined parameters

prm_random_state = 1    # to fix the data splitting result, random_state is an arbitrary integer, 1 in this case

prm_n_splits = 10 
# This is to use stratified 10-fold cross validation to estimate accuracy of multiple models in the section 5.3 Build Models.
# This will split our dataset into prm_n_splits (e.g., 10) parts, train on (prm_n_splits-1) (e.g., 9) and test on 1 and repeat for all combinations of train-test splits in X_train and y_train datasets.




########## 2.1 Loading the dataset.

df = pd.read_csv(arg_df_file_name,  sep=',', index_col=None, header=0)

print(df.head())
'''
   sepal-length  sepal-width  petal-length  petal-width        class
0           5.1          3.5           1.4          0.2  Iris-setosa
1           4.9          3.0           1.4          0.2  Iris-setosa
2           4.7          3.2           1.3          0.2  Iris-setosa
3           4.6          3.1           1.5          0.2  Iris-setosa
4           5.0          3.6           1.4          0.2  Iris-setosa
'''

print(df.columns)
'''
Index(['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class'], dtype='object')
'''



########## 2.2. Dataset pre-processing

df2 = df.drop(arg_class_name, axis=1)    #deleting one-column with the name of class, arg_class_name

if (arg_data_type) == 'r':
    print('', end='')    #(no processing as arg_data_type = 'r', raw data)
    #
elif (arg_data_type) == 's':
    df2 = (df2-df2.mean())/df2.std()    #processing as arg_data_type = 's', standardized data with the mean value = 0 and standard deviation = 1 for each non-class column)
    #
elif (arg_data_type) == 'n':
    df2 = (df2-df2.min())/(df2.max()-df2.min())    #processing as arg_data_type = 'n', normalized data with the min value = 0 and max value = 1 for each non-class column)
    #
else:
    print('Error: enter r, s, or n. r:raw data, s:standadized data, and n: normalized data')
    exit()
    #

df2[arg_class_name] = df[arg_class_name]    # df[arg_class_name], one-column class data frame, is added back to df2
print(df2)

df2.to_csv('df2.csv', sep=',', index=False, header=True)




########## 3. Summarizing the dataset.
'''
3.1 Dimensions of the dataset.
3.2 Peek at the data itself.
3.3 Statistical summary of all attributes.
3.4 Breakdown of the data by the class variable.
'''
##### 3.1 Dimensions of the dataset.

print(df2.shape)    #(150, 5)    (rows excluding header, columns)


##### 3.2 Peek at the data itself.
print(df2.head(10))


#3.3 Statistical summary of all attributes.
print(df2.describe())


#3.4 Breakdown of the data by the class variable.
print(df2.groupby(arg_class_name).size())




########## 4. Visualizing the dataset.

##### 4.1. Univariate Plots

#plt.figure(1)
#
df2.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
#
plt.savefig('Fig_4.1.1_Univariate_Plots.png')
#
#pyplot.show()
plt.show()
#
plt.close()


#plt.figure(2)
#
df2.hist()
#
plt.savefig('Fig_4.1.2_Univariate_Plots.png')
#
#pyplot.show()
plt.show()
#
plt.close()




##### 4.2 Multivariate Plots

#plt.figure(3)
#
scatter_matrix(df2)
#
plt.savefig('Fig_4.2_Multivariate_Plots.png')
#
#pyplot.show()
plt.show()
#
plt.close()




########## 5. Evaluating some algorithms.

#We will split the loaded dataset into two,
# 80% of which we will use to train, evaluate and select among our models, and
# 20% that we will hold back as a validation dataset.

#array = df2.values
#print(array.shape)
#
#X = array[:,0:4]
#y = array[:,4]

X = df2.drop(arg_class_name, axis=1)
y = df2[arg_class_name]

df2_X_train, df2_X_valid, df2_y_train, df2_y_valid = train_test_split(X, y, test_size=arg_test_size, random_state=prm_random_state)

print(df2_X_train.shape)    #(120, 4)
print(df2_X_valid.shape)    #(30, 4)
print(df2_y_train.shape)    #(120,)
print(df2_y_valid.shape)    #(30,)

df2_X_train.to_csv('df2_X_train.csv', sep=',', index=False, header=True)
df2_X_valid.to_csv('df2_X_valid.csv', sep=',', index=False, header=True)
df2_y_train.to_csv('df2_y_train.csv', sep=',', index=False, header=True)
df2_y_valid.to_csv('df2_y_valid.csv', sep=',', index=False, header=True)




##### 5.3 Build Models

'''
To test 6 different algorithms here:

LR:  Logistic Regression
LDA: Linear Discriminant Analysis
KNN: K-Nearest Neighbors
CART:Classification and Regression Trees
NB:  Gaussian Naive Bayes
SVM: Support Vector Machines
'''
models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))

# evaluate each model in turn
results = []
names = []
for name, model in models:
	kfold = StratifiedKFold(n_splits=prm_n_splits, random_state=prm_random_state, shuffle=True)
	cv_results = cross_val_score(model, df2_X_train, df2_y_train, cv=kfold, scoring='accuracy')
	results.append(cv_results)
	names.append(name)
	print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))

'''
LR: 0.941667 (0.065085)
LDA: 0.975000 (0.038188)
KNN: 0.958333 (0.041667)
CART: 0.933333 (0.050000)
NB: 0.950000 (0.055277)
SVM: 0.983333 (0.033333)
'''


##### 5.4 Select Best Model

plt.boxplot(results, labels=names)

plt.title('Algorithm Comparison')

plt.savefig('Fig_5.4_Select_Best_Model.png')

plt.show()

plt.close()




########## 6. Making some predictions.

##### 6.1 Make Predictions

if arg_model == 'LR':
    model = LogisticRegression(solver='liblinear', multi_class='ovr')
    #
elif arg_model == 'LDA':
    model = LinearDiscriminantAnalysis()
    #
elif arg_model == 'KNN':
    model = KNeighborsClassifier()
    #
elif arg_model == 'CART':
    model = DecisionTreeClassifier()
    #
elif arg_model == 'NB':
    model = GaussianNB()
    #
elif arg_model == 'SVM':
    model = SVC(gamma='auto')
    #
else:
    print('Error: set LR, LDA, KNN, CART, NB, or SVM')
    exit()


model.fit(df2_X_train, df2_y_train)

X_valid_pred = model.predict(df2_X_valid)

#print(np.unique(X_valid_pred))    #['Iris-setosa' 'Iris-versicolor' 'Iris-virginica']
#print(type(np.unique(X_valid_pred)))    #<class 'numpy.ndarray'>
lst_X_valid_pred = list(np.unique(X_valid_pred))


#X_valid_pred_df = pd.DataFrame(X_valid_pred, columns = ['class'])
df2_X_valid_pred = pd.DataFrame(X_valid_pred, columns = [arg_class_name + '_pred'])
df2_X_valid_pred.to_csv('df2_X_valid_pred.csv', sep=',', index=False, header=True)



##### 6.2 Evaluate Predictions

###Accuracy Score
print(accuracy_score(df2_y_valid, X_valid_pred))    #e.g., 0.9666666666666667


###Confusion Matrix
#
print(confusion_matrix(df2_y_valid, X_valid_pred))
cm = confusion_matrix(df2_y_valid, X_valid_pred)
#
ax = plt.subplot()
sns.heatmap(cm, annot=True, fmt='g', ax=ax);  #annot=True to annotate cells, ftm='g' to disable scientific notation
#
# labels, title and ticks
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
ax.set_title('Confusion Matrix'); 
#ax.xaxis.set_ticklabels(['1', '2', '3']); ax.yaxis.set_ticklabels(['1', '2', '3']);
ax.xaxis.set_ticklabels(lst_X_valid_pred); ax.yaxis.set_ticklabels(lst_X_valid_pred);
#
plt.savefig('Fig_6.2_Evaluate_Predictions.png')
#
plt.show()
#
plt.close()


###Classification Report
print(classification_report(df2_y_valid, X_valid_pred))




########## 7. Mergining all the dataset and saving them


df3_train      = pd.concat([df2_X_train, df2_y_train], axis='columns')
df3_train.to_csv('df3_train.csv', sep=',', index=False, header=True)

df3_valid      = pd.concat([df2_X_valid, df2_y_valid], axis='columns')
df3_valid.to_csv('df3_valid.csv', sep=',', index=False, header=True)


df3_valid        = df3_valid.reset_index(drop=True)
df2_X_valid_pred = df2_X_valid_pred.reset_index(drop=True)
#
df3_valid_pred = pd.concat([df3_valid, df2_X_valid_pred], axis='columns')
df3_valid_pred.to_csv('df3_valid_pred.csv', sep=',', index=False, header=True)
