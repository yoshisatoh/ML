#################### Multinomial Logistic Regression with L2 (or L1) Regularization of Standardized Training Data (plus Test Data) in Python ####################
# Note: Logistic Regression is actually a "classification" algorithm used to assign observations to a discrete set of classes.
# 
#
#  (C) 2021, Yoshimasa (Yoshi) Satoh, CFA 
#
#  All rights reserved.
#
# Created:      2021/09/25
# Last Updated: 2021/09/25
#
# Github:
# https://github.com/yoshisatoh/ML/tree/main/Supervised/Classification/Logistic_Regression/mlgstcreg.py
# https://github.com/yoshisatoh/ML/blob/main/Supervised/Classification/Logistic_Regression/mlgstcreg.py
#
#
########## Input Data Files
#
#y.csv
#X.csv
#
#
########## Usage Instructions
#
#Run this script on Terminal of MacOS as follows:
#python3 mlgstcreg.py l2 1
#python3 mlgstcreg.py (penalty/regularization of LogisticRegression()) (parameter c, 1 by default)
#
#Reference(s)
#http://ailaby.com/logistic_reg/
#
#
####################




########## Overview
'''
We have to avoid overfitting, i.e., letting a model learn outliers and noises of training data too well.
Reasons of overfitting are
(1) variables are too many,
(2) parameters are too big/impactful,  and/or
(3) data is not enough.

We can use L1 regularization (e.g., linear Lasso Regression) to eliminate redundant / unnecessary parameters as in (1).
Also, L2 regularization (e.g., linear Ridge Regression) is to avoid (2).
'''




########## import
'''
from sklearn import datasets
import matplotlib.pyplot as plt
import seaborn as sns
'''

#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sys
import math




########## arguments
pnlty = str(sys.argv[1])    #l2 or l1
c = int(sys.argv[2])    #C = 1 by default. The larger the C, the weaker regularization is going to be.




########## Multinomial Logistic Regression with Regularitation

''' 
##### generate dataset with TWO classes as the mixture of training data and test data
np.random.seed(seed=0)

X_0 = np.random.multivariate_normal( [2,2],  [[2,0],[0,2]],  50 )

y_0 = np.zeros(len(X_0))

X_1 = np.random.multivariate_normal( [6,7],  [[3,0],[0,3]],  50 )
y_1 = np.ones(len(X_1))
 
X = np.vstack((X_0, X_1))
#print(X)
#print(type(X))
#<class 'numpy.ndarray'>

y = np.append(y_0, y_1)
#print(y)
#print(type(y))
#<class 'numpy.ndarray'>


##### save dataset
pd.DataFrame(data=X).to_csv("X.csv", header=False, index=False)
pd.DataFrame(data=y).to_csv("y.csv", header=False, index=False)
'''


##### load raw dataset
X = pd.read_csv('X.csv', header=None).values
y = pd.read_csv('y.csv', header=None).values.ravel()

#print(type(X))
#<class 'numpy.ndarray'>

#print(type(y))
#<class 'numpy.ndarray'> 


##### plot raw data
plt.scatter(X[y==0, 0], X[y==0, 1], c='blue', marker='*', label='raw data 0')
plt.scatter(X[y==1, 0], X[y==1, 1], c='red', marker='*', label='raw data 1')

plt.legend(loc='upper left')
plt.title('Raw Data')
plt.xlabel('X1: Raw Data')
plt.ylabel('X2: Raw Data')

plt.savefig('Figure_1_Raw_Data.png')
plt.show()
plt.close()


##### splitting Training Data and Test Data
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3)
#random_state=3 is to fix results. You can change this number to, say, 1, 2, or any other integer you like.


##### Standardization of Training and Test Data (Average=0, SD=1)
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)


#####  max and min of X and y
xmax = max(
    max(X_train_std[y_train==0, 0]),
    max(X_train_std[y_train==1, 0]),
    max(X_test_std[y_test==0, 0]), 
    max(X_test_std[y_test==1, 0])
)
#
xmin = min(
    min(X_train_std[y_train==0, 0]),
    min(X_train_std[y_train==1, 0]),
    min(X_test_std[y_test==0, 0]), 
    min(X_test_std[y_test==1, 0])
)
#
#
ymax = max(
    max(X_train_std[y_train==0, 1]),
    max(X_train_std[y_train==1, 1]),
    max(X_test_std[y_test==0, 1]),
    max(X_test_std[y_test==1, 1])
)
#
ymin = min(
    min(X_train_std[y_train==0, 1]),
    min(X_train_std[y_train==1, 1]),
    min(X_test_std[y_test==0, 1]),
    min(X_test_std[y_test==1, 1])
)


##### plot trainging and test Data
plt.xlim([math.floor(xmin), math.ceil(xmax)])
plt.ylim([math.floor(ymin), math.ceil(ymax)])

plt.scatter(X_train_std[y_train==0, 0], X_train_std[y_train==0, 1], c='blue', marker='x', label='train 0')
plt.scatter(X_train_std[y_train==1, 0], X_train_std[y_train==1, 1], c='red', marker='x', label='train 1')
plt.scatter(X_test_std[y_test==0, 0], X_test_std[y_test==0, 1], c='blue', marker='o', s=60, label='test 0')
plt.scatter(X_test_std[y_test==1, 0], X_test_std[y_test==1, 1], c='red', marker='o', s=60, label='test 1')
 
plt.legend(loc='upper left')
plt.title('(Standardized) Training Data and Test Data')
plt.xlabel('X1: (Standardized) Training Data and Test Data')
plt.ylabel('X2: (Standardized) Training Data and Test Data')

plt.savefig('Figure_2_Standardized_Traing_Data_and_Test_Data.png')
plt.show()
plt.close()



 
########## Logistic Regression built by Standardized Training Data

#lr = LogisticRegression()
lr = LogisticRegression(C=c, penalty=pnlty)

lr.fit(X_train_std, y_train)


#check predict and score
#print(lr.predict(X_test_std))
#[1. 1. 0. 1. 1. 0. 1. 1. 0. 0. 1. 0. 1. 1. 0. 1. 1. 0. 1. 0.]
#
#print(y_test)
#[1. 1. 0. 1. 1. 1. 1. 0. 0. 0. 1. 0. 1. 1. 0. 1. 1. 0. 1. 0.]
#
#print(lr.predict(X_test_std))
#[1. 1. 0. 1. 1. 0. 1. 1. 0. 0. 1. 0. 1. 1. 0. 1. 1. 0. 1. 0.]
#
#[T  T  T  T  T  F  T  F  T  T  T  T  T  T  T  T  T  T  T  T ]
#Number of T (  correct answers) = 18
#Number of F (incorrect answers) = 2 
#18/(18+2) = 0.9

#print(lr.score(X_test_std, y_test))
#0.9


'''
A result of Logistic Regression is weights w0, w1, and w2 of a boundary line below:
w0+w1x+w2y=0

w0 is stored in intercept_ while w1 and w2 are included in coef_.

'''

#print (lr.intercept_)
#[-0.09150732]
 
#print (lr.coef_)
#[[1.99471124 2.32334603]]
 
w_0 = lr.intercept_[0]
w_1 = lr.coef_[0,0]
w_2 = lr.coef_[0,1]


# a boundary line 
#   w_0 + (w_1 * x) + (w_2 * y) = 0
#   y = ((-w_1 * x) - w_0) / w_2
#
print("y = ((-w_1 * x) - w_0) / w_2")
print("y = (-w_1/w_2) * x - w_0/w_2")
print("y: x2")
print("x: x1")
print("w_0 = ", w_0)
print("w_1 = ", w_1)
print("w_2 = ", w_2)
print("(-w_1/w_2) = ", (-w_1/w_2))
print("(-w_0/w_2) = ", (-w_0/w_2))

'''
y = ((-w_1 * x) - w_0) / w_2
y = (-w_1/w_2) * x - w_0/w_2
y: x2
x: x1
w_0 =  -0.09150731939635004
w_1 =  1.9947112354879184
w_2 =  2.3233460327656656
(-w_1/w_2) =  -0.8585510756283915
(-w_0/w_2) =  0.03938600540162393
'''


# plotting a boundary line
#plt.plot([-2,2], map(lambda x: (-w_1 * x - w_0)/w_2, [-2,2]))
#plt.plot([-2,2], list(map(lambda x: (-w_1 * x - w_0)/w_2, [-2,2])))
plt.plot([math.floor(xmin) - 1, math.ceil(xmax) + 1], list(map(lambda x: (-w_1 * x - w_0)/w_2, [math.floor(xmin) - 1, math.ceil(xmax) + 1])))


# plotting Training Data and Test Data
plt.xlim([math.floor(xmin), math.ceil(xmax)])
plt.ylim([math.floor(ymin), math.ceil(ymax)])

plt.scatter(X_train_std[y_train==0, 0], X_train_std[y_train==0, 1], c='blue', marker='x', label='train 0')
plt.scatter(X_train_std[y_train==1, 0], X_train_std[y_train==1, 1], c='red', marker='x', label='train 1')
plt.scatter(X_test_std[y_test==0, 0], X_test_std[y_test==0, 1], c='blue', marker='o', s=60, label='test 0')
plt.scatter(X_test_std[y_test==1, 0], X_test_std[y_test==1, 1], c='red', marker='o', s=60, label='test 1')

plt.legend(loc='upper left')
plt.title('(Standardized) Training Data and Test Data plus a Boundary Line')
plt.xlabel('X1: (Standardized) Training Data and Test Data')
plt.ylabel('X2: (Standardized) Training Data and Test Data')

#plt.text(-2,-2, 'Boundary Line: x2 = (-w_1/w_2) * x1 - w_0/w_2 = ' + str(-w_1/w_2) + ' * x1 + ' + str(-w_0/w_2), size=10)
plt.text(math.floor(xmin) + (math.ceil(xmax) - math.floor(xmin)) * 0.05, math.floor(ymin) + (math.ceil(ymax) - math.floor(ymin)) * 0.25, 'Regularization: ' + str(pnlty), size=9)
plt.text(math.floor(xmin) + (math.ceil(xmax) - math.floor(xmin)) * 0.05, math.floor(ymin) + (math.ceil(ymax) - math.floor(ymin)) * 0.20, 'c = ' + str(c), size=9)
plt.text(math.floor(xmin) + (math.ceil(xmax) - math.floor(xmin)) * 0.05, math.floor(ymin) + (math.ceil(ymax) - math.floor(ymin)) * 0.15, 'Training score = ' + str(round(lr.score(X_train_std, y_train),3)*100) + '%', size=9)
plt.text(math.floor(xmin) + (math.ceil(xmax) - math.floor(xmin)) * 0.05, math.floor(ymin) + (math.ceil(ymax) - math.floor(ymin)) * 0.10, 'Test score = ' + str(round(lr.score(X_test_std, y_test),3)*100) + '%', size=9)
plt.text(math.floor(xmin) + (math.ceil(xmax) - math.floor(xmin)) * 0.05, math.floor(ymin) + (math.ceil(ymax) - math.floor(ymin)) * 0.05, 'Boundary Line: x2 = ' + str(-w_1/w_2) + ' * x1 + ' + str(-w_0/w_2), size=9)

plt.savefig('Figure_3_Standardized_Traing_Data_and_Test_Data_plus_Boundary_Line.png')
plt.show()
plt.close()


##### Confusion Matrix

cm = confusion_matrix(y_test, lr.predict(X_test_std))

print ("Confusion Matrix : \n", cm)
'''
Confusion Matrix :
Confusion Matrix : 
 [[ 7  1]
 [ 0 12]]
'''
'''
Out of 20 :
TruePostive + TrueNegative = 12 + 7
FalsePositive + FalseNegative = 1 + 0

 [[TrueNegative(TN)  FalsePositive(FP)]
 [ FalseNegative(FN) TruePositive(TP)]]


                            Predicted Labels:
                            Negative              Positive
Actual Results: Negative    TrueNegative(TN)     FalsePositive(FP)
                Positive    FalseNegative(FN)    TruePositive(TP)
'''
#use 1, 2 as there are two independent variables x1 and x2
class_names=[1,2]
#fig, ax = plt.subplots()
fig, ax = plt.subplots(figsize=(6,5))
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
#sns.heatmap(pd.DataFrame(cm), annot=True, cmap="Blues" ,fmt='g')
sns.heatmap(pd.DataFrame(cm), annot=True, cmap="coolwarm", fmt='g')
ax.xaxis.set_label_position("top")
#plt.tight_layout()
plt.tight_layout(pad=3.00)
plt.title('Confusion Matrix (Test Data)')
plt.xlabel('Predicted Labels')
plt.ylabel('Actual Results')
#print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
plt.savefig("Figure_4_Confusion_Matrix_Test_Data.png")
plt.show()
plt.close()


########## Logistic Regression built by Standardized Training Data (+ Regularization parameter C)

'''
The larger the C, the weaker regularization is going to be.

C is 1.0 by default. We try 1, 10, 100, 1000 here.
'''


fig, axs = plt.subplots(2, 2, sharex=True, sharey=True)

plt.xlim([math.floor(xmin), math.ceil(xmax)])
plt.ylim([math.floor(ymin), math.ceil(ymax)])

plt.subplots_adjust(wspace=0.1, hspace=0.6)
c_params = [1.0, 10.0, 100.0, 1000.0]

#print(c_params)
#[1.0, 10.0, 100.0, 1000.0]
#
#print(type(c_params))
#<class 'list'>

#print(enumerate(c_params))
#<enumerate object at 0x127bec500>

for i, c in enumerate(c_params):
    #print(i, c)
    #
    #lr = LogisticRegression(C=c)
    lr = LogisticRegression(C=c, penalty=pnlty)
    lr.fit(X_train_std, y_train)
    #
    w_0 = lr.intercept_[0]
    w_1 = lr.coef_[0,0]
    w_2 = lr.coef_[0,1]
    score = lr.score(X_test_std, y_test)
    #
    #print(i/2, i%2)
    #print(math.floor(i/2), i%2)
    #####axs[i/2, i%2].set_title('C=' + str(c))
    axs[math.floor(i/2), i%2].set_title('C=' + str(c))
    #
    #####axs[i/2, i%2].plot([-2,2], map(lambda x: (-w_1 * x - w_0)/w_2, [-2,2]))
    axs[math.floor(i/2), i%2].plot([math.floor(xmin)-1, math.ceil(xmax)+1], list(map(lambda x: (-w_1 * x - w_0)/w_2, [math.floor(xmin)-1, math.ceil(xmax)+1])))
    #
    #####axs[i/2, i%2].scatter(X_train_std[y_train==0, 0], X_train_std[y_train==0, 1], c='red', marker='x', label='train 0')
    #axs[math.floor(i/2), i%2].scatter(X_train_std[y_train==0, 0], X_train_std[y_train==0, 1], c='red', marker='x', label='train 0')
    axs[math.floor(i/2), i%2].scatter(X_train_std[y_train==0, 0], X_train_std[y_train==0, 1], c='blue', marker='x', label='train 0')
    #
    #####axs[i/2, i%2].scatter(X_train_std[y_train==1, 0], X_train_std[y_train==1, 1], c='blue', marker='x', label='train 1')
    #axs[math.floor(i/2), i%2].scatter(X_train_std[y_train==1, 0], X_train_std[y_train==1, 1], c='blue', marker='x', label='train 1')
    axs[math.floor(i/2), i%2].scatter(X_train_std[y_train==1, 0], X_train_std[y_train==1, 1], c='red', marker='x', label='train 1')
    #
    #####axs[i/2, i%2].scatter(X_test_std[y_test==0, 0], X_test_std[y_test==0, 1], c='red', marker='o', s=60, label='test 0')
    #axs[math.floor(i/2), i%2].scatter(X_test_std[y_test==0, 0], X_test_std[y_test==0, 1], c='red', marker='o', s=60, label='test 0')
    axs[math.floor(i/2), i%2].scatter(X_test_std[y_test==0, 0], X_test_std[y_test==0, 1], c='blue', marker='o', s=60, label='test 0')
    #
    #####axs[i/2, i%2].scatter(X_test_std[y_test==1, 0], X_test_std[y_test==1, 1], c='blue', marker='o', s=60, label='test 1')
    #axs[math.floor(i/2), i%2].scatter(X_test_std[y_test==1, 0], X_test_std[y_test==1, 1], c='blue', marker='o', s=60, label='test 1')
    axs[math.floor(i/2), i%2].scatter(X_test_std[y_test==1, 0], X_test_std[y_test==1, 1], c='red', marker='o', s=60, label='test 1')
    #
    #
    if (i < 2):
        #####axs[i/2, i%2].text(0,-2.7, 'score ' + str(round(score,3)*100) + '%', size=13)
        axs[math.floor(i/2), i%2].text(math.floor(xmin) + (math.ceil(xmax) - math.floor(xmin)) * 0.10, math.ceil(ymax) - (math.ceil(ymax) - math.floor(ymin)) * 0.10, 'score ' + str(round(score,3)*100) + '%', size=10)
    else:
        #####axs[i/2, i%2].text(0,-3.3, 'score ' + str(round(score,3)*100) + '%', size=13)
        axs[math.floor(i/2), i%2].text(math.floor(xmin) + (math.ceil(xmax) - math.floor(xmin)) * 0.10, math.ceil(ymax) - (math.ceil(ymax) - math.floor(ymin)) * 0.10, 'score ' + str(round(score,3)*100) + '%', size=10)


plt.savefig('Figure_5_Standardized_Traing_Data_and_Test_Data_plus_Boundary_Line_for_various_Cs.png')
plt.show()
plt.close()