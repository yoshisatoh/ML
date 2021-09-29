#################### Logistic Regression for Multi-class Classification ####################
#
#  (C) 2021, Yoshimasa (Yoshi) Satoh, CFA 
#
#  All rights reserved.
#
# Created:      2021/09/29
# Last Updated: 2021/09/29
#
# Github:
# https://github.com/yoshisatoh/ML/tree/main/Supervised/Classification/Logistic_Regression/multi_class/lgstcrgmc.py
# https://github.com/yoshisatoh/ML/blob/main/Supervised/Classification/Logistic_Regression/multi_class/lgstcrgmc.py
#
#
########## Input Data Files
#
#iris.csv
#
# This data was originally created as follows:
#
# 1. Download the following data:
#https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data
#
# 2. Add the following column names to the first row
#"sepal_length","sepal_width","petal_length","petal_width","Species"
#
# 3. Save as iris.csv
#
#
########## Usage Instructions
#
# Run this script on Terminal of MacOS as follows:
#
#python lgstcrgmc.py iris.csv Species petal_length petal_width
#or
#python lgstcrgmc.py iris.csv Species sepal_length sepal_width
#
# Generally,
#python lgstcrgmc.py (dtf: data file) (categoryname: category name) (x-axis) (y-axis)
#
#
########## References
#https://teddykoker.com/2019/06/multi-class-classification-with-logistic-regression-in-python/
#https://github.com/teddykoker/blog/blob/master/_notebooks/2019-06-16-multi-class-classification-with-logistic-regression-in-python.ipynb
#https://archive.ics.uci.edu/ml/datasets/iris
#
#
####################


########## import

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import sys

plt.rcParams["figure.figsize"] = (5, 4)    # (w, h)
plt.rcParams["figure.dpi"] = 200

#print(plt.rcParams['image.cmap'])
#viridis

# training data and test data are always the same if you do not change
# the following seed(n)
np.random.seed(42)
#
# num_train (0.80) and training-test data ratio (training:test=0.80:0.20) as shown below




########## arguments

dtf          = str(sys.argv[1])
categoryname = str(sys.argv[2])    #'Species'
xn           = str(sys.argv[3])    #'petal_length'
yn           = str(sys.argv[4])    #'petal_width'




########## The Sigmoid Function
#
# Let’s say we want to classify our data into two categories: negative and positive.
#
# Unlike linear regression, where we want to predict a continuous value,
# we want our classifier to predict the probability that the data is positive (1), or negative (0).
# For this we will use the Sigmoid function:


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# If we plot the function, we will notice that as the input approaches +∞, the output approaches 1,
# and as the input approaches −∞, the output approaches 0.

x = np.linspace(-10, 10, 200)

plt.figure(1)

plt.plot(x, sigmoid(x))
plt.axvline(x=0, color='k', linestyle='--');
plt.title("Sigmoid");
plt.xlabel('x'); plt.ylabel('sigmoid(x)')
plt.savefig("Figure_1_Sigmoid_Function.png")
plt.show()

#exit()




########## Cost Function

# By passing the product of our inputs and parameters to the sigmoid function, g,
# we can form a prediction h of the probability of input x being classified as positive.
#
# hθ(x) = g(θT x)

# When we perform linear regression, we usually use Mean Squared Error as our cost function. This works well for regression.
# However, for classification we will use the Cross Entropy Loss function J.
# Logistic "regression" is actually for classification of discrete values, not regression of continuous values.

h = np.linspace(0, 1)[1:-1]

plt.figure(2)

for y in [0, 1]:
    plt.plot(h, -y * np.log(h) - (1 - y) * np.log(1 - h), label=f"y={y}")

plt.title("Cross Entropy Loss") 
#J: Cross Entropy Loss (=Cost) Functon
plt.xlabel('$h_ {\\theta}(x)$'); plt.ylabel('$J(\\theta)$')
plt.legend();
plt.savefig("Figure_2_Cross_Entropy_Loss_Function.png")
plt.show()

# We can see that a prediction matching the classification will have a cost of 0,
# but approach infinity as the prediction approaches the wrong classification.

#exit()




########## Gradient Function

# The following single Python function returns both our cost (function with respect to our parameters) and gradient:

def cost(theta, x, y):
    h = sigmoid(x @ theta)
    m = len(y)
    cost = 1 / m * np.sum(
        -y * np.log(h) - (1 - y) * np.log(1 - h)
    )
    grad = 1 / m * ((y - h) @ x)
    return cost, grad




########## Training

# Training function fit:
#
#max_iter=5000
#alpha=0.1
#
def fit(x, y, max_iter=5000, alpha=0.1):
    x = np.insert(x, 0, 1, axis=1)
    thetas = []
    classes = np.unique(y)
    costs = np.zeros(max_iter)
    #
    for c in classes:
        # one vs. rest binary classification
        binary_y = np.where(y == c, 1, 0)
        #
        theta = np.zeros(x.shape[1])
        #
        for epoch in range(max_iter):
            costs[epoch], grad = cost(theta, x, binary_y)
            theta += alpha * grad
            #
        thetas.append(theta)
    return thetas, classes, costs


# Prediction function that predicts a class label using the maximum hypothesis hθ(x):
#
def predict(classes, thetas, x):
    x = np.insert(x, 0, 1, axis=1)
    preds = [np.argmax(
        [sigmoid(xi @ theta) for theta in thetas]
    ) for xi in x]
    return [classes[p] for p in preds]




########## Data Set

# We will use the Iris Data Set, a commonly used dataset containing 3 species of iris plants.
# Each plant in the dataset has 4 attributes:
#    sepal length,
#    sepal width,
#    petal length, and
#    petal width.
# We will use our logistic regression model to predict flowers’ species using two specified attributes out of these all attributes.


#dtf = "iris.csv"
#
df = pd.read_csv(dtf, header=0)    #first row (0) is a header 


#print(df.head())
'''
   sepal_length  sepal_width  petal_length  petal_width      Species
0           5.1          3.5           1.4          0.2  Iris-setosa
1           4.9          3.0           1.4          0.2  Iris-setosa
2           4.7          3.2           1.3          0.2  Iris-setosa
3           4.6          3.1           1.5          0.2  Iris-setosa
4           5.0          3.6           1.4          0.2  Iris-setosa
'''
#exit()


#print(df[categoryname])
'''
0         Iris-setosa
1         Iris-setosa
2         Iris-setosa
3         Iris-setosa
4         Iris-setosa
            ...      
145    Iris-virginica
146    Iris-virginica
147    Iris-virginica
148    Iris-virginica
149    Iris-virginica
Name: Species, Length: 150, dtype: object
'''
#exit()


##### classification data frame dfclass

# Now we encode the species from string to integer
#
dfclass = pd.concat([df[categoryname], df[categoryname].astype('category').cat.codes], axis=1)
dfclass = pd.DataFrame(dfclass.drop_duplicates())
dfclass = dfclass.reset_index(drop=True)
#
#print(dfclass)
'''
           Species  0
0      Iris-setosa  0
1  Iris-versicolor  1
2   Iris-virginica  2
'''
#
#exit()
#
#
dfclass = dfclass.rename(columns={0: 'class'})
#
print(dfclass)
'''
           Species  class
0      Iris-setosa      0
1  Iris-versicolor      1
2   Iris-virginica      2
'''
#exit()
#
#print(dfclass.columns)
#Index(['Species', 'class'], dtype='object')
#
#exit()


df[categoryname] = df[categoryname].astype('category').cat.codes


#xn = 'petal_length'
xnn = df.columns.get_loc(xn)
#
#print(xnn)
#2
#
#
#yn = 'petal_width'
ynn = df.columns.get_loc(yn)
#
#print(ynn)
#3
#
#exit()


#print(df[categoryname])
'''
0      0
1      0
2      0
3      0
4      0
      ..
145    2
146    2
147    2
148    2
149    2
Name: Species, Length: 150, dtype: int8
'''
#exit()


# Shuffle the data, and split it into training and test data:
#
data = np.array(df)
np.random.shuffle(data)
#
num_train = int(.8 * len(data))  # 80/20 train/test split
#
x_train, y_train = data[:num_train, :-1], data[:num_train, -1]
x_test,  y_test  = data[num_train:, :-1], data[num_train:, -1]


# Save as csv files
np.savetxt('x_train.csv', x_train, delimiter=',', fmt='%.8f')
np.savetxt('y_train.csv', y_train, delimiter=',', fmt="%.0f")
np.savetxt( 'x_test.csv', x_test,  delimiter=',', fmt='%.8f')
np.savetxt( 'y_test.csv', y_test,  delimiter=',', fmt="%.0f")




##### Training Data

plt.figure(3)

#print(x_train)
#print(x_train[:, xnn])
#print(x_train[:, ynn])
#print(y_train)
#print(y_train[:])
#
#plt.scatter(x_train[:, xnn], x_train[:, ynn], c=y_train, alpha=0.5, s=10, marker='o')
#plt.plot(x_train[:, xnn], x_train[:, ynn], label=y_train)
#plt.scatter(x_train[:, xnn], x_train[:, ynn], c=y_train[:], alpha=0.5, s=10, marker='o', label=y_train[:])
#plt.scatter(x_train[:, xnn], x_train[:, ynn], c=y_train, alpha=0.5, s=10, marker='o', label=df[categoryname])
#plt.scatter(x_train[:, xnn], x_train[:, ynn], c=y_train, alpha=0.5, s=10, marker='o', label='test')
#
#
#print(len(x_train))
#print(len(y_train))
#
'''
for i in range(len(y_train)):
    print(y_train[i])
    plt.scatter(x_train[:, xnn], x_train[:, ynn], c=y_train, alpha=0.5, s=10, marker='o', label='0' if y_train[i] == 0 else '1')
'''
#
#
#print(len(np.unique(y_train)))
c = len(np.unique(y_train))
#
for i in range(c):
    #print(np.unique(y_train)[i])
    #print(y_train == i)
    #print(type(x_train[:, xnn]))
    #print(x_train[:, xnn][(y_train == i)])
    plt.scatter(x_train[:, xnn][(y_train == i)], x_train[:, ynn][(y_train == i)], alpha=0.5, s=10, marker='o', label=i)
#
#
###plt.scatter(x_train[:, xnn], x_train[:, ynn], c=y_train, alpha=0.5, s=10, marker='o', label=y_train)


plt.xlabel(xn); plt.ylabel(yn);
plt.title("Training Data")
#
#print(dfclass)
#plt.legend();
#plt.legend(labels=dfclass);
#plt.legend(labels=y_train[:], loc='best');
#plt.legend(labels=y_train, loc='best');
plt.legend(loc='best');
#plt.legend(labels=['0', '1', '2'], loc='best');
#plt.legend(labels=str(y_train), ncol=3);
#plt.legend();
#plt.legend(['0', '1', '2']);
#plt.legend(labels=dfclass[categoryname]);
#print(dfclass[categoryname])
#
plt.savefig("Figure_3_Training_Data.png")
plt.show()




##### cost improvement over each epoch

plt.figure(4)

thetas, classes, costs = fit(x_train[:, [xnn, ynn]], y_train)

plt.plot(costs)
plt.xlabel('Number of Epochs'); plt.ylabel('Cost');
plt.title("cost improvement over each epoch")
plt.savefig("Figure_4_cost_improvement_over_each_epoch.png")
plt.show()




########## linear boundaries generated by the parameters

#print(x_train)
'''
[[6.1 2.8 4.7 1.2]
...
 [5.2 2.7 3.9 1.4]]
'''
#
#print(x_train[:,2])
'''
[4.7 1.7 6.9 4.5 4.8 1.5 3.6 5.1 4.5 3.9 5.1 1.4 1.3 1.5 1.5 4.7 5.8 3.9
 4.5 5.6 1.6 4.9 1.6 5.6 6.4 5.2 5.8 5.9 1.4 1.6 1.  1.5 4.4 1.6 1.3 5.
 4.5 1.5 1.4 1.5 5.1 4.5 4.7 1.3 1.5 3.7 5.1 5.5 4.4 6.1 4.2 6.6 4.5 1.4
 6.7 4.1 1.4 1.3 1.9 3.5 4.9 1.9 1.6 1.7 4.2 1.5 4.2 6.7 1.4 4.3 5.  1.4
 4.8 5.1 4.  4.5 5.4 4.  1.7 3.3 5.3 1.4 1.2 3.8 5.  1.5 5.1 1.5 1.6 4.8
 3.  5.7 5.1 5.6 6.1 4.  1.4 1.1 5.  6.  1.5 1.4 1.3 4.9 5.6 1.4 5.5 6.
 1.3 4.7 4.6 4.8 4.7 5.3 1.6 5.4 4.2 5.2 3.5 3.9]
'''
#
#print(x_train[:, 3])
'''
[4.7 1.7 6.9 4.5 4.8 1.5 3.6 5.1 4.5 3.9 5.1 1.4 1.3 1.5 1.5 4.7 5.8 3.9
 4.5 5.6 1.6 4.9 1.6 5.6 6.4 5.2 5.8 5.9 1.4 1.6 1.  1.5 4.4 1.6 1.3 5.
 4.5 1.5 1.4 1.5 5.1 4.5 4.7 1.3 1.5 3.7 5.1 5.5 4.4 6.1 4.2 6.6 4.5 1.4
 6.7 4.1 1.4 1.3 1.9 3.5 4.9 1.9 1.6 1.7 4.2 1.5 4.2 6.7 1.4 4.3 5.  1.4
 4.8 5.1 4.  4.5 5.4 4.  1.7 3.3 5.3 1.4 1.2 3.8 5.  1.5 5.1 1.5 1.6 4.8
 3.  5.7 5.1 5.6 6.1 4.  1.4 1.1 5.  6.  1.5 1.4 1.3 4.9 5.6 1.4 5.5 6.
 1.3 4.7 4.6 4.8 4.7 5.3 1.6 5.4 4.2 5.2 3.5 3.9]
[1.2 0.3 2.3 1.5 1.4 0.4 1.3 2.3 1.5 1.2 2.  0.1 0.2 0.1 0.3 1.6 2.2 1.1
 1.3 2.2 0.2 1.8 0.4 2.1 2.  2.3 1.8 2.3 0.3 0.2 0.2 0.4 1.4 0.2 0.2 1.9
 1.5 0.2 0.2 0.1 1.9 1.6 1.5 0.4 0.2 1.  1.5 1.8 1.4 2.5 1.3 2.1 1.5 0.2
 2.  1.  0.2 0.3 0.4 1.  1.8 0.2 0.2 0.5 1.3 0.2 1.2 2.2 0.2 1.3 2.  0.2
 1.8 1.9 1.  1.5 2.3 1.3 0.4 1.  1.9 0.2 0.2 1.1 1.7 0.1 2.4 0.2 0.6 1.8
 1.1 2.3 1.6 1.4 2.3 1.3 0.2 0.1 1.5 1.8 0.2 0.3 0.2 1.5 2.4 0.3 2.1 2.5
 0.2 1.4 1.5 1.8 1.4 2.3 0.2 2.1 1.5 2.  1.  1.4]
'''


########## Figure: Training Data and Linear Boundaries

##### Training Data

plt.figure(5)

#print(len(np.unique(y_train)))
c = len(np.unique(y_train))
#
for i in range(c):
    #print(np.unique(y_train)[i])
    #print(y_train == i)
    #print(type(x_train[:, xnn]))
    #print(x_train[:, xnn][(y_train == i)])
    plt.scatter(x_train[:, xnn][(y_train == i)], x_train[:, ynn][(y_train == i)], alpha=0.5, s=10, marker='o', label=i)
#
###plt.scatter(x_train[:, xnn], x_train[:, ynn], c=y_train, alpha=0.5, s=10, marker='o')

plt.xlabel(xn); plt.ylabel(yn);

for theta in [thetas[0],thetas[2]]:
    j = np.array([x_train[:, xnn].min(), x_train[:, xnn].max()])
    k = -(j * theta[1] + theta[0]) / theta[2]
    plt.plot(j, k, color='k', linestyle="--")

#plt.legend()
#plt.legend(labels=y_train, loc='best');
plt.legend(loc='best');
plt.title("Training Data and Linear Boundaries")
plt.savefig("Figure_5_Training_Data_and_Linear_Boundaries.png")
plt.show()




########## Solving Simultaneous Linear Equations

def makeLinearEquation(x1, y1, x2, y2):
	line = {}
	if y1 == y2:
		# a line that is parallel to y-axis
		line["y"] = y1
	elif x1 == x2:
		# a line that is parallel to x-axis
		line["x"] = x1
	else:
		# y = mx + n
		line["m"] = (y1 - y2) / (x1 - x2)
		line["n"] = y1 - (line["m"] * x1)
	return line

#print(json.dumps(makeLinearEquation(2, 4, 3, 7), indent=4))
#a line that passes both (2, 4) and (3, 7)
#{
#    "m": 3.0,
#    "n": -2.0
#}

'''
For instance, to solve simultaneous linear equations below,
2x+y=4
x+3y=7
 
A = np.matrix([
[2, 1],
[1, 3]
])

Y = np.matrix([
[4],
[7]
])

np.linalg.inv(A)*Y    or    np.linalg.solve(A,Y)
matrix([[1.],
        [2.]])

This means x=1.0, y=2.0.

'''




########## Figure: Training Data and Linear Boundaries (+Test Data)

plt.figure(6)

##### Training Data
#
#change cmap
plt.rcParams['image.cmap'] = 'viridis'
#
#print(len(np.unique(y_train)))
c = len(np.unique(y_train))
#
for i in range(c):
    #print(np.unique(y_train)[i])
    #print(y_train == i)
    #print(type(x_train[:, xnn]))
    #print(x_train[:, xnn][(y_train == i)])
    plt.scatter(x_train[:, xnn][(y_train == i)], x_train[:, ynn][(y_train == i)], alpha=0.5, s=10, marker='o', label=i)
#
###plt.scatter(x_train[:, xnn], x_train[:, ynn], c=y_train, alpha=0.5, s=10, marker='o')
#plt.legend(y_train)
#
plt.xlabel(xn); plt.ylabel(yn);
#plt.legend(labels=y_train, loc='best');
plt.legend(loc='best');


for theta in [thetas[0],thetas[2]]:
    #
    j = np.array([x_train[:, xnn].min(), x_train[:, xnn].max()])
    k = -(j * theta[1] + theta[0]) / theta[2]
    #
    #print(j)
    #print(k)
    #
    print('y = mx + n')
    #print(json.dumps(makeLinearEquation(j[0], k[0], j[1], k[1]), indent=4))
    #print(makeLinearEquation(j[0], k[0], j[1], k[1]))
    #print(type(makeLinearEquation(j[0], k[0], j[1], k[1])))
    #<class 'dict'>
    print('m = ' + str(makeLinearEquation(j[0], k[0], j[1], k[1])['m']))
    print('n = ' + str(makeLinearEquation(j[0], k[0], j[1], k[1])['n']))
    print('')
    #
    plt.plot(j, k, color='k', linestyle="--")
    #plt.text((j.max()-j.min())*0.50, (k.max()-k.min())*0.50, 'y = (' + str(makeLinearEquation(j[0], k[0], j[1], k[1])['m']) + ')* x + (' + str(makeLinearEquation(j[0], k[0], j[1], k[1])['n']) + ')', size=7)
    plt.text((j.max()-j.min())*0.50, k.max()-max(x_train[:, ynn])*0.50, 'y = (' + str(makeLinearEquation(j[0], k[0], j[1], k[1])['m']) + ')* x + (' + str(makeLinearEquation(j[0], k[0], j[1], k[1])['n']) + ')', size=7)
    #


plt.title("Training Data and Linear Boundaries (+ Test Data)")

# Three types of Species:
#
#Iris-setosa (0)
#Iris-versicolor (1)
#Iris-virginica (2)


#print(plt.rcParams['image.cmap'])
#viridis
#
#change cmap
plt.rcParams['image.cmap'] = 'rainbow'
#plt.rcParams['image.cmap'] = 'jet'


##### Test Data

c = len(np.unique(y_test))
#
for i in range(c):
    #plt.scatter(x_test[:, xnn][(y_test == i)], x_test[:, ynn][(y_test == i)], alpha=0.5, s=10, marker='*', label=i)
    plt.scatter(x_test[:, xnn][(y_test == i)], x_test[:, ynn][(y_test == i)], alpha=0.5, s=10, marker='*')
#
###plt.scatter(x_train[:, xnn], x_train[:, ynn], c=y_train, alpha=0.5, s=10, marker='o')
#plt.legend(y_train)
#
plt.xlabel(xn); plt.ylabel(yn);
#plt.legend(labels=y_test, loc='best', ncol=3);
plt.legend(loc='best', ncol=3);
#
###plt.scatter(x_test[:, xnn], x_test[:, ynn], c=y_test, alpha=1.0, s=10, marker='*')


#reset cmap
plt.rcParams['image.cmap'] = 'viridis'


plt.savefig("Figure_6_Training_Data_and_Linear_Boundaries_plus_Test_Data.png")
plt.show()




########## Train and Test Accuracy

def score(classes, theta, x, y):
    return (predict(classes, theta, x) == y).mean()


##### Using only two features (xn and yn)

print("These are the results of TWO SPECIFIED explanatory variables:")
print(f"Train Accuracy: {score(classes, thetas, x_train[:, [xnn, ynn]], y_train):.3f}")
print(f"Test  Accuracy: {score(classes, thetas, x_test[:, [xnn, ynn]], y_test):.3f}")

#Train Accuracy: 0.942
#Test Accuracy: 0.933

print("")


##### Using all features

thetas, classes, costs = fit(x_train, y_train)

print("If we use ALL explanatory variables:")
print(f"Train Accuracy: {score(classes, thetas, x_train, y_train):.3f}")
print(f"Test  Accuracy: {score(classes, thetas, x_test, y_test):.3f}")

#Train Accuracy: 0.967
#Test Accuracy: 0.967