#################### Logistic Regression for Multi-class Classification ####################
#
#  (C) 2021, Yoshimasa (Yoshi) Satoh, CFA 
#
#  All rights reserved.
#
# Created:      2021/09/29
# Last Updated: 2021/09/30
#
# Github:
# https://github.com/yoshisatoh/ML/tree/main/Supervised/Classification/Logistic_Regression/multi_class/lgstcrgmc.py
# https://github.com/yoshisatoh/ML/blob/main/Supervised/Classification/Logistic_Regression/multi_class/lgstcrgmc.py
#
#
########## Input Data File(s)
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
#python lgstcrgmc.py iris.csv Species petal_length petal_width 5000 0.1
#or
#python lgstcrgmc.py iris.csv Species sepal_length sepal_width 5000 0.1
#
# Generally,
#python lgstcrgmc.py (dtf: data file) (categoryname: category names for classification) (x-axis) (y-axis) (max_iter) (alpha: learning rate)
#
# x and y-axes will be used to draw a graph with boundary lines for classification.
#
# The learning rate (alpha) determines how rapidly we update the parameters.
# If the learning rate is too large, we may "overshoot" the optimal value.
# Similarly, if it is too small, then we will need so many iterations to converge to the best values.
# Thus, it is crucial to use a well-tuned learning rate.
#
#
########## References
#https://archive.ics.uci.edu/ml/datasets/iris
#https://teddykoker.com/2019/06/multi-class-classification-with-logistic-regression-in-python/
#https://github.com/teddykoker/blog/blob/master/_notebooks/2019-06-16-multi-class-classification-with-logistic-regression-in-python.ipynb
#
#
####################




########## import Python libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import sys




########## color settings

plt.rcParams["figure.figsize"] = (5, 4)    # (w, h)
plt.rcParams["figure.dpi"]     = 200
#
#print(plt.rcParams['image.cmap'])
#viridis


# color map for labels of legend in graphs
#cm = plt.cm.get_cmap('RdYlBu')
#cm = plt.cm.get_cmap('rainbow')
#cm = plt.cm.get_cmap('viridis')
cm = ['#FF0000', '#00FF00', '#0000FF', '#FF00FF']    ##### Add other colors if you have more than 4 classifications/classes
#print(type(cm))
#print(cm[0])




########## np.random.seed(n)

# training data and test data are always the same if you do not change the following [1] seed(n)
np.random.seed(42)
#
# .. and also
# [2] num_train (0.80), and
# [3] training-test data ratio (training data : test data = 0.80 : 0.20) as shown below




########## arguments

#print(str(sys.argv[0]))    #0 specifies a python script itself, 'lgstcrgmc.py', not an argument

dtf          = str(sys.argv[1])    #'iris.csv'
categoryname = str(sys.argv[2])    #'Species'
xn           = str(sys.argv[3])    #'petal_length'
yn           = str(sys.argv[4])    #'petal_width'

max_iter     = int(sys.argv[5])    #5000
alpha        = float(sys.argv[6])  #0.1




########## Figure 1: A Sigmoid Function
#
# Let’s say we want to classify our data into two categories: negative (0) and positive (1).
#
# Unlike linear regression, where we predict a continuous value,
# in classification, we let our classifier to predict the probability that the data is positive (1), or negative (0).
# For the latter classification problem, especially in logistic regression, we use a Sigmoid function.
# Note that logistic "regression" is actually used for classification of discrete numbers (classes), not regression of continuous values.


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# If we plot the sigmoid function, we notice that as the input x approaches +∞, the output sigmoid(x) approaches 1,
# and as the input x approaches −∞, the output sigmoid(x) approaches 0.

x = np.linspace(-10, 10, 200)

plt.figure(1)

plt.plot(x, sigmoid(x))
plt.axvline(x=0, color='k', linestyle='--');
plt.title("Sigmoid Function");
plt.xlabel('x'); plt.ylabel('sigmoid(x)')
plt.savefig("Figure_1_Sigmoid_Function.png")
plt.show()

#exit()




########## Figure 2: Cost Function J

# By passing the product of our inputs x and parameters θT to the sigmoid function, g,
# we can form a prediction h of the probability of input x being classified as positive (1).
#
# hθ(x) = g(θT x)
#
# When we perform linear regression, we usually use Mean Squared Error as our cost function. This works well for liner regression.
#
# However, for classification, we use the Cross Entropy Loss function J.
# Once again, please note that logistic "regression" is actually used for classification of discrete values, not regression of continuous values.

h = np.linspace(0, 1)[1:-1]

plt.figure(2)

for y in [0, 1]:
    plt.plot(h, -y * np.log(h) - (1 - y) * np.log(1 - h), label=f"y={y}")

plt.title("Cross Entropy Loss") 
# J: Cross Entropy Loss (=Cost) Functon
plt.xlabel('$h_ {\\theta}(x)$'); plt.ylabel('$J(\\theta)$')
plt.legend();
plt.savefig("Figure_2_Cross_Entropy_Loss_Function.png")
plt.show()

# We can see that a prediction matching the classification will have a cost of 0,
# but approach infinity as the prediction approaches the wrong classification.

#exit()




########## A Gradient Function

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

##### Training function fit
#
#
#print(max_iter)
#5000
#
#print(alpha)
#0.1
#
#
#def fit(x, y, max_iter=5000, alpha=0.1):
def fit(x, y, max_iter, alpha):
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

#exit()


##### Prediction function that predicts a class label using the maximum hypothesis hθ(x):
#
def predict(classes, thetas, x):
    x = np.insert(x, 0, 1, axis=1)
    preds = [np.argmax(
        [sigmoid(xi @ theta) for theta in thetas]
    ) for xi in x]
    return [classes[p] for p in preds]




########## Input Data Set

# We will use the iris data set, a commonly used dataset containing 3 species of iris plants. These 3 species will be used as 3 distinct classification categories (0, 1, and 2).
#
# Each plant in the dataset has 4 attributes:
#    sepal length,
#    sepal width,
#    petal length, and
#    petal width.
#
# We will use our logistic regression model to predict flowers’ species using TWO SPECIFIED attributes out of these 4 attributes.


##### raw input dataset

#dtf = "iris.csv"
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


##### classification data frame (dfclass)

# Now we encode classification categories from string to integer
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
print('')
#exit()
#
#
#print(dfclass.columns)
#Index(['Species', 'class'], dtype='object')
#exit()
#
#
#print(dfclass[categoryname].unique())
#['Iris-setosa' 'Iris-versicolor' 'Iris-virginica']
#exit()
#
#
#print(dfclass['class'].unique())
#[0 1 2]
#exit()


df[categoryname] = df[categoryname].astype('category').cat.codes


#xn = 'petal_length'
xnn = df.columns.get_loc(xn)
#
#print(xnn)
#2


#yn = 'petal_width'
ynn = df.columns.get_loc(yn)
#
#print(ynn)
#3


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


##### Shuffle the data, and split it into training and test data


# shuffle the data
#
data = np.array(df)
np.random.shuffle(data)


# split data into training and test data
#
num_train = int(0.80 * len(data))    # 80/20 train/test split
#
x_train, y_train = data[:num_train, :-1], data[:num_train, -1]
x_test,  y_test  = data[num_train:, :-1], data[num_train:, -1]


# save as csv files
#
np.savetxt('x_train.csv', x_train, delimiter=',', fmt='%.8f')
np.savetxt('y_train.csv', y_train, delimiter=',', fmt="%.0f")
np.savetxt( 'x_test.csv', x_test,  delimiter=',', fmt='%.8f')
np.savetxt( 'y_test.csv', y_test,  delimiter=',', fmt="%.0f")




########## Figure 3: Training Data

plt.figure(3)

c = len(np.unique(y_train))
#
for i in range(c):
    #plt.scatter(x_train[:, xnn][(y_train == i)], x_train[:, ynn][(y_train == i)], alpha=0.5, s=10, marker='o', label=i)
    #plt.scatter(x_train[:, xnn][(y_train == i)], x_train[:, ynn][(y_train == i)], alpha=0.5, s=10, marker='o', cmap=cm, label=i)
    plt.scatter(x_train[:, xnn][(y_train == i)], x_train[:, ynn][(y_train == i)], alpha=0.5, s=10, marker='o', color=cm[i], label=i)
#

plt.xlabel(xn); plt.ylabel(yn);
plt.title("Training Data")
plt.legend(loc='best');
plt.savefig("Figure_3_Training_Data.png")
plt.show()




########## Figure 4: Training Data and cost improvement over each epoch

plt.figure(4)

thetas, classes, costs = fit(x_train[:, [xnn, ynn]], y_train, max_iter, alpha)

plt.plot(costs)
plt.xlabel('number of epochs'); plt.ylabel('cost');
plt.title("Training Data and cost improvement over each epoch")
plt.savefig("Figure_4_Training_Data_and_cost_improvement_over_each_epoch.png")
plt.show()




########## Figure 5: Training Data and Linear Boundaries

##### Training Data

plt.figure(5)

c = len(np.unique(y_train))


for i in range(c):
    #plt.scatter(x_train[:, xnn][(y_train == i)], x_train[:, ynn][(y_train == i)], alpha=0.5, s=10, marker='o', label=i)
    #plt.scatter(x_train[:, xnn][(y_train == i)], x_train[:, ynn][(y_train == i)], alpha=0.5, s=10, marker='o', cmap=cm, label=i)
    plt.scatter(x_train[:, xnn][(y_train == i)], x_train[:, ynn][(y_train == i)], alpha=0.5, s=10, marker='o', color=cm[i], label=i)


plt.xlabel(xn); plt.ylabel(yn);

for theta in [thetas[0],thetas[2]]:
    j = np.array([x_train[:, xnn].min(), x_train[:, xnn].max()])
    k = -(j * theta[1] + theta[0]) / theta[2]
    plt.plot(j, k, color='k', linestyle="--")


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


# For example,
#print(json.dumps(makeLinearEquation(2, 4, 3, 7), indent=4))
# draws a line that passes both (2, 4) and (3, 7), which is y = (m * x) + n where m = 3.0 and n = -2.0
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




########## Figure 6: Training Data and Linear Boundaries (+ Test Data)

plt.figure(6)

##### Training Data
#
#
c = len(np.unique(y_train))
#
#
for i in range(c):
    #plt.scatter(x_train[:, xnn][(y_train == i)], x_train[:, ynn][(y_train == i)], alpha=0.5, s=10, marker='o', label=i)
    #plt.scatter(x_train[:, xnn][(y_train == i)], x_train[:, ynn][(y_train == i)], alpha=0.5, s=10, marker='o', cmap=cm, label=i)
    plt.scatter(x_train[:, xnn][(y_train == i)], x_train[:, ynn][(y_train == i)], alpha=0.5, s=10, marker='o', color=cm[i], label=i)
#
#
plt.xlabel(xn); plt.ylabel(yn);
plt.legend(loc='best');


##### Boundary Lines (calculated by Training Data)
for theta in [thetas[0],thetas[2]]:
    #
    j = np.array([x_train[:, xnn].min(), x_train[:, xnn].max()])
    k = -(j * theta[1] + theta[0]) / theta[2]
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
    plt.text((j.max()-j.min())*0.50, k.max()-max(x_train[:, ynn])*0.50, 'y = (' + str(makeLinearEquation(j[0], k[0], j[1], k[1])['m']) + ')* x + (' + str(makeLinearEquation(j[0], k[0], j[1], k[1])['n']) + ')', size=7)


# Three types of Species:
#
#Iris-setosa (0)
#Iris-versicolor (1)
#Iris-virginica (2)


##### Test Data

c = len(np.unique(y_test))
#
for i in range(c):
    plt.scatter(x_test[:, xnn][(y_test == i)], x_test[:, ynn][(y_test == i)], alpha=0.5, s=10, marker='*', color=cm[i], label= 'Test ' + str(i))
#
#


plt.title("Training Data and Linear Boundaries (+ Test Data)")
plt.xlabel(xn); plt.ylabel(yn);
plt.legend(loc='best', ncol=2);


plt.savefig("Figure_6_Training_Data_and_Linear_Boundaries_plus_Test_Data.png")
plt.show()




########## Train and Test Accuracy


def score(classes, theta, x, y):
    return (predict(classes, theta, x) == y).mean()


##### Using only TWO SPECIFIED features (xn and yn)

print("These are the results of TWO SPECIFIED explanatory variables:")
print(f"Train Accuracy: {score(classes, thetas, x_train[:, [xnn, ynn]], y_train):.3f}")
print(f"Test  Accuracy: {score(classes, thetas, x_test[:, [xnn, ynn]], y_test):.3f}")

#Train Accuracy: 0.942
#Test  Accuracy: 0.933

print("")


##### Using ALL the features

thetas, classes, costs = fit(x_train, y_train, max_iter, alpha)

print("If we use ALL the explanatory variables:")
print(f"Train Accuracy: {score(classes, thetas, x_train, y_train):.3f}")
print(f"Test  Accuracy: {score(classes, thetas, x_test, y_test):.3f}")

#Train Accuracy: 0.967
#Test  Accuracy: 0.967

print("")