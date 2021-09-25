#################### Deep Learning (Regression, Supervised Learning): Implementation and Showing Biases and Weights ####################
#
#  (C) 2021, Yoshimasa (Yoshi) Satoh, CFA 
#
#  All rights reserved.
#
# Created:      2021/09/25
# Last Updated: 2021/09/25
#
# Github:
# https://github.com/yoshisatoh/ML/tree/main/Supervised/Regression/DNN/dlregrwgts.py
# https://github.com/yoshisatoh/ML/blob/main/Supervised/Regression/DNN/dlregrwgts.py
#
#
########## Input Data Files
#
#train_data_raw.csv
#train_targets_raw.csv
#test_data_raw.csv
#test_targets_raw.csv
#
#
########## Usage Instructions
#
# You can run this code on your MacOS Terminal (or Windows Command Prompt) as follows:
#
# python3 dlregrwgts.py 500 l1l2 0.0001
# python3 dlregrwgts.py (num_epochs: number of epochs) (regl1l2: regularization) (regl1l2f: learning rate of regularization)
#
#
####################




########## import sys

import sys




########## Argument(s)

#num_epochs = 20
num_epochs = int(sys.argv[1])

#regl1l2  = 'None'
#regl1l2  = 'l1l2'
regl1l2 = str(sys.argv[2])

#regl1l2f = 0.0200
regl1l2f = float(sys.argv[3])

#dropout_rate = 0
#dropout_rate = float(sys.argv[4])




########## import others

import numpy as np

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, cohen_kappa_score
from sklearn.metrics import r2_score

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras import Model
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Dropout

print(tf.__version__)
#2.3.0




########## Loading raw data (before standardization)

train_data_raw    = np.loadtxt('train_data_raw.csv', dtype='float64', delimiter=',', skiprows=1)

train_targets_raw = np.loadtxt('train_targets_raw.csv', dtype='int', delimiter=',', skiprows=1)

test_data_raw     = np.loadtxt('test_data_raw.csv', dtype='float64', delimiter=',', skiprows=1)

test_targets_raw  = np.loadtxt('test_targets_raw.csv', dtype='int', delimiter=',', skiprows=1)




########## Standardization (data/features to have average = 0, standard deviation = 1)

sc = StandardScaler()

train_data = sc.fit_transform(train_data_raw)
#train_data    = train_data_raw    # no standardization
np.savetxt('train_data.csv', train_data, fmt ='%.8f', delimiter=',')
#
print(train_data.shape)
#(39, 2)
#
#If you look at train_data_raw.csv, then it has 39 rows (excluding header) and 2 columns (transactionType,orderId)
#
print(train_data.shape[0])
#39
#
print(train_data.shape[1])
#2

train_targets = train_targets_raw
np.savetxt('train_targets.csv', train_targets, fmt ='%i', delimiter=',')

test_data = sc.fit_transform(test_data_raw)
#test_data     = test_data_raw    # no standardization
np.savetxt('test_data.csv', test_data, fmt ='%.8f', delimiter=',')

test_targets  = test_targets_raw
np.savetxt('test_targets.csv', test_targets, fmt ='%i', delimiter=',')




##### Regularization

if regl1l2 == 'None':
    rg = None
    #
elif regl1l2 == 'l1':
    rg = regularizers.l1(l1=regl1l2f)    # L1 regularization
    #
elif regl1l2 == 'l2':
    rg = regularizers.l2(l2=regl1l2f)    # L2 regularization
    #
elif regl1l2 == 'l1l2':
    rg = regularizers.l1_l2(l1=regl1l2f, l2=regl1l2f)    # L1 & L2 regularization
    #
else:
    print('Error: The second argument should be None, l1, l2, or l1l2.')
    exit()




########## Model

#all-node-connected network

model = Sequential([
    #
    Dense(train_data.shape[1], kernel_regularizer=rg, activation='relu', name='L0_dense', use_bias=True, input_shape=(train_data.shape[1],)),
    #Dropout(dropout_rate, name='L1_dropout'),
    #
    #
    ##### ********** Binary Classification **********
    #####Dense(1, activation='sigmoid', name='L1_dense')
    #
    ##### ********** Regression **********
    Dense(1, name='L1_dense')
    #####
])

model.summary()
'''
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
L0_dense (Dense)             (None, 2)                 6         
_________________________________________________________________
L1_dense (Dense)             (None, 1)                 3         
=================================================================
Total params: 9
Trainable params: 9
Non-trainable params: 0
_________________________________________________________________
'''
'''
See Fig.png. (This is a case of binary classification with a sigmoid function at the output layer, NOT regression.)


L0_dense (Dense):

Param #
1st hidden layer (with the nodes y0_0 and y0_1) has 2 input parameters for each node and
1 output parameter for each node.
Thus, 2 input parameters * 2 nodes + 1 output parameters * 2 nodes = 2 * 2 + 1 * 2 = 6 parameters

Output Shape
There are 2 outputs (one for each node, i.e., nodes y0_0 and y0_1)


L1_dense (Dense):

Param #
output layer has 2 input parameters and 1 output parameter. Thus, there are 3 parameters.

Output Shape
output layer has 1 output parameter
'''




########## Model Compiling: Classification (Binary Class: 0 or 1)

##### ********** Binary Classification **********
##### model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#
##### ********** Regression **********
model.compile(optimizer='rmsprop', loss='mse', metrics=['mean_absolute_error'])
#
#####




########## Model Fitting and History Recording

history = model.fit( train_data,
                     train_targets,
                     validation_data=(test_data, test_targets),
                     epochs=num_epochs,
                     batch_size=1,
                     verbose=1)    # verbose=1: Show progresses in training (verbose=0: Not showing progresses)

'''
Epoch 1/500
39/39 [==============================] - 0s 4ms/step - loss: 0.7375 - mean_absolute_error: 0.6619 - val_loss: 0.6704 - val_mean_absolute_error: 0.6125
...
Epoch 500/500
39/39 [==============================] - 0s 1ms/step - loss: 0.0109 - mean_absolute_error: 0.0490 - val_loss: 0.0104 - val_mean_absolute_error: 0.0465
'''


#print(history.history)

loss     = history.history['loss']
val_loss = history.history['val_loss']

##### Binary Classification
#####acc      = history.history['accuracy']
#####val_acc  = history.history['val_accuracy']
#
##### Regression
mae      = history.history['mean_absolute_error']
val_mae  = history.history['val_mean_absolute_error']
#####

epochs = range(1, len(loss)+1)




########## Drawing figures

##### Loss
plt.plot(epochs, loss, 'r', label='Training')
plt.plot(epochs, val_loss, 'b', label='Validation')

plt.xlabel('epochs')
plt.ylabel('loss')

plt.title('Training and validation loss')
plt.legend()

plt.savefig('Fig_1_Loss.png')
plt.show()


##### ********** Binary Classification **********
##### Accuracy
##### plt.plot(epochs, acc, 'r', label='Training')
##### plt.plot(epochs, val_acc, 'b', label='Validation')
##### plt.ylabel('accuracy')
#
##### ********** Regression **********
##### MAE
plt.plot(epochs, mae, 'r', label='Training')
plt.plot(epochs, val_mae, 'b', label='Validation')
plt.ylabel('MAE')
#####


plt.xlabel('epochs')


plt.title('Training and validation accuracy')
plt.legend()

##### ********** Binary Classification **********
##### Accuracy
##### plt.savefig('Fig_2_Accuracy.png')
#
##### ********** Regression **********
##### MAE
plt.savefig('Fig_2_MAE.png')


plt.show()




########## Model Evaluation by Test Data and Test Targets

#model.evaluate(test_data, test_targets)
score = model.evaluate(test_data, test_targets)
#2/2 [==============================] - 0s 620us/step - loss: 0.0880 - mean_absolute_error: 0.0904


print(score)
#[val_loss, val_mean_absolute_error]
#[0.01039028912782669, 0.04646139219403267]




########## Model Predictions by using Test Data

test_targets_pred_raw = model.predict(test_data)
#test_targets_pred = model.predict_classes(test_data)
#
#predict will return the scores of the regression and predict_class will return the class of your prediction. Although it seems similar there are some differences:
#
#Imagine you are trying to predict if the picture is a dog or a cat (you have a classifier):
#
#predict will return you: 0.6 cat and 0.4 dog (for example).
#predict_class will return you cat

#results before binary conversion
print(test_targets_pred_raw)
'''
[[-3.1396568e-02]
 [-2.0962298e-02]
 [-3.2768905e-02]
 [ 2.6732028e-01]
 [ 9.8438460e-01]
 [ 9.8438460e-01]
 [ 9.8438460e-01]
 [ 9.8438460e-01]
 [ 9.8438460e-01]
 [ 9.8438460e-01]
 [ 9.8438460e-01]
 [ 9.8438460e-01]
 [ 9.8438460e-01]
 [ 9.8438460e-01]
 [-4.7281212e-01]
 [-2.3706734e-02]
 [ 2.4425220e-01]
 [-3.3312380e-02]
 [-1.2601972e-02]
 [-9.5689297e-03]
 [-6.5361261e-03]
 [-3.5032034e-03]
 [-4.7022104e-04]
 [ 8.6286068e-03]
 [-1.3272464e-02]
 [ 7.5959563e-03]
 [ 2.8464556e-02]
 [-2.2878230e-02]
 [-2.0096302e-03]
 [ 1.8858969e-02]
 [-1.2443900e-02]
 [ 8.4246397e-03]
 [ 2.9293239e-02]
 [-1.4660358e-03]
 [ 1.9402504e-02]
 [ 6.1139584e-02]
 [ 3.8958013e-02]
 [ 6.9287479e-02]
 [ 1.5450442e-01]]
'''
np.savetxt('test_targets_pred_raw.csv', test_targets_pred_raw, fmt ='%.8f', delimiter=',')


##### ********** Binary Classification **********
#
#binary classification: 0 or 1 (If a predicted target is more than 0.5, then it is regarded as 1.)
#####test_targets_pred = test_targets_pred_raw > 0.5
#
##### ********** Regression **********
#no conversion for regression
test_targets_pred = test_targets_pred_raw
#
#####


np.savetxt('test_targets_pred.csv', test_targets_pred, fmt ='%.8f', delimiter=',')


##### ********** Binary Classification **********
#
#Confusion matrix
#print(confusion_matrix(test_targets, test_targets_pred))
'''
[[20  9]
 [ 0 10]]

                                 Predicted Labels
                                 0 (Negative)      1 (Positive)
Actual Labels    0 (Negative)    True  Negative    False Positive
                 1 (Positive)    False Negative    True  Positive

Namely,
[[TN FP]
 [FN TP]]
'''
#
##### ********** Regression **********
#No confusion matrix for regression
#####




########## Showing model weights

print(len(model.layers))
#2

l0 = model.layers[0]
l1 = model.layers[1]


##### Model Weights

print('##### Model Weights #####')

for w in model.weights:
    print('{:<25}{}'.format(w.name, w.shape))
'''
L0_dense/kernel:0        (2, 2)
L0_dense/bias:0          (2,)
L1_dense/kernel:0        (2, 1)
L1_dense/bias:0          (1,)
'''


##### Layer 0

print('##### Layer 0: Dense #####')


### kernel

#print(l0.weights[0])

print(l0.weights[0].name)
#L0_dense/kernel:0

print(l0.weights[0].numpy())
'''
[[-0.23233399  0.00596758]
 [ 0.28243706 -0.5938109 ]]
'''


### bias

#print(l0.weights[1])

print(l0.weights[1].name)
#L0_dense/bias:0

print(l0.weights[1].numpy())
'''
[-0.05158948  0.6012353 ]
'''




##### Layer 1

print('##### Layer 1: Dense #####')
#print('##### Layer 1: Dropout #####')


### kernel

#print(l1.weights[0])

print(l1.weights[0].name)
#L1_dense/kernel:0

print(l1.weights[0].numpy())
'''
[[-1.4531652 ]
 [-0.97440845]]
'''


### bias

#print(l1.weights[1])

print(l1.weights[1].name)
#L1_dense/bias:0

print(l1.weights[1].numpy())
#[0.9873226]




########## Notes: How can we interpret these results?

##### Layer 0

### kernel

#print(l0.weights[0].name)
#L0_dense/kernel:0
#
#print(l0.weights[0].numpy())
'''
[[-0.23233399  0.00596758]
 [ 0.28243706 -0.5938109 ]]
'''


### bias

#print(l0.weights[1].name)
#L0_dense/bias:0
#
#print(l0.weights[1].numpy())
'''
[-0.05158948  0.6012353 ]
'''


##### Layer 1

### kernel

#print(l1.weights[0].name)
#L1_dense/kernel:0
#
#print(l1.weights[0].numpy())
'''
[[-1.4531652 ]
 [-0.97440845]]
'''


### bias

#print(l1.weights[1].name)
#L1_dense/bias:0

#print(l1.weights[1].numpy())
#[0.9873226]




'''
By using these weights, we can derive the equations below:


Layer 0:

y0_0 = (-0.05158948) + (-0.23233399) * x0 + (0.28243706) * x1
y0_1 = (0.6012353)   + (0.00596758)  * x0 + (-0.5938109) * x1


Layer 1:

y = y1
  = (0.9873226) + (-1.4531652) * y0_0 + (-0.97440845) * y0_1

'''

# Compare (1) calculated results by using this equation and (2) test_targets_pred: results of model.predict(test_data)
#
#test_data: 4th and 5th rows
'''
(x0), (x1)
...
-1.10732304,1.02604273
1.01655886,1.02604273
...
'''
#
# (1) calculated results by using this equation
'''

For the 4th row data,
x0 = -1.10732304,
x1 =  1.02604273


Layer 0:

y0_0 = (-0.05158948) + (-0.23233399) * x0 + (0.28243706) * x1
y0_1 = (0.6012353)   + (0.00596758)  * x0 + (-0.5938109) * x1

y0_0 = (-0.05158948) + (-0.23233399) * (-1.10732304) + (0.28243706) * (1.02604273)
y0_1 = (0.6012353)   + (0.00596758)  * (-1.10732304) + (-0.5938109) * (1.02604273)

y0_0 = 	0.495471792197703
y0_1 = 	-0.0146480957668002


Layer 1:

y = y1
  = (0.9873226) + (-1.4531652) * y0_0 + (-0.97440845) * y0_1
  = (0.9873226) + (-1.4531652) * (0.495471792197703)+ (-0.97440845) * (-0.0146480957668002)
  = 0.281593462288246

Thus, the predicted result is 0.281593462288246 as in the 4th row of test_targets_pred.csv. (0.26732028. This value is a bit different due to an rounding error?)

'''




'''
For the 5th row data,
x0 = 1.01655886,
x1 = 1.02604273


Layer 0:

y0_0 = (-0.05158948) + (-0.23233399) * x0 + (0.28243706) * x1
y0_1 = (0.6012353)   + (0.00596758)  * x0 + (-0.5938109) * x1

y0_0 = (-0.05158948) + (-0.23233399) * (1.01655886) + (0.28243706) * (1.02604273)
y0_1 = (0.6012353)   + (0.00596758)  * (1.01655886) + (-0.5938109) * (1.02604273)

y0_0 = 0.0020218360819224
y0_1 = -0.0019736606179982


Layer 1:

y = y1
  = (0.9873226) + (-1.4531652) * y0_0 + (-0.97440845) * y0_1
  = (0.9873226) + (-1.4531652) * (0.0020218360819224)+ (-0.97440845) * (-0.0019736606179982)
  = 0.986307689749256

Thus, the predicted result is 0.986307689749256 as in the 5th row of test_targets_pred.csv. (0.98438460. This value is a bit different due to an rounding error?)

'''