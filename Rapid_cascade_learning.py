# -*- coding: utf-8 -*-
"""Rapid cascade learning


"""

import keras

from keras.utils import to_categorical

import numpy as np

"""**Data Import**"""


# read the data
#(X_train, y_train), (X_test, y_test) = [your dataset]

"""**Data preproprocessing **"""

#number of classes
no_classes = y_test.shape[1]
no_samples = y_test.shape[0]
num_features = X_train.shape[1] # size of features

"""**First Neural Network Implementation**"""

#The first neural network trained on train data and evaluated on test data, here named model1

"""# **Rapid Cascade Training**

Extract probabilities
"""
y_softmax = model.predict(X_test)

y_max_softmax = np.amax(y_softmax, axis=1, keepdims=True)

y_estimated = model.predict_classes(X_test)


k = 0
l = 0
scaling_function = 1+((no_samples-200)/4000)
threshold = 1-1.2*(no_classes/(10*scaling_function))
upper_threshold = 1-(no_classes/(100*scaling_function))
flags = []
probability = np.zeros((y_max_softmax.shape[0], 1))  # it needs double (())

recalc_x_test = []
recalc_y_test = []

accepted_x_test = []
accepted_y_test = []
accepted_y_est = []

#keras can take numpy ndarray as input


for i, value in enumerate(y_estimated, 0):
  probability[i] = 1/np.exp(np.absolute(1-y_max_softmax[i])*2*np.log(2))
  if probability[i] < threshold:
    k = k+1
    flags.append(i)
    recalc_x_test.append(X_test[i, :])
    recalc_y_test.append(y_test[i, :])
  if probability[i] >= threshold:
    l = l+1
    flags.append(i)
    accepted_x_test.append(X_test[i, :])
    accepted_y_test.append(y_test[i, :])
    accepted_y_est.append(y_estimated[i])

iter=0
iter=iter+1
l = 0

#final y est
#classes_2
expanded_x=[]
expanded_y=[]

for i, value in enumerate(y_estimated, 0):#0 is the starting index of the counter

  #  probability[i]=1/np.exp(np.absolute(final_y_est[i]-classes_2[i])*2*np.log(2))
  if probability[i] > upper_threshold:
    expanded_x.append( X_test[i,:] )
    expanded_y.append( y_test[i,:] )
    l = l+1


expanded_x_np = np.asarray(expanded_x)
expanded_y_np = np.asarray(expanded_y)

recalc_x_test_np = np.asarray(recalc_x_test)
recalc_y_test_np = np.asarray(recalc_y_test)

accepted_x_test_np = np.asarray(accepted_x_test)
accepted_y_test_np = np.asarray(accepted_y_test)



x_train_expanded = np.copy(X_train)
y_train_expanded = np.copy(y_train)
#expanded y is one hot


#else
if expanded_x_np.shape[1] > 0:
  x_train_expanded = np.vstack([x_train_expanded, expanded_x_np])
  y_train_expanded = np.vstack([y_train_expanded, expanded_y_np])
else:
  x_train_expanded = x_train_expanded
  y_train_expanded = y_train_expanded


"""**SECOND NEURAL NETWORK TRAINING **"""

#model2 parameters definition

num_features = X_train.shape[1]  # size of features

#The Second neural network trained on expanded train data and evaluated on test data, here named model2


recalc_x_est2 = np.copy(recalc_x_test_np)
recalc_y_est2 = model2.predict_classes(recalc_x_est2)

recalc_y_est2_one_hot = to_categorical(recalc_y_est2, num_classes=no_classes)


recalc_y_test_np = np.asarray(recalc_y_test)
accepted_y_test_np = np.asarray(accepted_y_test_np)



accepted_y_est_np = np.asarray(accepted_y_est)
accepted_y_est_np_one_hot = to_categorical(accepted_y_est_np, num_classes=no_classes)



#x_final = np.vstack([accepted_x_test, recalc_x_est2])

y_est_final = np.vstack([accepted_y_est_np_one_hot, recalc_y_est2_one_hot])
y_test_final = np.vstack([accepted_y_test_np, recalc_y_test_np])

acc = np.equal(y_est_final, y_test_final)
acc2 = np.prod(acc, keepdims=True, axis=1)


print((np.sum(acc2))/y_est_final.shape[0])