# Loading dependencies.
import sys
import os
import time
import matplotlib.pyplot as plt
import seaborn
from time import sleep
# Data pre-processing libraries.
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import imutils

# Machine learning modelling libraries.
import tensorflow as ts
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization
from keras.layers import Activation, Dropout
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.optimizers import Adam,SGD
"""
import training data adn process
"""
path = '/Users/linyuchun/Desktop/Coop program/MRR/MRR_data/crack/train/image/'
path2 = '/Users/linyuchun/Desktop/Coop program/MRR/MRR_data/good phone/data/train/smartphone/'

label_data = []

demage_num = 0
for file in os.listdir(path):
    if os.path.isfile(path+file):
            label_data.append(path+file)
            demage_num += 1

#process the data
x_var = []
y_var = []

for lab in label_data:
     img = cv2.imread(lab)
     img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
     x_var.append(img.tolist())


label_data = []

complete_num = 0
for file in os.listdir(path2):
    if os.path.isfile(path2+file):
            label_data.append(path2+file)
            complete_num += 1


for lab in label_data:
     img = cv2.imread(lab)
     img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
     x_var.append(img.tolist())


#inital the output of the training 
outputVectors = []
for i in range(1, demage_num+1):
    outputVectors.append([1, 0])

for i in range(1, complete_num+1):
    outputVectors.append([0, 1])


x_var = np.asarray(x_var)
y_var = np.asarray(outputVectors)


# Rescaling pixel values between 0 and 1.
x_var = np.interp(x_var, (x_var.min(), x_var.max()), (0, 1))

    # Splitting data into training and test sets.
(train_x, test_x, train_y, test_y) = train_test_split(x_var, y_var,
                                                          test_size=0.3, random_state=42)

    # Model input parameters.
height = x_var.shape[-2]
#width = x_var.shape[-3]
depth = x_var.shape[-1]
input_shp = (height,depth,1)
"""
Model training structure
"""
model = Sequential()

#add the hidden layer: convolution(amount of the kernal, kernal_size, activation type, imput shape)
model.add(Conv2D(32, kernel_size=(3,3), activation='relu', input_shape = input_shp))
model.add(BatchNormalization())
#add the hidden later: pooling
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.3))

#second layer
model.add(Conv2D(64, kernel_size=(3,3), activation = 'relu',padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.3))

#third layer
model.add(Conv2D(64, kernel_size=(3,3), activation='relu',padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.3))

#flatten the feature map
model.add(Flatten())
model.add(Dense(units = 256, activation = 'relu'))
model.add(Dropout(0.5))

# model.add(Dense(units = 128, activation = "sigmoid"))
# model.add(Dropout(0.5))

#add the fully connecting output layer
model.add(Dense(y_var.shape[-1], activation = "sigmoid"))

#summrize the model
model.compile(optimizer= Adam(learning_rate=0.01), loss='categorical_crossentropy', metrics=['accuracy'])
# #train the model
model.fit(train_x, train_y, epochs= 20, batch_size = 64, validation_data=(test_x, test_y)) #train the model

loss, accuracy = model.evaluate(test_x, test_y) # Evaluate
print(f'Test loss: {loss}, Test accuracy: {accuracy}')                                                                                             

model.save('Phone_crack_recog_v1.models')
