#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 11:30:16 2020

@author: alessandro
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
from contact_angles import *
from math import atan, degrees, pi, tan, sin, sqrt, asin, cos, radians
imageformat = ".png"
path = '/20x20/'
imfilelist = [os.path.join(path,f) for f in os.listdir(path) if f.endswith(imageformat)]
imfilelist.sort()

contact_angles = [x for x in np.linspace(1,179,179)]
del contact_angles[89]

def replicate(List, n):
    return List*n

contact_angles = replicate(contact_angles,28)

import pandas as pd
contact_angles = pd.DataFrame(data =contact_angles, columns=['Contact Angles'])
contact_angles= contact_angles/180

x = np.linspace(1,179,179)
x = np.delete(x,89)
y = np.arange(7)

array1 = []
for i in range(len(y)):
    for j in range(len(x)):
        z = ["lines%01d%03d" %(y[i] , x[j])+".png" ]
        array1.append(z)

array2 = []
for i in range(3):
    for j in range(1246):
        zz = ["blur%01d%04d" %(i,j)+".png" ]
        array2.append(zz)

array = array1 +array2
#array = ["lines%01d%03d" %(y[9] , x[45])+".png" ]

        
data_x = pd.DataFrame(data=array, columns=['Filename'],dtype=str)
my_data = pd.concat([data_x,contact_angles],axis=1) # axis = 1 means columns

from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Layer
import tensorflow as tf

# =============================================================================
# onfig = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.9
# session = tf.Session(config=config)
# config.gpu_options.allow_growth = True
# =============================================================================


# =============================================================================
# from tensorflow.compat.v1 import InteractiveSession
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# session = InteractiveSession(config=config)
# =============================================================================

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

class MCDropout(Layer):
    def __init__(self, rate):
        super(MCDropout, self).__init__()
        self.rate = rate

    def call(self, inputs):
        return tf.nn.dropout(inputs, rate=self.rate)

input_shape = (20,20,1)
# Initializing the CNN
model = Sequential()
# Convolution
model.add(Conv2D(64, (3, 3), input_shape = input_shape, activation='relu')) # 32 Feature Detector with dimension (3,3)
# Pooling
model.add(MaxPooling2D(pool_size = (2, 2))) # Reducing the size of the feature map

model.add(Conv2D(128, (3, 3), input_shape = input_shape, activation='relu')) # 64 Feature Detector with dimension (3,3)
model.add(MaxPooling2D(pool_size = (2, 2)))
#model.add(Conv2D(256, (3, 3), input_shape = input_shape, activation='relu')) # 64 Feature Detector with dimension (3,3)
#model.add(MaxPooling2D(pool_size = (2, 2)))


# Flattening
model.add(Flatten())
model.add(MCDropout(rate=0.5))
# Full connection
model.add(Dense(32, activation='relu'))
#model.add(Dense(512, activation='relu'))

model.add(Dense(1, activation='sigmoid')) # with binary output we use sigmoid
# Compiling
model.compile(optimizer='adam',
              loss='mse',
              metrics=['accuracy'])


from keras.preprocessing.image import ImageDataGenerator

#train_datagen = ImageDataGenerator(rescale=1./255,featurewise_center=True, featurewise_std_normalization=True)
train_datagen = ImageDataGenerator(rescale=1./255,
        #width_shift_range=0.1,
        #shear_range=0.2,
        #zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.1)


training_set = train_datagen.flow_from_dataframe(my_data, directory=path, x_col='Filename', 
                    y_col='Contact Angles', weight_col=None, target_size=(20, 20), 
                    color_mode='grayscale', classes=None, class_mode='raw', 
                    batch_size=32, shuffle=True,
                    subset='training')


test_set = train_datagen.flow_from_dataframe(my_data, directory=path, x_col='Filename', 
                    y_col='Contact Angles', weight_col=None, target_size=(20, 20), 
                    color_mode='grayscale', classes=None, class_mode='raw', 
                    batch_size=32, shuffle=True,
                    subset='validation')


history = model.fit_generator(
        training_set,
        steps_per_epoch = training_set.samples,
        validation_data=test_set,
        validation_steps = test_set.samples,
        epochs=15)


loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# =============================================================================
# # Testing the dataset on new images
# from keras.preprocessing import image
# img = image.load_img('/crop0137.png', target_size=(20,20,1), color_mode='grayscale') #20,44,121,190
# #img = image.load_img('/crop5a.png', target_size=(20,20,1), color_mode='grayscale')
# #img = image.load_img('/20x20_test/blur20177.png', target_size=(20,20,1), color_mode='grayscale') 
# img = image.img_to_array(img)
# img = img /255
# #img[20:30,0:30,:]=0 #[y,x]
# img = img.reshape((1,) + img.shape)
# prediction = model.predict(img)
# print(prediction*180)
# 
# from tqdm import tqdm
# #import contact_angles
# imageformat = ".png"
# path = '/error_analysis/1200/'
# 
# imfilelist = [os.path.join(path,f) for f in os.listdir(path) if f.endswith(imageformat)]
# imfilelist.sort()
# 
# from keras.preprocessing import image
# folder = []
# for IMG in imfilelist:
#    from keras.preprocessing import image
#    ima = image.load_img(IMG, target_size=(20,20,1), color_mode='grayscale')
#    folder.append(ima)
# 
# # CNN Prediction 
# contact_angle = []
# trend_err = np.zeros([len(folder)])
# for img in tqdm(folder):
#     img = image.img_to_array(img)
#     img = img /255
#     img = img.reshape((1,) + img.shape)
#     prediction = model.predict(img)
#     contact_angle.append(prediction*180)
# 
# #contact_angle = np.array(contact_angle)
# #contact_angle = contact_angle[:,0,0]
# 
# arr = [contact_angles_ellipse]
# error = np.zeros(len(folder))
# for i in range(len(folder)):
#     #error[i] = abs((contact_angle[i]-arr[0][i])/arr[0][i])
#     error[i] = abs((contact_angle[i]-arr[0][i]))
# #ERROR= sum(list(error))/len(error) #average    
# savepath = '/error_csv/'
# cnn_measurment = list(zip(np.array(arr[0]), error))
# 
# av_err = sum(list(error))/len(error)
# 
#  #average 
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.plot(np.array(arr[0]),error,'co-')
# ax.set_xlabel('Angles')
# ax.set_ylabel('Error')
# plt.show()
# 
# df = pd.DataFrame(cnn_measurment)
# df.to_csv(savepath+'CNN_ERROR.csv',sep=' ',header=None, index=False)
# =============================================================================