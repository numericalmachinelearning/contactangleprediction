#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 09:14:18 2022

@author: alessandro
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#import cv2
import os
from contact_angles import *
imageformat = ".png"
path = '/20x20/'
imfilelist = [os.path.join(path,f) for f in os.listdir(path) if f.endswith(imageformat)]
imfilelist.sort()

contact_angles = [x for x in np.linspace(1,179,179)]
del contact_angles[89]

def replicate(List, n):
    return List*n

contact_angles = replicate(contact_angles,28)

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


# Importing the dataset
imageformat = ".png"
imfilelist = [os.path.join(path,f) for f in os.listdir(path) if f.endswith(imageformat)]
imfilelist.sort()
from keras.preprocessing import image
X = []
for IMG in imfilelist:
   ima = image.load_img(IMG, target_size=(20,20), color_mode='grayscale')
   ima = image.img_to_array(ima)
   ima /= 255
   ima = ima.reshape((400))
   X.append(ima)
X = np.array(X).reshape(-1,400)
y = my_data.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1)


# Training the Random Forest Regression model on the whole dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

# Evaluating the Model Performance
from sklearn.metrics import r2_score
r2_score(y_test, y_pred)

# =============================================================================
# img = image.load_img('/crop0137.png', target_size=(20,20,1), color_mode='grayscale') #20,44,121,190
# img = image.img_to_array(img)
# img = img /255
# img = img.reshape(1,400)
# prediction = regressor.predict(img)
# print(prediction*180)
# =============================================================================


from tqdm import tqdm
#import contact_angles
imageformat = ".png"
path = '/error_analysis/1200/'

imfilelist = [os.path.join(path,f) for f in os.listdir(path) if f.endswith(imageformat)]
imfilelist.sort()

from keras.preprocessing import image
folder = []
for IMG in imfilelist:
   ima = image.load_img(IMG, target_size=(20,20), color_mode='grayscale')
   ima = image.img_to_array(ima)
   ima /= 255
   ima = ima.reshape(1,400)
   folder.append(ima)

# RF Prediction 
contact_angle = []
trend_err = np.zeros([len(folder)])
for img in tqdm(folder):
    prediction = regressor.predict(img)
    contact_angle.append(prediction*180)

arr = [contact_angles_ellipse]
error = np.zeros(len(folder))
for i in range(len(folder)):
    #error[i] = abs((contact_angle[i]-arr[0][i])/arr[0][i])
    error[i] = abs((contact_angle[i]-arr[0][i]))
    
cnn_measurment = list(zip(np.array(arr[0]), error))

av_err = sum(list(error))/len(error)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(np.array(arr[0]),error,'co-')
ax.set_xlabel('Angles')
ax.set_ylabel('Error')
plt.show()