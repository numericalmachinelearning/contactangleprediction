#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 15:09:31 2019

@author: alessandro
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
imageformat = ".png"
path = '/lines_20/'
savepath = '/20x20_test/'
imfilelist = [os.path.join(path,f) for f in os.listdir(path) if f.endswith(imageformat)]
imfilelist.sort()
data = []
for IMG in imfilelist:
   image = cv2.imread(IMG)
   data.append(image)

standardized = []
i = 0
for image in data:
    img = cv2.GaussianBlur(image,(9,9),0)
    img[18:,:]=0
    standardized.append(img)
    filename = savepath+'blur2%04d' %i +'.png'
    cv2.imwrite(filename, img)
    i=i+1



x = np.linspace(1,179,179)
x = np.delete(x,89)
y = np.arange(7)

array = []
for i in range(len(y)):
    for j in range(len(x)):
        z = ["lines%01d%03d" %(y[i] , x[j])+".png" ]
        array.append(z)
        
def listToString(s):  
    str1 = " "  
    return (str1.join(s)) 

for i in range(len(array)):
    array[i] = listToString(array[i])

data = []   
for i in range(len(array)):
    img = cv2.imread(path+array[i])
    data.append(img)