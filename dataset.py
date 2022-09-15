#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 14:21:18 2020

@author: alessandro
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
from scipy.integrate import quad
path = '/lines_20/'
#  Covert alpha deg to alpha rad
alpha = np.linspace(1,89,89)
rad =[]
for i in range(len(alpha)):
    rad.append(np.array([radians(alpha[i])]))
rad = np.array(rad)
ang_coeff =[]
for i in range(len(alpha)):
    ang_coeff.append(np.array([tan(rad[i])]))
ang_coeff = np.array(ang_coeff)


def line(ang_coeff,x):
    return ang_coeff*x

index = 0
gap = 1
verticalShift = 2
for horizontalShift in range(7,14):
    img = np.ones(shape=(20,20,3),dtype=float)
    lines = []
    for i in range((len(rad))):
        lines.append(img.copy())
    f1 = (int(index/89))*1000
    for image in lines:
        image[0:verticalShift,:]=0
        for i in range(img.shape[0]-horizontalShift): 
            I = quad(line,i,i+1,args=(ang_coeff[index%89]))
            remainder = 1-I[0]%1 # 0 means black, we want to fill the percentage of the pixels with the right gray scale 
            integer = int(I[0]//1) # index of the column below which is being calculating the integral, namely the area that must be filled with the value 0
            if integer > img.shape[0]-1-verticalShift:
                 image[:,i+horizontalShift]=0
            else:
                image[integer+verticalShift,i+horizontalShift] = remainder #fill the current pixel on the straight line (img[y,x])
                for j in range (integer):
                    image[j+verticalShift,i+horizontalShift]=0
        image = np.flipud(image)
        index = index +1
        image = image*255 
        filename = path+'lines%04d' %f1 +'.png'
        cv2.imwrite(filename,image)
        f1 = f1+1
        
    for i in range(len(lines)):
        lines[i] = np.flipud(lines[i])
    lines2=[]
    
    for i in range(len(lines))[::-1]:
        a =1-lines[i]
        a[20-verticalShift:20,:]=0
        lines2.append(a)
    
    f=(int(index/89)-1)*1000+91
    for image in lines2:
        image = np.float32(image)
        image = image*255 
        filename = path+'lines%04d' %f +'.png'
        cv2.imwrite(filename,image)
        f=f+1