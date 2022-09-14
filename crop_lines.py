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
from numba import jit
imageformat = ".png"
path = '/home/alessandro/Documents/Python/droplet/droplet_database/600_3/'
savepath = '/home/alessandro/Documents/Python/droplet/droplet_database/error_analysis/600/'
imfilelist = [os.path.join(path,f) for f in os.listdir(path) if f.endswith(imageformat)]
imfilelist.sort()
data = []
for IMG in imfilelist:
   image = cv2.imread(IMG)
   data.append(image)


#path = '/home/alessandro/Documents/Python/droplet/droplet_database/'
#n = 44
#image = cv2.imread(path +'300_3/' + '%04d' %n +'.png')
#img = image.copy()

def pre(img):
    image = img[:,:,0]
    (thresh, blackAndWhiteImage) = cv2.threshold(image, 67, 255, cv2.THRESH_BINARY)
    @jit(nopython=True)
    def find_first(item, vec):
        """return the index of the first occurence of item in vec"""
        for i in range(len(vec)):
            if item == vec[i]:
                return i
        return -1
    
    window = blackAndWhiteImage
    xy = []
    for i in range(1,window.shape[0]-1)[::-1]:
         for j in range(1,window.shape[1]-1):
             if (int(window[i-1,j])-int(window[i,j]) !=0 or int(window[i+1,j])-int(window[i,j]) !=0 or int(window[i,j+1])-int(window[i,j]) !=0 or int(window[i,j-1])-int(window[i,j]) !=0):
                 xy.extend((j,i))
    
    data = np.array(xy).reshape((-1, 2))
    #edges = np.zeros((300,300), np.uint8)
    edges = np.zeros((600,600), np.uint8)
    for i,j in ((data)):
        edges[j,i] = 255
    xy = []
    for i in range(edges.shape[0])[::-1]:
       # for j in range(window.shape[0]):
        col = edges[i,:]
        j = find_first(255, col)
        if j != -1:
            while (col[j] == 255):
                xy.extend((j, i))
                j += 1
             
    data = np.array(xy).reshape((-1, 2))
    unique = (np.unique(data[:,1], axis=0))[::-1]
    data = data[data[:,1] <= unique[2]]
    P0=data[0]

    img = np.float32(img)
    img = cv2.normalize(img, None, 0.0, 1.0, cv2.NORM_MINMAX)
    crop = img[P0[1]-17:P0[1]+3,P0[0]-10:P0[0]+10]
    
    return crop

# =============================================================================
# def crop(img):
#     for i in range(img.shape[0]-1):
#         if(img[i,0,0]-img[i+1,0,0]>0.2):
#             lineNumber = i
#             break
#     #print(lineNumber)
#     for i in range(img.shape[0]-1):
#         if(img[lineNumber,i,0]-img[lineNumber,i+1,0]>0.2):
#             break
#     #print(i)
#     #crop = img[lineNumber-12:lineNumber+3,i-7:i+8] # img[lineNumber-27:lineNumber+3,i-15:i+15]
#     crop = img[lineNumber-17:lineNumber+3,i-10:i+10]
#     return crop
# =============================================================================


standardized = []
i = 0 #201
for image in data:
    #img = np.float32(image)
    #img = cv2.normalize(img, None, 0.0, 1.0, cv2.NORM_MINMAX) #uncomment these when using crop
    img = pre(image)
    for l in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[l,j,0] < 0.18:
                img[l,j] = 0
    img = img*255
    standardized.append(img)
    filename = savepath+'crop%04d' %i +'.png'
    cv2.imwrite(filename, img)
    i=i+1

# plot the result
image = cv2.imread('crop0079.png')
#image = cv2.imread('lines5074.png')
image = image /255
#image[20:30,0:30,:]=0 #[y,x]
plt.imshow(image, cmap='gray')
