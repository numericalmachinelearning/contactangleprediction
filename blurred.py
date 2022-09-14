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
path = '/home/alessandro/droplet_database/lines/lines_20/'
savepath = '/home/alessandro/droplet_database/lines/20x20_test/'
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
    



img = cv2.imread('/home/alessandro/droplet_database/lines/lines_20/lines0178.png') #blur00129.png, lines2129.png
img = cv2.GaussianBlur(img,(3,3),0)
img[18:,:]=0
cv2.imwrite('test.png', img)
plt.imshow(img) 

img = cv2.imread('/home/alessandro/droplet_database/lines/20x20_test/blur20163.png') #blur00129.png, lines2129.png

image = cv2.imread('test.png') #blur00129.png, lines2129.png
plt.imshow(image) 
#image = cv2.imread('/home/alessandro/droplet_database/lines/1000_2/crop0020.png')
#image = cv2.imread('/home/alessandro/droplet_database/pic/4.png')
image = image /255
#img = cv2.imread('crop0020.png')
#img = cv2.blur(img,(7,7))
img[18:,:]=0
plt.imshow(img) 
img = cv2.imread('/home/alessandro/droplet_database/lines/20x20_test/blur20163.png') #blur00129.png, lines2129.png
img = img /255
img[1:,:]=0
#image[20:30,0:30,:]=0 #[y,x]
plt.imshow(img, cmap='gray')
