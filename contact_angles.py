#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 14:43:11 2019

@author: alessandro
"""
#import cv2
import numpy as np
#from matplotlib import pyplot as plt
import os
# =============================================================================
# imageformat = ".png"
# path = '/crop/'
# imfilelist = [os.path.join(path,f) for f in os.listdir(path) if f.endswith(imageformat)]
# imfilelist.sort()
# =============================================================================

from math import atan, degrees, pi, tan, sin, sqrt, asin, cos, radians

def contact_angle(ry,h,a):
    if (h > ry):
        H = 2*ry-h
        b = ry-H
        R = sqrt(b**2+a**2)
        alpha = asin(b/R)
        alpha_deg = degrees(alpha)
        con_ang = alpha_deg+90
    else:
        b = ry-h
        R = sqrt(b**2+a**2)
        alpha = asin(b/R)
        alpha_deg = degrees(alpha)
        con_ang = 90-alpha_deg
       
    return con_ang


def ell_CA(ry,rx,a,h): #deve prendere h e fare il confronto come sopra
    if (rx==a):
        y = 90
    else:
        alpha = atan(-a/(rx**2*sqrt(1-a**2/rx**2)))
        y = abs(degrees(alpha))
    if (h > ry):
        y = 180-y
    return y
# =============================================================================
# del data
# import gc
# gc.collect()
# =============================================================================
# Function that return Shpere Parameters for a given height of the droplet (inside the plane) along z axis
def sphere(z):
    ry = 1
    H = 2+z # z has negative value
    b = abs(1+z)
    alpha = asin(b)
    a = cos(alpha)
    return ry,H,a

z_drop = np.linspace(-0.04,-1.9,201)
# Major Axis of the Ellipse
rx = np.linspace(1.,2.,201)

# Function that return Ellipse Parameters for a given height of the droplet (inside the plane) along z axis
def ell(z,rx):
    ry = 1
    H = 2+z # z has negative value
    b = abs(1+z)
    a = sqrt((rx**2)*(1-b**2))
    return ry,H,a

# Array that contain all shpere parameters
sphere_parameters = []
for i in range(len(z_drop)):
    sphere_parameters.append([sphere(z_drop[i])[0],sphere(z_drop[i])[1],sphere(z_drop[i])[2]])
sphere_parameters = np.array(sphere_parameters)

# Array that contains the calculated contact angles for the sphere
contact_angles_sphere = []   
for i in range(len(z_drop)):
    contact_angles_sphere.append(contact_angle(sphere_parameters[i,0],sphere_parameters[i,1],sphere_parameters[i,2]))

# Array that contain all ellipse parameters
ellipse_parameters = []
for i in range(len(z_drop)):
    ellipse_parameters.append([ell(z_drop[i],rx[i])[0],ell(z_drop[i],rx[i])[1],ell(z_drop[i],rx[i])[2]])
ellipse_parameters = np.array(ellipse_parameters)

# Array that contains the calculated contact angles for the ellipse
contact_angles_ellipse = []   
for i in range(len(z_drop)):
    #contact_angles_ellipse.append(contact_angle(ellipse_parameters[i,0],ellipse_parameters[i,1],ellipse_parameters[i,2]))
    contact_angles_ellipse.append(ell_CA(1,rx[i],ellipse_parameters[i,2],ellipse_parameters[i,1]))

contact_angles = contact_angles_sphere + contact_angles_ellipse

import pandas as pd
contact_angles = pd.DataFrame(data =contact_angles, columns=['Contact Angles'])
contact_angles= contact_angles/180
array = ["crop%04d" %i+".png" for i in range(402)]
data_x = pd.DataFrame(data=array, columns=['Filename'],dtype=str)
my_data = pd.concat([data_x,contact_angles],axis=1) # axis = 1 means columns