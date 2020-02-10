# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 21:16:32 2019

@author: Mohammadmahdi
"""

import cv2 as cv 
import matplotlib.pyplot as plt 
import numpy as np
from numpy import linalg as la
from fcmeans import FCM

img = cv.imread('1.jpg')
img = cv.resize(img,(256,256))
imgg = cv.cvtColor(img, cv.COLOR_BGR2GRAY)    # gray level image 
plt.subplot(231)
plt.imshow(imgg,cmap="gray"),plt.xticks([]),plt.yticks([]),plt.title('Gray level image ')
plt.subplot(232)
plt.imshow(img[:,:,::-1]),plt.xticks([]),plt.yticks([]),plt.title('RGB image ')
plt.subplot(233)
plt.imshow(img[:,:,0],cmap="gray"),plt.xticks([]),plt.yticks([]),plt.title('Gray level image B ')
plt.subplot(234)
plt.imshow(img[:,:,1],cmap="gray"),plt.xticks([]),plt.yticks([]),plt.title('Gray level image G ')
plt.subplot(235)
plt.imshow(img[:,:,2],cmap="gray"),plt.xticks([]),plt.yticks([]),plt.title('Gray level image R ')


#%% Initialization 

#m = np.array([[190,70,30,230,203],[48,210,48,100,113],[0,50,210,220,3]])  # flower mean
m = np.array([[237,34,63,163,255],[28,200,72,73,127],[36,76,204,164,39]])

z  = np.zeros((3,256*256))
X  = np.transpose(np.zeros((3,256**2)))
for i in range(3):
    z[i,:] = np.reshape(img[:,:,2-i],(1,256**2))
    X[:,i] = np.reshape(img[:,:,2-i],(1,256**2))
def kmeans(z,m):
    _,n = np.shape(m)
    dist = np.zeros((1,n))
    for i in range(n):
        dist[0,i] = la.norm(z - m[:,i])
    k = np.argmin(dist)
    m[:,k] = (m[:,k] + z)/2
    return (k,m)

for i in range(256**2):
    if (z[0,i] != 0):
        z[:,i],m = kmeans(z[:,i],m)


def mapper(z):
    if (z==0):
        z = 32
    elif (z==1):
        z = 64
    elif (z==2):
        z = 128
    elif (z==3):
        z = 191
    elif (z==4):
        z = 255
    return z

for i in range(256**2):
    z[0,i] = mapper(z[0,i])

Z = np.reshape(z[0,:],(256,256))
    
plt.subplot(236)
plt.imshow(Z,cmap="gray"),plt.xticks([]),plt.yticks([]),plt.title('Clustred image based on color')

def finder(Z):
    a0 = np.where(Z==32)
    a1 = np.where(Z==64)
    a2 = np.where(Z==128)
    a3 = np.where(Z==191)
    a4 = np.where(Z==255)
    return (a0,a1,a2,a3,a4)  

_,a1,a2,a3,a4 = finder(Z)  
    
data = (a1,a2,a3,a4)
colors = ("green","blue","magenta","yellow")
groups = ("Green", "Blue", "Magenta","Orange")

# Create plot
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

for data, color, group in zip(data, colors, groups):
    x,y = data
    ax.scatter(x, y, alpha=0.8, c=color, edgecolors='none', s=30, label=group)

plt.title('Scatter plot of clusters')
plt.legend(loc=1)
plt.show()




#%% testing FCM

n_bins = 5  # use 5 bins for calibration_curve as we have 5 clusters here
centers = [(237,28,36), (34,177,76), (63,72,204),(163,73,164),(255,127,39)]

# fit the fuzzy-c-means
fcm = FCM(n_clusters=5)
fcm.fit(X)

# outputs
fcm_centers = fcm.centers
fcm_labels  = fcm.u.argmax(axis=1)

A = np.zeros((256**2,))

for i in range(256**2):
    A[i] = mapper(fcm_labels[i])
A = np.reshape(A,(256,256))    


a0,_,a2,a3,a4 = finder(A) 

data = (a0,a2,a3,a4)
colors = ("red","blue","magenta","yellow")
groups = ("Red", "Blue", "Magenta","Orange")

# Create plot
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

for data, color, group in zip(data, colors, groups):
    x,y = data
    ax.scatter(x, y, alpha=0.8, c=color, edgecolors='none', s=30, label=group)

plt.title('Scatter plot of clusters')
plt.legend(loc=1)
plt.show()
