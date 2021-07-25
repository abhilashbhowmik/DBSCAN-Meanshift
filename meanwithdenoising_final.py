#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  1 09:41:39 2021

@author: pratyaksh
"""

import numpy as np    
import cv2    
import matplotlib.pyplot as plt
from sklearn.cluster import MeanShift, estimate_bandwidth

originImg = cv2.imread('peppers.jpeg')
blur = cv2.medianBlur(originImg, 5) 
originShape = blur.shape
flatImg=np.reshape(blur, [-1, 3]) 
bandwidth = estimate_bandwidth(flatImg, quantile=0.1, n_samples=100)    
ms = MeanShift(bandwidth = bandwidth, bin_seeding=True)  
ms.fit(flatImg)  
labels=ms.labels_  
cluster_centers = ms.cluster_centers_    
Shape = originImg.shape
flatImg1=np.reshape(originImg, [-1, 3])
bandwidth1 = estimate_bandwidth(flatImg, quantile=0.1, n_samples=100)    
ms = MeanShift(bandwidth = bandwidth1, bin_seeding=True)
ms.fit(flatImg1)
labels1=ms.labels_
cluster_centers1 = ms.cluster_centers_   
segmentedImg1 = cluster_centers1[np.reshape(labels1, Shape[:2])]
segmentedImg = cluster_centers[np.reshape(labels, originShape[:2])]
t = segmentedImg1.astype(np.uint8)


plt.subplot(2, 2, 1)
plt.imshow(cv2.cvtColor(originImg, cv2.COLOR_BGR2RGB))
plt.title("ORIGINAL IMG")

plt.subplot(2, 2, 2)
plt.imshow(cv2.cvtColor(blur, cv2.COLOR_BGR2RGB))
plt.title("AFTER DENOISING")
plt.subplot(2, 2, 3)
plt.imshow(cv2.cvtColor(t, cv2.COLOR_BGR2RGB))
plt.title("SEGMENTED-IMG W/O DENOISING")


plt.subplot(2, 2, 4)
plt.imshow(cv2.cvtColor(segmentedImg.astype(np.uint8), cv2.COLOR_BGR2RGB))
plt.title("SEGMENTED-IMG W DENOISING")
plt.axis('off')
plt.show()
