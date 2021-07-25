import numpy as np    
import cv2    
import matplotlib.pyplot as plt
from sklearn.cluster import MeanShift, estimate_bandwidth


#Loading original image
originImg = cv2.imread('peppers.png')

# Shape of original image    
originShape = originImg.shape


# Converting image into array of dimension [nb of pixels in originImage, 3]
# based on r g b intensities    
flatImg=np.reshape(originImg, [-1, 3])


# Estimate bandwidth for meanshift algorithm    
bandwidth = estimate_bandwidth(flatImg, quantile=0.1, n_samples=100)    
ms = MeanShift(bandwidth = bandwidth, bin_seeding=True)

# Performing meanshift on flatImg    
ms.fit(flatImg)

# (r,g,b) vectors corresponding to the different clusters after meanshift    
labels=ms.labels_

# Remaining colors after meanshift    
cluster_centers = ms.cluster_centers_    

# Finding and diplaying the number of clusters    
labels_unique = np.unique(labels)    
    
   

# Displaying segmented image    


segmentedImg = cluster_centers[np.reshape(labels, originShape[:2])]

#plt.figure(2)
#plt.subplot(2, 1, 1)
a=cv2.cvtColor(originImg, cv2.COLOR_BGR2RGB)
cv2.imshow('A',a)
#plt.axis('off')

# plt.subplot(2, 1, 2)
# plt.imshow(np.reshape(labels, [rows, cols]))
# plt.axis('off')

#plt.subplot(2, 1, 2)
b=cv2.cvtColor(segmentedImg.astype(np.uint8), cv2.COLOR_BGR2RGB)
cv2.imshow('B',b)
#plt.axis('off')
#plt.show()
