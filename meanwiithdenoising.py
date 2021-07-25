import numpy as np    
import cv2    
import matplotlib.pyplot as plt
from sklearn.cluster import MeanShift, estimate_bandwidth


#Loading original image
originImg = cv2.imread('image.orig/331.jpg')
#Denoising
blur = cv2.GaussianBlur(originImg,(5,5),0)
# Shape of original image    
originShape = blur.shape


# Converting image into array of dimension [nb of pixels in originImage, 3]
# based on r g b intensities    
flatImg=np.reshape(blur, [-1, 3])


# Estimate bandwidth for meanshift algorithm    
bandwidth = estimate_bandwidth(flatImg, quantile=0.1, n_samples=100)    
ms = MeanShift(bandwidth = bandwidth, bin_seeding=True)

# Performing meanshift on flatImg    
ms.fit(flatImg)

# (r,g,b) vectors corresponding to the different clusters after meanshift    
labels=ms.labels_

# Remaining colors after meanshift    
cluster_centers = ms.cluster_centers_    

  

###
Shape = originImg.shape
flatImg1=np.reshape(originImg, [-1, 3])
   
bandwidth1 = estimate_bandwidth(flatImg, quantile=0.1, n_samples=100)    
ms = MeanShift(bandwidth = bandwidth1, bin_seeding=True)
  
ms.fit(flatImg1)
   
labels1=ms.labels_
 
cluster_centers1 = ms.cluster_centers_    
   
   

# Displaying segmented image with filtering    


segmentedImg1 = cluster_centers1[np.reshape(labels1, Shape[:2])]


# Displaying segmented image    


segmentedImg = cluster_centers[np.reshape(labels, originShape[:2])]

plt.subplot(2, 2, 1)
plt.imshow(cv2.cvtColor(originImg, cv2.COLOR_BGR2RGB))


plt.subplot(2, 2, 2)
plt.imshow(cv2.cvtColor(segmentedImg1.astype(np.uint8), cv2.COLOR_BGR2RGB))

plt.show()

plt.subplot(2, 2, 3)
plt.imshow(cv2.cvtColor(segmentedImg.astype(np.uint8), cv2.COLOR_BGR2RGB))

plt.show()
