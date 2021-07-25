import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN

img = cv2.imread('image.orig/234.jpg')

Z = np.float32(img.reshape((-1,3)))
db = DBSCAN(eps=0.3, min_samples=100).fit(Z[:,:2])

plt.imshow(np.uint8(db.labels_.reshape(img.shape[:2])))
plt.show()