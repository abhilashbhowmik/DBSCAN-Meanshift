import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
img= cv2.imread('peppers.jpeg') 
img1 = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
blur = median = cv2.medianBlur(img1, 5) 


feature_image=np.reshape(blur, [-1, 3])
rows, cols , chs = blur.shape

db = DBSCAN(eps=1.2, min_samples=10, metric = 'euclidean',algorithm ='auto')
db.fit(feature_image)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

segmented = np.reshape(labels, [rows, cols])
image = segmented.astype(np.uint8)

print(image)

#neigh = NearestNeighbors(n_neighbors=2)
#nbrs = neigh.fit(image)
#distances, indices = nbrs.kneighbors(image)

#distances = np.sort(distances, axis=0)
#distances = distances[:,1]
#plt.plot(distances)

#m = DBSCAN(eps=0.3, min_samples=5)
#m.fit(image)

#clusters = m.labels_

#colors = ['royalblue', 'maroon', 'forestgreen', 'mediumorchid', 'tan', 'deeppink', 'olive', 'goldenrod', 'lightcyan', 'navy']
#vectorizer = np.vectorize(lambda x: colors[x % len(colors)])

#plt.scatter(X[:,0], X[:,1], c=vectorizer(clusters))

plt.figure(2)
plt.subplot(2, 2, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.subplot(2, 2, 2)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.subplot(2, 2, 3)
plt.imshow(image)
plt.axis('off')
plt.show()