import numpy as np
import cv2
import time

def my_kmeans(matrix, nclusters, include_pos, meanspp, weight, start):
    #define array to hold means (centres of clusters)
    means = np.zeros((nclusters, 3 + include_pos*2))

    #define array to track the means each vector is closest to 
    members = np.zeros((matrix.shape[0], matrix.shape[1]))
    #distance weighting
    
    #append indexes to matrix if required
    new_matrix = 0
    if(include_pos == 1):
        new_matrix = np.zeros((matrix.shape[0], matrix.shape[1], 5))
        new_matrix[:,:,0:3] = matrix    
        new_matrix[:,:,3] = weight * np.tile(np.arange(matrix.shape[0]), (matrix.shape[1], 1)).T
        new_matrix[:,:,4] = weight  * np.tile(np.arange(matrix.shape[1]), (matrix.shape[0], 1))
        matrix = new_matrix
    
    #initialise means either randomly or with k-means++
    if(meanspp == 0):
        for i in range(0, nclusters):
            a = np.random.randint(matrix.shape[0])
            b = np.random.randint(matrix.shape[1])
            means[i] = matrix[a, b]

    #boolean to see if change was made throughout entire image
    change = 1
   
    while(change == 1):
        change = 0
        #variables to be used for calculating new means
        means_total = np.zeros(means.shape)
        number = np.zeros(nclusters)

        for i in range(0, matrix.shape[0]):
            for j in range(0, matrix.shape[1]):
                
                current = matrix[i, j]
                #closest mean for this vector found from members
                closest_mean = members[i, j]
                #calculate distance to this closest mean
                closest_distance = np.linalg.norm(means[int(closest_mean)] - current)

                #calculate distance between current vector and every mean, update and indicate
                #a change has occurred if necessary
                for k in range(0, nclusters):
                    distance = np.linalg.norm(current - means[k])
                    if(distance < closest_distance):
                        members[i, j] = k
                        closest_distance = distance
                        change = 1
                        
                #update variables keeping track of sum of clusters and number of points
                means_total[int(members[i,j])] += current
                number[int(members[i,j])] += 1

        #recalculate means after completing iterations
        for k in range(0, nclusters):
            if(number[k]!= 0):
                means[k] = means_total[k]/number[k]
        
    #once convergence has been completed, lists of coordinates are returned
    coordinates = []
    means = np.uint8(means)
    for i in range(0, nclusters):
        coordinates.append([])

    for i in range(0, matrix.shape[0]):
        for j in range(0, matrix.shape[1]):
            coordinates[int(members[i,j])].append((i, j))
   
    return means, coordinates

#code to load image
start = time.time()
#for command line arguments
image_name = 'peppers.png'
nclusters = 3
include_pos = 0
meanspp = 0
weight = 0

#read in image and convert to LAB domain
image = cv2.imread(image_name)
image = np.uint8(image)
image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

#create string for output image

outstring = "w=" + str(weight) + str(include_pos) + str(meanspp) + str(nclusters) + image_name 
print("Running ", outstring)

#use my_kmeans to find the means and the lists of pixels associated with them
means, coordinates = my_kmeans(image, nclusters,include_pos, meanspp, weight, start)
segmented = np.zeros(image.shape)

#iterate through coordinates list of lists. The list each coordinate is stored in corresponds with its mean
for i in range(0, len(coordinates)):
    for j in range(0, len(coordinates[i])):
        segmented[coordinates[i][j]] = means[i][0:3]

segmented = np.uint8(segmented)
segmented = cv2.cvtColor(segmented, cv2.COLOR_LAB2BGR)
#cv2.imwrite(outstring, segmented)
cv2.imshow(outstring, segmented)
time_taken = time.time() - start
time_str = "Time taken was:" + str(time_taken) + "\n"
print(time_str)
print(outstring,"complete\n")
cv2.waitKey()