""" plot_3d.py: Plotting the data distribution into 3 dimensions.
    
    Quick tutorial on how to use kmeans with openCV.
    
    __author__: sabin lee
    __license__: MIT
    __version__: 0.1
"""
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Generate random 3D points
X = np.random.randint(25,50,(25,3))
Y = np.random.randint(60,85,(25,3))
Z = np.random.randint(100,150,(25,3))
L = np.vstack((X,Y,Z))
# convert to np.float32
L = np.float32(L)

K = 3
# define criteria and apply kmeans()
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
ret,label,center=cv.kmeans(L,K,None,criteria,10,cv.KMEANS_RANDOM_CENTERS)

# Now separate the data, Note the flatten()
A = L[label.ravel()==0]
B = L[label.ravel()==1]
C = L[label.ravel()==2]

# plot into 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(A[:,0], A[:,1], A[:,2], label='Cluster 1')
ax.scatter(B[:,0], B[:,1], B[:,2], label='Cluster 2')
ax.scatter(C[:,0], C[:,1], C[:,2], label='Cluster 3')
ax.scatter(center[:,0], center[:,1], center[:,2], s=80, c='y', marker='s', label='Centroids')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()