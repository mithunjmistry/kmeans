#See the jupyter notebook in the repository for step by step implementation.

import numpy as np
import matplotlib.pyplot as plt

#points are data points
def initialize_centroids(points, k):
    centroids = points.copy()
    np.random.shuffle(centroids)
    return centroids[:k]

def closest_centroid(points, centroids):
    #return array with index of data to nearest centroid
    distances = np.sqrt(((points - centroids[:, np.newaxis])**2).sum(axis=2))
    return np.argmin(distances, axis=0)

def move_centroids(points, closest, centroids):
    #returns new centroids
    return np.array([points[closest==k].mean(axis=0) for k in range(centroids.shape[0])])

def own_kmeans(data, k):
    c = initialize_centroids(data, k)
    for i in range(0, 500):
        new_centroids = move_centroids(data, closest_centroid(data, c), c)
        if np.array_equal(new_centroids,c):
            print(closest_centroid(data, c))
            print("\n")
            print(i)
            print("\n")
            print(c)
            return new_centroids, closest_centroid(data, c)
        else:
            c = new_centroids

data = np.array([[1,1], [2,1], [4,3], [5,4]]) #choose your data
centroids, target = own_kmeans(data, 2)
plt.scatter(data[:,0], data[:,1], c=target,cmap='rainbow')
plt.scatter(centroids[:,0], centroids[:1], color="black", marker='*')
plt.show()