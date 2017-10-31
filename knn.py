# k-nearest neighbors algorithm
# input : set of samples X
# output : the array of length k
import numpy as np
from sklearn.neighbors import NearestNeighbors

def knn(k, X):
    neigh = NearestNeighbors(k+1,metric='euclidean', n_jobs=-1).fit(X)
    kneighbors = neigh.kneighbors(X,k+1,)
    distance = np.array(kneighbors[0][:,1:])
    indices = np.array(kneighbors[1][:,1:])
    return distance, indices
