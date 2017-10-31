import numpy as np
from knn import *
import time
from scipy.misc import *

def w_matrix(data, distance, indices, Ks, a):

    n = len(data)

    weight_matrix = np.zeros([n, n])

    sigma2 = (a / n / Ks) * np.linalg.norm(distance)**2
    #print(len(distance), sigma2)

    if Ks==1:
        for i in range(n):
            #for j in range(n):
            j=indices[i][0]
            weight_matrix[i][j] = np.exp(-1 * (np.linalg.norm(data[i]- data[j])** 2) / sigma2)
    else:
        for i in range(n):
            #for j in range(n):
            for j in indices[i]:
                weight_matrix[i][j] = np.exp(-1 * (np.linalg.norm(data[i]- data[j])** 2) / sigma2)

    return weight_matrix, sigma2

def k0graph(X,distance,indices, a):

    W, sigma2 = w_matrix(X, distance, indices, 1, a)

    Vc = []
    n = len(W)

    x,y = np.where(W>0)

    #print(len(x), len(y))

    for i in range(len(x)):
        x_index, y_index = -1,-1
        for k in range(len(Vc)):
            if y[i] in Vc[k]:
                y_index = k
            if x[i] in Vc[k]:
                x_index = k


        if x_index < 0 and y_index < 0:
            Vc.append([x[i],y[i]])
        elif x_index >= 0 and y_index < 0:
            Vc[x_index].append(y[i])
        elif x_index < 0 and y_index >=0:
            Vc[y_index].append(x[i])
        elif x_index == y_index:
            continue
        else:
            Vc[x_index].extend(Vc[y_index])
            del Vc[y_index]


    return Vc
