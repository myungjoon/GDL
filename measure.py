from sklearn import metrics
from munkres import Munkres, print_matrix, make_cost_matrix
import numpy as np


def NMI(labels, labels_pred):
    nmi = metrics.normalized_mutual_info_score(labels, labels_pred)
    return nmi

def ACC(labels, labels_pred):

    m = Munkres()
    n = len(set(labels_pred))
    c_mat = np.zeros([n,n],dtype="i")
    for i in range(len(labels)):
        c_mat[labels[i],labels_pred[i]] += 1
    cost_matrix = make_cost_matrix(c_mat, lambda cost:np.amax(c_mat) - cost)
    indexes = m.compute(cost_matrix)
    total = 0
    for row, col in indexes:
        value = c_mat[row][col]
        total += value
    accuracy = total/len(labels)
    return accuracy
