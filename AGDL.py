import numpy as np
from knngraph import *
from loadData import *
from measure import *



def getAffinityMaxtrix(Vc,W):
    nc = len(Vc)

    print(nc)

    affinity = np.zeros([nc,nc])

    for i in range(nc):
        for j in range(i+1,nc):
            ij = np.ix_(Vc[i],Vc[j])
            ji = np.ix_(Vc[j],Vc[i])

            W_ij, W_ji = W[ij], W[ji]
            Ci, Cj = len(Vc[i]),len(Vc[j])

            ones_i = np.ones((Ci,1))
            ones_j = np.ones((Cj,1))
            affinity[i][j] = (1/Ci**2)*np.transpose(ones_i).dot(W_ij).dot(W_ji).dot(ones_i) + (1/Cj**2)*np.transpose(ones_j).dot(W_ji).dot(W_ij).dot(ones_j)
            affinity[j][i] = affinity[i][j]
    return affinity

def getAffinityBtwCluster(C1, C2, W):


    ij = np.ix_(C1, C2)
    ji = np.ix_(C2, C1)

    W_ij, W_ji = W[ij], W[ji]
    Ci, Cj = len(C1), len(C2)

    ones_i = np.ones((Ci, 1))
    ones_j = np.ones((Cj, 1))
    affinity = (1/Ci**2)*np.transpose(ones_i).dot(W_ij).dot(W_ji).dot(ones_i) + (1/Cj**2)*np.transpose(ones_j).dot(W_ji).dot(W_ij).dot(ones_j)
    #print(affinity)
    return affinity[0,0]

def getNeighbor(Vc, Kc, W):
    Ns, As = [], []
    #time1 = time.time()
    print("affinity")
    A = getAffinityMaxtrix(Vc, W)
    #np.save('A_CIFAR_5000.npy',A)
    #A = np.load('A_MNIST.npy')

    for i in range(len(A)):
        As.append([x for x in sorted(list(A[i]))[-1 * Kc:] if x > 0])
        #As.append([x for x in sorted(list(A[i]))[-1 * Kc:]])
        n = len(As[i])
        if n==0:
            Ns.append([])
        else:
            Ns.append(A[i].argsort()[-1*n:].tolist())

    return Ns,As


def AGDL(data, targetClusterNum, Ks, Kc):
    print("data length : ", len(data))
    distance, indices = knn(Ks, data)

    cluster = k0graph(data, distance, indices)
    length = 0
    for i in range(len(cluster)):
        length += len(cluster[i])

    print("data before clustering : ", length)
    W = w_matrix(data, distance, indices, Ks)
    print("neighbor")
    neighborSet, affinitySet = getNeighbor(cluster, Kc, W)
    currentClusterNum = len(cluster)

    print("After k0 clustering : ", currentClusterNum)
    while currentClusterNum > targetClusterNum:

        max_affinity = 0
        max_index1 = 0
        max_index2 = 0
        for i in range(len(neighborSet)):
            if len(neighborSet[i])==0:
                continue
            aff = max(affinitySet[i])
            if aff > max_affinity:
                j = int(neighborSet[i][affinitySet[i].index(aff)])
                max_affinity = aff

                if i < j:
                    max_index1 = i
                    max_index2 = j
                else:
                    max_index1 = j
                    max_index2 = i

        if max_index1 == max_index2:
            print("index alias")
            print(affinitySet)
            break


        #merge two cluster
        cluster[max_index1].extend(cluster[max_index2])
        cluster[max_index2] = []


        if max_index2 in neighborSet[max_index1]:
            p = neighborSet[max_index1].index(max_index2)
            del neighborSet[max_index1][p]
            #del affinitySet[max_index1][index]
        if max_index1 in neighborSet[max_index2]:
            p = neighborSet[max_index2].index(max_index1)
            del neighborSet[max_index2][p]
            #del affinitySet[max_index2][index]


        for i in range(len(neighborSet)):
            if i==max_index1 or i==max_index2:
                continue


            if max_index1 in neighborSet[i]:
                aff_update = getAffinityBtwCluster(cluster[i], cluster[max_index1], W)

                p = neighborSet[i].index(max_index1)
                affinitySet[i][p] = aff_update # fix the affinity values

            if max_index2 in neighborSet[i]:

                p = neighborSet[i].index(max_index2)
                del neighborSet[i][p]
                del affinitySet[i][p]
                #print("delete")

                if max_index1 not in neighborSet[i]:
                    aff_update = getAffinityBtwCluster(cluster[i], cluster[max_index1], W)
                    neighborSet[i].append(max_index1)
                    affinitySet[i].append(aff_update)
                    #print("append")

        neighborSet[max_index1].extend(neighborSet[max_index2])
        neighborSet[max_index1] = list(set(neighborSet[max_index1]))

        affinitySet[max_index1] = []

        neighborSet[max_index2] = []
        affinitySet[max_index2] = []

        # Fine the Kc-nearest clusters for Cab

        for i in range(len(neighborSet[max_index1])):
            target_index = neighborSet[max_index1][i]
            newAffinity = getAffinityBtwCluster(cluster[target_index], cluster[max_index1], W)
            affinitySet[max_index1].append(newAffinity)

        #print(len(affinitySet[max_index1]), len(neighborSet[max_index1]))

        if len(affinitySet[max_index1]) > Kc:
            index = np.argsort(affinitySet[max_index1])
            new_neighbor = []
            new_affinity = []
            for j in range(Kc):
                new_neighbor.append(neighborSet[max_index1][index[-1*j]])
                new_affinity.append(affinitySet[max_index1][index[-1*j]])

            #print(new_neighbor, new_affinity)
            neighborSet[max_index1] = new_neighbor
            affinitySet[max_index1] = new_affinity
        #print(len(neighborSet[max_index1]), len(affinitySet[max_index1]))

        currentClusterNum = currentClusterNum - 1


    reduced_cluster = []
    for i in range(len(cluster)):
        if len(cluster[i]) != 0:
            reduced_cluster.append(cluster[i])
    #print("Cluster : ", Cluster)
    length = 0
    for i in range(len(reduced_cluster)):
        length += len(reduced_cluster[i])
    print("Data number, Cluster number : ", length, len(reduced_cluster))

    return reduced_cluster



if __name__ == '__main__':
    data, labels, CLUSTER_NUMBER = load_coil20()
    n = np.arange(-2,2.1,0.5)

    for a in n:
        print(n)
        Ks = 20
        Kc = 10
        cluster = AGDL(data, CLUSTER_NUMBER, Ks, Kc)

        labels_pred = np.zeros(len(labels), dtype='i')

        for i in range(len(cluster)):
            for j in range(len(cluster[i])):
                labels_pred[cluster[i][j]] = i

        #ACC_sf.append(ACC(labels, labels_pred))
        print(ACC(labels, labels_pred))

