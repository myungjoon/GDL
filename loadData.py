import numpy as np
from scipy.misc import *
import os, struct

def load_MNIST_test():
    path = './MNIST'
    fname_img = os.path.join(path, 't10k-images.idx3-ubyte')
    fname_lbl = os.path.join(path, 't10k-labels.idx1-ubyte')

    n_cluster = 10

    with open(fname_lbl, 'rb') as flbl:
        magic, num = struct.unpack(">II", flbl.read(8))
        labels = np.fromfile(flbl, dtype=np.int8)

    with open(fname_img, 'rb') as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        data = np.fromfile(fimg, dtype=np.uint8).reshape(len(labels), rows, cols)

    data = np.reshape(data, [len(data), 28 * 28])[:10000]
    return data, labels, n_cluster

def load_coil20():
    data = np.zeros((72*20,128,128))
    labels = np.zeros(72,dtype='i')
    n_cluster = 20

    for i in range(1,n_cluster):
        labels = np.concatenate((labels,np.full(72,i)),axis=0)

    for i in range(n_cluster):
        for j in range(72):
            index = i*72+j
            data[index]=imread("./coil-20/obj" + str(i+1) + "__" + str(j) + ".png","L")
    #print("number of data : ", len(labels))
    data = data.reshape(len(data), 128 * 128)
    return data, labels, n_cluster

def load_CMUPIE():
    import h5py
    f = h5py.File('./CMUPIE/CMU-PIE.h5', 'r')
    data = f['data'].value

    for i in range(len(data)):
        data[i][0] = np.rot90(data[i][0],3)
        imsave('./cmu2/' + str(i) + '.png', data[i][0])

    data = data.reshape([len(data), 32 * 32])
    labels = f['labels'].value
    labels = [int(x)-1 for x in labels]

    f.close()
    n_cluster = 68

    return data, labels, n_cluster
