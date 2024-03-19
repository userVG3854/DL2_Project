import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import matplotlib.pyplot as plt
import struct


def load_idx3_ubyte(file_path):
    with open(file_path, 'rb') as f:
        magic, num_images = struct.unpack(">II", f.read(8))
        rows, cols = struct.unpack(">II", f.read(8))
        images = np.fromfile(f, dtype=np.uint8).reshape(num_images, rows, cols)
    return images
def load_idx1_ubyte(file_path):
    with open(file_path, 'rb') as f:
        magic, num_items = struct.unpack(">II", f.read(8))
        labels = np.fromfile(f, dtype=np.uint8)
    return labels


train_images_path = './Data/train-images.idx3-ubyte'
test_images_path = './Data/t10k-images.idx3-ubyte'
train_labels_path = './Data/train-labels.idx1-ubyte'
test_labels_path = './Data/t10k-labels.idx1-ubyte'

#########

X_mnist_train = load_idx3_ubyte(train_images_path)
X_mnist_test = load_idx3_ubyte(test_images_path)
y_mnist_train = load_idx1_ubyte(train_labels_path)
y_mnist_test = load_idx1_ubyte(test_labels_path)

binary_alpha_digits = loadmat('./Data/binaryalphadigs.mat')

def lire_alpha_digit(data,L):
    X=data['dat'][L[0]]
    for i in range(1,len(L)) :
        X_bis=data['dat'][L[i]]
        X=np.concatenate((X,X_bis),axis=0)
    n=X.shape[0]
    X=np.concatenate(X).reshape((n,320))
    return X

def display_images(images, size):
    for image in images:
        image = image.reshape(size)
        plt.imshow(image, cmap='gray')
        plt.show()