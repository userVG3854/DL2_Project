import numpy as np
import matplotlib.pyplot as plt
import struct
import scipy

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

train_images_path = './Data/Data_MNIST/train-images.idx3-ubyte'
test_images_path = './Data/Data_MNIST/t10k-images.idx3-ubyte'
train_labels_path = './Data/Data_MNIST/train-labels.idx1-ubyte'
test_labels_path = './Data/Data_MNIST/t10k-labels.idx1-ubyte'

def load_mnist():
    X_mnist_train = load_idx3_ubyte(train_images_path)
    X_mnist_test = load_idx3_ubyte(test_images_path)
    y_mnist_train = load_idx1_ubyte(train_labels_path)
    y_mnist_test = load_idx1_ubyte(test_labels_path)
    return X_mnist_train, X_mnist_test, y_mnist_train, y_mnist_test

def display_images(images, size=(28, 28)):
    for image in images:
        image = image.reshape(size)
        plt.imshow(image, cmap='viridis')
        plt.show()

def lire_MNIST(split):
    data = scipy.io.loadmat("./Data/Data_MNIST/MNIST.mat")

    X_list = []
    y_list = []
    for i in range(10):
        X_i = data[split + str(i)]
        y_i = i * np.ones(X_i.shape[0])
        X_list.append(X_i)
        y_list.append(y_i)
    X = np.vstack(X_list)
    y = np.concatenate(y_list)

    X = X >= 128
    X = np.array(X, dtype=int)
    y = np.array(y, dtype=int)

    return X, y

def lire_MNIST2(split):
    X_mnist, X_test, y_mnist, y_test = load_mnist()

    if split == "train":
        X = X_mnist
        y = y_mnist
    elif split == "test":
        X = X_test
        y = y_test
    else:
        raise ValueError("split must be 'train' or 'test'")

    X = X >= 128
    X = np.array(X, dtype=int)
    y = np.array(y, dtype=int)

    # Reshape X to be a 1D array of size 784
    X = X.reshape(-1, 784)
    y = y.reshape(-1, 784)

    return X, y