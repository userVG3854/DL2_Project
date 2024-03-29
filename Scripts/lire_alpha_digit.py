import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import matplotlib.pyplot as plt
import scipy


binary_alpha_digits = loadmat('./Data/Data_Binary/binaryalphadigs.mat')


def lire_alpha_digit(*chars):
    binary_alpha_digits = scipy.io.loadmat('./Data/Data_Binary/binaryalphadigs.mat')
    class_labels = (
        np.array(binary_alpha_digits["classlabels"][0].tolist()).flatten().tolist()
    )

    result_list = []
    for char in chars:
        index = class_labels.index(char)
        data = binary_alpha_digits["dat"][index]
        char_list = [matrix.flatten() for matrix in data]
        result_list.extend(char_list)

    return np.array(result_list)

def lire_alpha_digit2(data,L):
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
        plt.imshow(image, cmap='viridis')
        plt.show()