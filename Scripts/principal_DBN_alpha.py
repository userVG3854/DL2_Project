import numpy as np
from Scripts.principal_RBM_alpha import RBM

class DBN:
    def __init__(self, sizes):
        self.sizes = sizes
        self.rbm_layers = []
        for i in range(len(sizes) - 1):
            self.rbm_layers.append(RBM(sizes[i], sizes[i + 1]))

    def init_DBN(self):
        for layer in self.rbm_layers:
            layer.init_RBM()

    def train_DBN(self, X, learning_rate, len_batch, n_epochs, verbose=False):
        for layer in self.rbm_layers:
            layer.train_RBM(X, learning_rate, len_batch, n_epochs, verbose)
            X = layer.entree_sortie_RBM(X)

    def generer_image_DBN(self, nb_images, nb_iter, size_img):
        images = []
        for _ in range(nb_images):
            v = (np.random.rand(self.rbm_layers[0].W.shape[0]) < 0.5) * 1
            for layer in self.rbm_layers:
                for _ in range(nb_iter):
                    h = (np.random.rand(layer.W.shape[1]) < layer.entree_sortie_RBM(v)) * 1
                    v = (np.random.rand(layer.W.shape[0]) < layer.sortie_entree_RBM(h)) * 1
            v = v.reshape(size_img)
            images.append(v)
        return images
