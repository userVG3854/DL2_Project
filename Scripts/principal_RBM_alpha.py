import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import matplotlib.pyplot as plt
import copy

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class RBM:
    def __init__(self, p, q):
        self.a = np.zeros(p)
        self.b = np.zeros(q)
        self.W = np.random.normal(size=(p, q)) * np.sqrt(0.01)

    def init_RBM(self):
        # Initialisation des poids et biais
        self.a = np.zeros(self.W.shape[0])
        self.b = np.zeros(self.W.shape[1])
        self.W = np.random.normal(size=self.W.shape) * np.sqrt(0.01)

    def entree_sortie_RBM(self, V):
        # Calcul des sorties
        return sigmoid(V @ self.W + self.b)

    def sortie_entree_RBM(self, H):
        # Calcul des entrées
        return sigmoid(H @ self.W.T + self.a)

    def train_RBM(self, X, learning_rate, len_batch, n_epochs, verbose=False):
        weights = []
        losses = []

        for i in range(n_epochs):
            np.random.shuffle(X)
            n = X.shape[0]
            for i_batch in range(0, n, len_batch):
                X_batch = X[i_batch:min(i_batch + len_batch, n), :]
                t_batch_i = X_batch.shape[0]

                V0 = copy.deepcopy(X_batch)
                pH_V0 = self.entree_sortie_RBM(V0)
                H0 = (np.random.rand(t_batch_i, self.W.shape[1]) < pH_V0) * 1
                pV_H0 = self.sortie_entree_RBM(H0)
                V1 = (np.random.rand(t_batch_i, self.W.shape[0]) < pV_H0) * 1
                pH_V1 = self.entree_sortie_RBM(V1)

                da = np.sum(V0 - V1, axis=0)
                db = np.sum(pH_V0 - pH_V1, axis=0)
                dW = V0.T @ pH_V0 - V1.T @ pH_V1

                self.a += learning_rate * da
                self.b += learning_rate * db
                self.W += learning_rate * dW

                weights.append(np.mean(self.W))

            H = self.entree_sortie_RBM(X)
            X_rec = self.sortie_entree_RBM(H)
            loss = np.mean((X - X_rec) ** 2)
            losses.append(loss)
            if i % 10 == 0 and verbose:
                print("epoch " + str(i) + "/" + str(n_epochs) + " - loss : " + str(loss))

        plt.plot(losses)
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.title('Evolution of the loss through ' + str(n_epochs) + ' epochs')
        plt.show()
        print("Final loss:", losses[-1])

        plt.xlabel('epochs')
        plt.ylabel('mean elements of weight W')
        plt.plot(weights)
        plt.show()

    def generer_image_RBM(self, nb_images, nb_iter, size_img):
        images = []
        for i in range(nb_images):
            v = (np.random.rand(self.W.shape[0]) < 0.5) * 1
            for j in range(nb_iter):
                h = (np.random.rand(self.W.shape[1]) < self.entree_sortie_RBM(v)) * 1
                v = (np.random.rand(self.W.shape[0]) < self.sortie_entree_RBM(h)) * 1
            v = v.reshape(size_img)
            images.append(v)
        return images


