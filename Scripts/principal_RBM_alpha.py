import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from tqdm.auto import tqdm

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class RBM:
    def __init__(self, p, q):
        self.p = p
        self.q = q
        self.a = np.zeros(p)
        self.b = np.zeros(q)
        self.W = np.random.normal(loc=0, scale=np.sqrt(0.01), size=(p, q))

    def entree_sortie_RBM(self, X):
        out = sigmoid(X @ self.W + self.b)
        return out

    def sortie_entree_RBM(self, H):
        in_ = sigmoid(H @ self.W.T + self.a)
        return in_
    
    def calcul_softmax(self, X):
        out = X @ self.W + self.b

        proba = np.exp(out) / np.sum(np.exp(out), axis=1, keepdims=True)

        return proba

    def train_RBM(self, X, n_epochs, learning_rate, batch_size, plot = True):
        error_history = []
        with tqdm(range(n_epochs)) as pbar:
            for _ in pbar:
                X_copy = X.copy()
                np.random.shuffle(X_copy)

                for batch in range(0, X.shape[0], batch_size):
                    X_batch = X_copy[batch : min(batch + batch_size, X.shape[0])]
                    tb = X_batch.shape[0]

                    v0 = X_batch
                    p_h_v0 = self.entree_sortie_RBM(v0)

                    # Sample according to a Bernoulli
                    h0 = (np.random.random((tb, self.q)) < p_h_v0) * 1
                    p_v1_h0 = self.sortie_entree_RBM(h0)
                    v1 = (np.random.random((tb, self.p)) < p_v1_h0) * 1
                    p_h_v1 = self.entree_sortie_RBM(v1)

                    # Calculate the gradient
                    grad_a = np.sum(v0 - v1, axis=0)  # size p
                    grad_b = np.sum(p_h_v0 - p_h_v1, axis=0)  # size q
                    grad_W = v0.T @ p_h_v0 - v1.T @ p_h_v1

                    # Update the parameters
                    self.a += (learning_rate / tb) * grad_a
                    self.b += (learning_rate / tb) * grad_b
                    self.W += (learning_rate / tb) * grad_W

                H = self.entree_sortie_RBM(X_copy)
                X_reconstruction = self.sortie_entree_RBM(H)

                error = np.mean((X_copy - X_reconstruction) ** 2)
                error_history.append(error)
                pbar.set_description(f"error {error:.4f} ")
        if plot:
            plt.plot(error_history)
            plt.grid()
            plt.show()

        return error_history

    def generer_image_RBM(self, nb_images, nb_iter):
        images = []
        for i in range(nb_images):
            v = (np.random.rand(self.W.shape[0]) < 0.5) * 1
            for j in range(nb_iter):
                h = (np.random.rand(self.W.shape[1]) < self.entree_sortie_RBM(v)) * 1
                v = (np.random.rand(self.W.shape[0]) < self.sortie_entree_RBM(h)) * 1
            images.append(v)
        return images


