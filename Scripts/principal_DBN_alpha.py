import numpy as np
from Scripts.principal_RBM_alpha import RBM

class DBN:
    def __init__(self, config_list):
        self.RBM_list = [RBM(*config) for config in config_list]

    def train(self, X, n_epochs, learning_rate, batch_size, plot = True):
        X = X.copy()
        error_history = []
        for rbm in self.RBM_list:
            history = rbm.train_RBM(X, n_epochs, learning_rate, batch_size, plot=plot)
            X = rbm.entree_sortie_RBM(X)
            error_history.append(history)
        return error_history

    def generer_image(self, n_images, n_gibbs):
        output = self.RBM_list[-1].generer_image_RBM(n_images, n_gibbs)

        for rbm in self.RBM_list[len(self.RBM_list) - 2 :: -1]:
            for i in range(len(output)):
                v = output[i]
                temp = rbm.sortie_entree_RBM(v)
                output[i] = (np.random.random(temp.shape[0]) < temp) * 1

        return output
