import numpy as np
from Scripts.principal_DBN_alpha import DBN
from Scripts.principal_RBM_alpha import RBM



class DNN:
    def __init__(self, sizes):
        self.sizes = sizes
        self.dbn = DBN(sizes[:-1])
        self.W = np.random.normal(size=(sizes[-2], sizes[-1])).astype(np.float32)
        self.b = np.zeros(sizes[-1]).astype(np.float32)

    def init_DNN(self):
        self.dbn.init_DBN()
        self.W = np.random.normal(size=self.W.shape).astype(np.float32)
        self.b = np.zeros(self.b.shape).astype(np.float32)

    def pretrain_DNN(self, X, learning_rate, len_batch, n_epochs, verbose=False):
        self.dbn.train_DBN(X, learning_rate, len_batch, n_epochs, verbose)

    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    def calcul_softmax(self, X):
        return self.softmax(X @ self.W + self.b)

    def entree_sortie_reseau(self, X):
        outputs = [X]
        for layer in self.dbn.rbm_layers:
            X = layer.entree_sortie_RBM(X)
            outputs.append(X)
        X = self.calcul_softmax(X)
        outputs.append(X)
        return outputs

    def retropropagation(self, X, y, learning_rate, len_batch, n_epochs, verbose=False):
        n = X.shape[0]
        for i in range(n_epochs):
            np.random.shuffle(X)
            y = y.reshape(-1, 1)
            for i_batch in range(0, n, len_batch):
                X_batch = X[i_batch:min(i_batch + len_batch, n), :]
                y_batch = y[i_batch:min(i_batch + len_batch, n), :]

                # Forward pass
                outputs = self.entree_sortie_reseau(X_batch)
                y_pred = outputs[-1]

                # Backward pass
                dW = X_batch.T @ (y_pred - y_batch)
                db = np.sum(y_pred - y_batch, axis=0)
                self.W -= learning_rate * dW
                self.b -= learning_rate * db

                # Update DBN weights
                for i, layer in enumerate(self.dbn.rbm_layers):
                    dW = outputs[i].T @ (outputs[i + 1] - outputs[i] * (1 - outputs[i])) @ layer.W.T
                    db = np.sum(outputs[i + 1] - outputs[i] * (1 - outputs[i]), axis=0)
                    layer.W -= learning_rate * dW
                    layer.b -= learning_rate * db

            # Compute cross-entropy
            cross_entropy = -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
            if i % 10 == 0 and verbose:
                print("epoch " + str(i) + "/" + str(n_epochs) + " - cross-entropy : " + str(cross_entropy))

    def test_DNN(self, X_test, y_test):
        y_pred = np.argmax(self.entree_sortie_reseau(X_test)[-1], axis=1)
        error_rate = np.mean(y_pred != y_test)
        return error_rate