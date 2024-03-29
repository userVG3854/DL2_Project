import numpy as np
from Scripts.principal_DBN_alpha import DBN
from Scripts.principal_RBM_alpha import RBM
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from tqdm.auto import tqdm


class DNN2:
    def __init__(self, sizes):
        self.sizes = sizes
        self.dbn = DBN(sizes[:-1])
        self.W = np.random.normal(size=(sizes[-2][0], sizes[-1][0])).astype(np.float32)
        self.b = np.zeros(sizes[-1][0]).astype(np.float32)

    def pretrain(self, X, learning_rate, len_batch, n_epochs, verbose=False):
        self.dbn.train_DBN(X, learning_rate, len_batch, n_epochs, verbose)

    def entree_sortie_reseau(self, X):
        outputs = [X]
        for layer in self.dbn.rbm_layers:
            X = layer.entree_sortie_RBM(X)
            outputs.append(X)
        X = self.calcul_softmax(X @ self.W + self.b)
        outputs.append(X)
        return outputs

    def retropropagation(self, X, y, learning_rate, len_batch, n_epochs, verbose=False):
        n = X.shape[0]
        for i in range(n_epochs):
            np.random.shuffle(X)
            y = y.reshape(-1, 1)
            if len_batch == 0:  # add this check
                len_batch = n
            for i_batch in range(0, n, int(len_batch)):  # change len_batch to int
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





class DNN:
    def __init__(self, config_list):
        self.DBN = DBN(config_list[:-1])
        self.RBM_classif = RBM(*config_list[-1])

    def pretrain(self, X, n_epochs, learning_rate, batch_size, plot = True):       
        self.DBN.train(
            X=X,
            n_epochs=n_epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
            plot=plot,
        )

    def entree_sortie_reseau(self, X):
        out_list = [X]
        for rbm in self.DBN.RBM_list:
            sortie_RBM = rbm.entree_sortie_RBM(out_list[-1])
            out_list.append(sortie_RBM)

        sortie_DBN = out_list[-1]

        sortie_RBM = self.RBM_classif.entree_sortie_RBM(sortie_DBN)
        out_list.append(sortie_RBM)
        proba = self.RBM_classif.calcul_softmax(sortie_DBN)

        return out_list, proba

    def retropropagation(self, X, y, n_epochs, learning_rate, batch_size):
        X = X.copy()
        y = y.copy()
        loss_history = []
        acc_history = []
        with tqdm(range(n_epochs)) as pbar:
            for _ in pbar:
                X, y = shuffle(X, y)

                for batch in range(0, X.shape[0], batch_size):
                    X_batch = X[batch : min(batch + batch_size, X.shape[0])]
                    y_batch = y[batch : min(batch + batch_size, X.shape[0])]
                    tb = X_batch.shape[0]

                    out_list, _ = self.entree_sortie_reseau(X_batch)

                    y_onehot = np.eye(out_list[-1].shape[1])[y_batch]
                    dL_dxp_tilde = out_list[-1] - y_onehot
                    cp = dL_dxp_tilde

                    xp_moins_1 = out_list[-2]

                    dL_dWp = xp_moins_1.T @ cp
                    dL_dbp = cp.sum(axis=0)

                    self.RBM_classif.W -= (learning_rate / tb) * dL_dWp
                    self.RBM_classif.b -= (learning_rate / tb) * dL_dbp

                    W_plus_1 = self.RBM_classif.W
                    cp_plus_1 = cp
                    xp = xp_moins_1
                    for p in range(len(self.DBN.RBM_list) - 1, -1, -1):
                        xp_moins_1 = out_list[p]
                        cp = (cp_plus_1 @ W_plus_1.T) * (xp * (1 - xp))

                        dL_dWp = xp_moins_1.T @ cp
                        dL_dbp = cp.sum(axis=0)

                        self.DBN.RBM_list[p].W -= (learning_rate / tb) * dL_dWp
                        self.DBN.RBM_list[p].b -= (learning_rate / tb) * dL_dbp

                        W_plus_1 = self.DBN.RBM_list[p].W
                        cp_plus_1 = cp
                        xp = xp_moins_1

                out_list, proba = self.entree_sortie_reseau(X)
                loss = cross_entropy(proba, y)
                pred = proba.argmax(axis=1)
                acc = (pred == y).sum() / len(pred)
                loss_history.append(loss)
                acc_history.append(acc)
                pbar.set_description(f"loss {loss:.4f} - acc {acc:.3f} ")

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].plot(loss_history, label="Loss")
        axes[1].plot(acc_history, label="Accuracy")
        for ax in axes:
            ax.legend()
            ax.grid()
        plt.show()

    def test(self, X, y):
        _, proba = self.entree_sortie_reseau(X)
        loss = cross_entropy(proba, y)
        pred = proba.argmax(axis=1)
        acc = (pred == y).sum() / len(pred)
        print(f"loss {loss:.4f} - acc {acc:.3f} ")
        return loss, acc


def cross_entropy(proba, y):
    output = 0
    for i in range(proba.shape[0]):
        proba_i = proba[i]
        y_i = y[i]
        output -= np.log(proba_i[y_i])
    return output / proba.shape[0]