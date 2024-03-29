import matplotlib.pyplot as plt
import numpy as np
import random
from sklearn.model_selection import train_test_split
from Scripts.principal_DNN_mnist import DNN
from Scripts.principal_DBN_alpha import DBN
from Scripts.principal_RBM_alpha import RBM
from Scripts.lire_mnist import lire_MNIST
from Scripts.lire_alpha_digit import lire_alpha_digit

def plot_grid(X, image_size):
    fig, axes = plt.subplots(2, 10, figsize=(10, 2.5))

    for ax in axes.ravel():
        index = random.randint(0, len(X) - 1)
        image = X[index].reshape(image_size)
        ax.imshow(image, cmap="inferno")
        ax.axis(False)

    plt.tight_layout()

    plt.show()

# Define a function to train and test the networks for a given configuration
def train_test(config, hparams, X_train, y_train, X_test, y_test, pretrain):
    dnn = DNN(config)
    X_train_small, _, y_train_small, _ = train_test_split(
        X_train, y_train, train_size=10_000
    )
    if pretrain:
        dnn.pretrain(
            X_train_small,
            hparams["n_epochs_RBM"],
            hparams["learning_rate"],
            hparams["batch_size"],
            plot=False,
        )
    dnn.retropropagation(
        X_train_small,
        y_train_small,
        hparams["n_epochs_retro"],
        hparams["learning_rate"],
        hparams["batch_size"],
    )
    loss, acc = dnn.test(X_test, y_test)
    return loss, acc

def experiment_RBM(*chars):
    print("Training RBM with characters:", chars)
    X = lire_alpha_digit(*chars)
    image_size = (20, 16)
    plot_grid(X, image_size=image_size)

    Q_LIST = [2, 5, 10, 50, 100, 500, 1000]
    HISTORY_LIST = []
    for q in Q_LIST:
        hparams = {
            "p": 320,
            "q": q,
            "n_epochs": 500,
            "learning_rate": 0.1,
            "batch_size": 10,
        }

        rbm = RBM(hparams["p"], hparams["q"])

        error_history = rbm.train_RBM(
            X,
            n_epochs=hparams["n_epochs"],
            learning_rate=hparams["learning_rate"],
            batch_size=hparams["batch_size"],
            plot=False,
        )
        HISTORY_LIST.append(error_history)

        Y = rbm.generer_image_RBM(20, 10)
        plot_grid(Y, image_size=image_size)

    for i, q in enumerate(Q_LIST):
        plt.plot(HISTORY_LIST[i], label=f"q={q}")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid()
    plt.yscale("log")
    plt.show()

def experiment_DBN(*chars, n_neurons):
    print("Training DBN with characters:", chars)
    X = lire_alpha_digit(*chars)
    image_size = (20, 16)
    plot_grid(X, image_size=image_size)

    CONFIG_LIST = [
        [[320, n_neurons], [n_neurons, n_neurons]],
        [[320, n_neurons], [n_neurons, n_neurons], [n_neurons, n_neurons]],
        [
            [320, n_neurons],
            [n_neurons, n_neurons],
            [n_neurons, n_neurons],
            [n_neurons, n_neurons],
        ],
        [
            [320, n_neurons],
            [n_neurons, n_neurons],
            [n_neurons, n_neurons],
            [n_neurons, n_neurons],
            [n_neurons, n_neurons],
        ],
    ]
    HISTORY_LIST = []
    for config in CONFIG_LIST:
        hparams = {
            "config": config,
            "n_epochs": 500,
            "learning_rate": 0.1,
            "batch_size": 10,
        }

        dbn = DBN(hparams["config"])

        error_history = dbn.train(
            X,
            n_epochs=hparams["n_epochs"],
            learning_rate=hparams["learning_rate"],
            batch_size=hparams["batch_size"],
            plot=False,
        )
        HISTORY_LIST.append(error_history)

        Y = dbn.generer_image(20, 10)
        plot_grid(Y, image_size=image_size)

    for i, config in enumerate(CONFIG_LIST):
        for j in range(len(HISTORY_LIST[i])):
            plt.plot(HISTORY_LIST[i][j], label=f"config={config} dbn{j}")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid()
    plt.yscale("log")
    plt.show()