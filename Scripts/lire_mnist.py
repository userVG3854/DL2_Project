import tensorflow as tf

def load_mnist():
    # Load the MNIST dataset
    mnist = tf.keras.datasets.mnist
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # Normalize the data
    X_train, X_test = X_train / 255.0, X_test / 255.0

    return X_train, y_train, X_test, y_test
