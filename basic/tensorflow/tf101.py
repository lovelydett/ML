# tt
# 2021.9.17
# Try tensorflow2

import tensorflow as tf

def minist_demo():
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # define a model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax")
    ])

    # compile model
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=['accuracy'])

    # fit model
    model.fit(x_train, y_train, epochs=10)
    model.evaluate(x_test, y_test, verbose=2)


if __name__ == "__main__":
    minist_demo()
