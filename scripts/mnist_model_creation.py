"""create a simple CNN classification model with keras"""
from tensorflow import keras


def main():
    (train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()

    # improves accuracy ~89% to ~93%
    train_images, test_images = train_images / 255.0, test_images / 255.0

    inputs = keras.Input(shape=(28, 28))
    x = keras.layers.Flatten()(inputs)
    x = keras.layers.Dropout(0.25)(x, training=True)
    outputs = keras.layers.Dense(10, activation="softmax")(x)
    # outputs = keras.layers.Dropout(0.25)(x, training=True)
    model = keras.Model(inputs, outputs)

    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    model.fit(train_images, train_labels, epochs=5)
    model.evaluate(test_images, test_labels)

    model.save("../app/predictors/mnist_dropout")


if __name__ == "__main__":
    main()
