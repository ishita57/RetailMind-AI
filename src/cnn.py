"""Day 6: the CNN used for both Fashion-MNIST and the product images."""
from tensorflow import keras
from tensorflow.keras import layers


def build_cnn(input_shape, num_classes, augment=False):
    """conv -> pool -> conv -> pool -> dropout -> dense.

    Pixels go in as 0-255 numbers; the model rescales them itself.
    augment=True adds random flips/rotations/zoom (only active while training).
    """
    steps = [keras.Input(shape=input_shape)]
    if augment:
        steps += [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.05),
            layers.RandomZoom(0.1),
        ]
    steps += [
        layers.Rescaling(1.0 / 255),
        layers.Conv2D(32, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Dropout(0.25),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ]
    model = keras.Sequential(steps)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model
