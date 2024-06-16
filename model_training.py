import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# Parameters
img_size = 224
categories = ["Cephalometric", "Anteroposterior", "OPG"]


def create_model():
    model = Sequential(
        [
            Conv2D(32, (3, 3), activation="relu", input_shape=(img_size, img_size, 1)),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation="relu"),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(128, activation="relu"),
            Dense(len(categories), activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )
    model.summary()
    return model


def train_model():
    X_train = np.load("X_train.npy")
    y_train = np.load("y_train.npy")
    X_val = np.load("X_val.npy")
    y_val = np.load("y_val.npy")

    model = create_model()

    checkpoint = ModelCheckpoint(
        "best_model.keras", monitor="val_loss", save_best_only=True, mode="min"
    )
    early_stopping = EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )

    model.fit(
        X_train,
        y_train,
        epochs=50,
        validation_data=(X_val, y_val),
        callbacks=[checkpoint, early_stopping],
    )

    model.save("Xry_Classifier.keras")


if __name__ == "__main__":
    train_model()
