"""Train and evaluate a CNN on Fashion-MNIST.

This script mirrors the Colab notebook and saves evaluation artifacts locally.
"""

from pathlib import Path
import json
import random

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix


SEED = 42
CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]
ARTIFACT_DIR = Path("artifacts")


def set_reproducible_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def load_and_prepare_data():
    """Load Fashion-MNIST, create a 50k/10k/10k split, and normalise images."""
    (x_full, y_full), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    x_train, y_train = x_full[:50_000], y_full[:50_000]
    x_val, y_val = x_full[50_000:], y_full[50_000:]

    def prepare(images):
        return np.expand_dims(images.astype("float32") / 255.0, axis=-1)

    return prepare(x_train), y_train, prepare(x_val), y_val, prepare(x_test), y_test


def build_model() -> tf.keras.Model:
    """Construct and compile the CNN."""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(28, 28, 1)),
        tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Dropout(0.30),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.40),
        tf.keras.layers.Dense(10, activation="softmax"),
    ], name="fashion_mnist_cnn")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def save_training_curves(history: tf.keras.callbacks.History) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.history["accuracy"], label="Training")
    axes[0].plot(history.history["val_accuracy"], label="Validation")
    axes[0].set(title="Model accuracy", xlabel="Epoch", ylabel="Accuracy")
    axes[0].legend()
    axes[1].plot(history.history["loss"], label="Training")
    axes[1].plot(history.history["val_loss"], label="Validation")
    axes[1].set(title="Model loss", xlabel="Epoch", ylabel="Loss")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(ARTIFACT_DIR / "training_curves.png", dpi=160)
    plt.close(fig)


def main() -> None:
    set_reproducible_seed()
    ARTIFACT_DIR.mkdir(exist_ok=True)
    x_train, y_train, x_val, y_val, x_test, y_test = load_and_prepare_data()
    print(f"Training: {x_train.shape}; validation: {x_val.shape}; test: {x_test.shape}")

    model = build_model()
    model.summary()
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=3, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5
        ),
    ]
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=15,
        batch_size=128,
        callbacks=callbacks,
        verbose=2,
    )
    save_training_curves(history)

    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    probabilities = model.predict(x_test, verbose=0)
    predictions = probabilities.argmax(axis=1)
    report = classification_report(
        y_test, predictions, target_names=CLASS_NAMES, output_dict=True
    )
    (ARTIFACT_DIR / "classification_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    matrix = confusion_matrix(y_test, predictions)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        matrix, annot=True, fmt="d", cmap="Blues", square=True,
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax,
    )
    ax.set(title="Fashion-MNIST confusion matrix", xlabel="Predicted", ylabel="True")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    fig.tight_layout()
    fig.savefig(ARTIFACT_DIR / "confusion_matrix.png", dpi=160)
    plt.close(fig)

    model.save(ARTIFACT_DIR / "fashion_mnist_cnn.keras")
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(classification_report(y_test, predictions, target_names=CLASS_NAMES))


if __name__ == "__main__":
    main()
