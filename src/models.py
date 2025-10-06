"""CNN model architectures and training callbacks for pneumonia detection."""

from pathlib import Path
from typing import Dict, Optional

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import (
    VGG16,
    ResNet50,
    InceptionV3,
    DenseNet121,
    EfficientNetB0,
)


class CNNModels:
    """Collection of CNN builders and shared training utilities."""

    @staticmethod
    def custom_cnn(input_shape=(224, 224, 3), num_classes=1):
        """Build a custom CNN architecture."""
        activation = "sigmoid" if num_classes == 1 else "softmax"
        loss = "binary_crossentropy" if num_classes == 1 else "categorical_crossentropy"

        model = models.Sequential(
            [
                layers.Input(shape=input_shape),
                # Block 1
                layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
                layers.BatchNormalization(),
                layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
                layers.BatchNormalization(),
                layers.MaxPooling2D((2, 2)),
                layers.Dropout(0.25),
                # Block 2
                layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
                layers.BatchNormalization(),
                layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
                layers.BatchNormalization(),
                layers.MaxPooling2D((2, 2)),
                layers.Dropout(0.25),
                # Block 3
                layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
                layers.BatchNormalization(),
                layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
                layers.BatchNormalization(),
                layers.MaxPooling2D((2, 2)),
                layers.Dropout(0.25),
                # Block 4
                layers.Conv2D(256, (3, 3), activation="relu", padding="same"),
                layers.BatchNormalization(),
                layers.Conv2D(256, (3, 3), activation="relu", padding="same"),
                layers.BatchNormalization(),
                layers.MaxPooling2D((2, 2)),
                layers.Dropout(0.25),
                # Dense head
                layers.Flatten(),
                layers.Dense(512, activation="relu"),
                layers.BatchNormalization(),
                layers.Dropout(0.5),
                layers.Dense(256, activation="relu"),
                layers.BatchNormalization(),
                layers.Dropout(0.5),
                layers.Dense(num_classes, activation=activation),
            ]
        )

        metrics = ["accuracy"]
        if num_classes == 1:
            metrics.extend(
                [
                    tf.keras.metrics.Precision(name="precision"),
                    tf.keras.metrics.Recall(name="recall"),
                ]
            )

        model.compile(optimizer="adam", loss=loss, metrics=metrics)
        return model

    @staticmethod
    def transfer_learning_model(
        base_model_name="VGG16",
        input_shape=(224, 224, 3),
        num_classes=1,
        trainable_layers=0,
    ):
        """Build a transfer learning model with the requested backbone."""
        base_models = {
            "VGG16": VGG16,
            "ResNet50": ResNet50,
            "InceptionV3": InceptionV3,
            "DenseNet121": DenseNet121,
            "EfficientNetB0": EfficientNetB0,
        }

        if base_model_name not in base_models:
            raise ValueError(
                f"Model {base_model_name} not supported. Choose from {list(base_models)}"
            )

        base_model = base_models[base_model_name](
            weights="imagenet", include_top=False, input_shape=input_shape
        )

        base_model.trainable = False
        if trainable_layers > 0:
            for layer in base_model.layers[-trainable_layers:]:
                layer.trainable = True

        model = models.Sequential(
            [
                base_model,
                layers.GlobalAveragePooling2D(),
                layers.Dense(512, activation="relu"),
                layers.BatchNormalization(),
                layers.Dropout(0.5),
                layers.Dense(256, activation="relu"),
                layers.BatchNormalization(),
                layers.Dropout(0.5),
                layers.Dense(num_classes, activation="sigmoid" if num_classes == 1 else "softmax"),
            ]
        )

        metrics = ["accuracy"]
        if num_classes == 1:
            metrics.extend(
                [
                    tf.keras.metrics.Precision(name="precision"),
                    tf.keras.metrics.Recall(name="recall"),
                ]
            )

        loss = "binary_crossentropy" if num_classes == 1 else "categorical_crossentropy"
        model.compile(optimizer="adam", loss=loss, metrics=metrics)
        return model

    @staticmethod
    def get_callbacks(
        model_name: str,
        patience: int = 5,
        models_dir: Optional[str] = "models",
        results_dir: Optional[str] = "results",
        monitor: str = "val_accuracy",
    ):
        """Create shared training callbacks."""
        models_path = Path(models_dir or "models")
        results_path = Path(results_dir or "results")
        models_path.mkdir(parents=True, exist_ok=True)
        results_path.mkdir(parents=True, exist_ok=True)

        best_model_path = models_path / f"{model_name}_best.keras"
        csv_log_path = results_path / f"{model_name}_training_log.csv"

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=patience,
                restore_best_weights=True,
                verbose=1,
            ),
            tf.keras.callbacks.ModelCheckpoint(
                best_model_path.as_posix(),
                monitor=monitor,
                save_best_only=True,
                verbose=1,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=3,
                min_lr=1e-7,
                verbose=1,
            ),
            tf.keras.callbacks.CSVLogger(csv_log_path.as_posix()),
        ]

        return callbacks


def compare_architectures() -> Dict[str, Dict[str, int]]:
    """Return basic statistics for the supported transfer-learning backbones."""
    stats: Dict[str, Dict[str, int]] = {}
    for arch in ["VGG16", "ResNet50", "InceptionV3", "DenseNet121", "EfficientNetB0"]:
        model = CNNModels.transfer_learning_model(base_model_name=arch)
        stats[arch] = {
            "total_params": model.count_params(),
            "trainable_params": sum(tf.size(w).numpy() for w in model.trainable_weights),
            "layers": len(model.layers),
        }
    return stats


if __name__ == "__main__":
    print("Creating custom CNN model...")
    custom_model = CNNModels.custom_cnn()
    custom_model.summary()

    print("\n" + "=" * 80 + "\n")
    print("Creating VGG16 transfer learning model...")
    vgg_model = CNNModels.transfer_learning_model("VGG16")
    vgg_model.summary()
