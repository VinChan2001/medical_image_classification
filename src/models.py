"""
CNN model architectures for medical image classification.
Includes both custom CNN and transfer learning approaches.
"""

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import (
    VGG16, ResNet50, InceptionV3, DenseNet121, EfficientNetB0
)


class CNNModels:
    """Collection of CNN models for medical image classification."""

    @staticmethod
    def custom_cnn(input_shape=(224, 224, 3), num_classes=1):
        """
        Build a custom CNN architecture.

        Args:
            input_shape: Input image shape (height, width, channels)
            num_classes: Number of output classes (1 for binary)

        Returns:
            Compiled Keras model
        """
        model = models.Sequential([
            # Block 1
            layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),

            # Block 2
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),

            # Block 3
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),

            # Block 4
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),

            # Fully connected layers
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation='sigmoid')
        ])

        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        )

        return model

    @staticmethod
    def transfer_learning_model(base_model_name='VGG16', input_shape=(224, 224, 3),
                                num_classes=1, trainable_layers=0):
        """
        Build a transfer learning model.

        Args:
            base_model_name: Name of pretrained model ('VGG16', 'ResNet50', 'InceptionV3',
                           'DenseNet121', 'EfficientNetB0')
            input_shape: Input image shape
            num_classes: Number of output classes
            trainable_layers: Number of top layers to make trainable (0 = freeze all)

        Returns:
            Compiled Keras model
        """
        # Select base model
        base_models = {
            'VGG16': VGG16,
            'ResNet50': ResNet50,
            'InceptionV3': InceptionV3,
            'DenseNet121': DenseNet121,
            'EfficientNetB0': EfficientNetB0
        }

        if base_model_name not in base_models:
            raise ValueError(f"Model {base_model_name} not supported. Choose from {list(base_models.keys())}")

        # Load base model without top layers
        base_model = base_models[base_model_name](
            weights='imagenet',
            include_top=False,
            input_shape=input_shape
        )

        # Freeze base model layers
        base_model.trainable = False

        # Optionally unfreeze top layers
        if trainable_layers > 0:
            for layer in base_model.layers[-trainable_layers:]:
                layer.trainable = True

        # Build model
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation='sigmoid')
        ])

        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        )

        return model

    @staticmethod
    def get_callbacks(model_name, patience=5):
        """
        Create training callbacks.

        Args:
            model_name: Name for saving model checkpoints
            patience: Early stopping patience

        Returns:
            List of Keras callbacks
        """
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=patience,
                restore_best_weights=True,
                verbose=1
            ),
            tf.keras.callbacks.ModelCheckpoint(
                f'models/{model_name}_best.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-7,
                verbose=1
            ),
            tf.keras.callbacks.CSVLogger(
                f'results/{model_name}_training_log.csv'
            )
        ]

        return callbacks


def compare_architectures():
    """
    Compare different model architectures.

    Returns:
        Dictionary with model statistics
    """
    models_info = {}

    architectures = ['VGG16', 'ResNet50', 'InceptionV3', 'DenseNet121', 'EfficientNetB0']

    for arch in architectures:
        model = CNNModels.transfer_learning_model(base_model_name=arch)
        models_info[arch] = {
            'total_params': model.count_params(),
            'trainable_params': sum([tf.size(w).numpy() for w in model.trainable_weights]),
            'layers': len(model.layers)
        }

    return models_info


if __name__ == "__main__":
    # Test model creation
    print("Creating custom CNN model...")
    custom_model = CNNModels.custom_cnn()
    custom_model.summary()

    print("\n" + "="*80 + "\n")
    print("Creating VGG16 transfer learning model...")
    vgg_model = CNNModels.transfer_learning_model('VGG16')
    vgg_model.summary()
