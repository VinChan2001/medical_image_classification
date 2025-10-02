"""
Data loader and preprocessing utilities for medical image classification.
This script uses the Chest X-Ray Images (Pneumonia) dataset from Kaggle.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import cv2
from tqdm import tqdm


class MedicalImageLoader:
    """Load and preprocess medical images for classification."""

    def __init__(self, data_dir, img_size=(224, 224), batch_size=32):
        """
        Initialize the data loader.

        Args:
            data_dir: Directory containing the dataset
            img_size: Target image size (height, width)
            batch_size: Batch size for training
        """
        self.data_dir = data_dir
        self.img_size = img_size
        self.batch_size = batch_size

    def create_data_generators(self, validation_split=0.2):
        """
        Create data generators with augmentation.

        Args:
            validation_split: Fraction of training data to use for validation

        Returns:
            train_generator, val_generator, test_generator
        """
        # Data augmentation for training
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest',
            validation_split=validation_split
        )

        # Only rescaling for test data
        test_datagen = ImageDataGenerator(rescale=1./255)

        # Training generator
        train_generator = train_datagen.flow_from_directory(
            os.path.join(self.data_dir, 'train'),
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='binary',
            subset='training',
            shuffle=True
        )

        # Validation generator
        val_generator = train_datagen.flow_from_directory(
            os.path.join(self.data_dir, 'train'),
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='binary',
            subset='validation',
            shuffle=False
        )

        # Test generator
        test_generator = test_datagen.flow_from_directory(
            os.path.join(self.data_dir, 'test'),
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='binary',
            shuffle=False
        )

        return train_generator, val_generator, test_generator

    def load_images_from_directory(self, directory, label):
        """
        Load images from a directory.

        Args:
            directory: Path to image directory
            label: Class label (0 or 1)

        Returns:
            images, labels as numpy arrays
        """
        images = []
        labels = []

        for filename in tqdm(os.listdir(directory), desc=f"Loading {os.path.basename(directory)}"):
            if filename.endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(directory, filename)
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.resize(img, self.img_size)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    images.append(img)
                    labels.append(label)

        return np.array(images), np.array(labels)

    def get_class_weights(self, train_generator):
        """
        Calculate class weights for imbalanced datasets.

        Args:
            train_generator: Training data generator

        Returns:
            Dictionary of class weights
        """
        from sklearn.utils import class_weight

        classes = train_generator.classes
        class_weights = class_weight.compute_class_weight(
            'balanced',
            classes=np.unique(classes),
            y=classes
        )

        return dict(enumerate(class_weights))


def download_dataset_instructions():
    """Print instructions for downloading the dataset."""
    print("=" * 80)
    print("DATASET DOWNLOAD INSTRUCTIONS")
    print("=" * 80)
    print("\nThis project uses the Chest X-Ray Images (Pneumonia) dataset.")
    print("\nOption 1: Kaggle API (Recommended)")
    print("-" * 40)
    print("1. Install kaggle: pip install kaggle")
    print("2. Set up Kaggle API credentials (kaggle.json)")
    print("3. Run the following command:")
    print("   kaggle datasets download -d paultimothymooney/chest-xray-pneumonia")
    print("4. Unzip to ./data/ directory")
    print("\nOption 2: Manual Download")
    print("-" * 40)
    print("1. Visit: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia")
    print("2. Download the dataset")
    print("3. Extract to ./data/ directory")
    print("\nExpected structure:")
    print("data/")
    print("├── train/")
    print("│   ├── NORMAL/")
    print("│   └── PNEUMONIA/")
    print("└── test/")
    print("    ├── NORMAL/")
    print("    └── PNEUMONIA/")
    print("=" * 80)


if __name__ == "__main__":
    download_dataset_instructions()
