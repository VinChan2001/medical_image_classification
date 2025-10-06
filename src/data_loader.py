"""Data loading and preprocessing helpers for medical image classification."""

import os
from pathlib import Path
from typing import Optional

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm

try:  # Optional dependency for direct image loading utilities
    import cv2
except ImportError:  # pragma: no cover - handled gracefully for non-OpenCV setups
    cv2 = None


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
        self.using_separate_val = False
        self.val_image_count = 0

    def create_data_generators(
        self,
        validation_split=0.2,
        seed: Optional[int] = 42,
        min_val_images: int = 64,
    ):
        """
        Create data generators with augmentation.

        Args:
            validation_split: Fraction of training data to use for validation
            seed: Random seed for generator shuffling

        Returns:
            train_generator, val_generator, test_generator
        """
        train_dir = Path(self.data_dir) / "train"
        val_dir = Path(self.data_dir) / "val"
        test_dir = Path(self.data_dir) / "test"

        if not train_dir.exists():
            raise FileNotFoundError(f"Expected training directory at {train_dir}")

        # Training augmentation
        augmentation_kwargs = dict(
            rescale=1.0 / 255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode="nearest",
        )

        val_image_count = self._count_directory_images(val_dir)
        self.val_image_count = val_image_count
        self.using_separate_val = val_image_count >= max(min_val_images, self.batch_size)

        if self.using_separate_val:
            train_datagen = ImageDataGenerator(**augmentation_kwargs)
        else:
            train_datagen = ImageDataGenerator(**augmentation_kwargs, validation_split=validation_split)

        test_datagen = ImageDataGenerator(rescale=1.0 / 255)

        # Training generator
        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode="binary",
            subset=None if self.using_separate_val else "training",
            shuffle=True,
            seed=seed,
        )

        # Validation generator (auto-detect separate validation directory)
        if self.using_separate_val:
            val_generator = test_datagen.flow_from_directory(
                val_dir,
                target_size=self.img_size,
                batch_size=self.batch_size,
                class_mode="binary",
                shuffle=False,
            )
        else:
            val_generator = train_datagen.flow_from_directory(
                train_dir,
                target_size=self.img_size,
                batch_size=self.batch_size,
                class_mode="binary",
                subset="validation",
                shuffle=False,
                seed=seed,
            )

        # Test generator
        test_generator = test_datagen.flow_from_directory(
            test_dir,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode="binary",
            shuffle=False,
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
        if cv2 is None:
            raise ImportError(
                "OpenCV is required for load_images_from_directory(). "
                "Install opencv-python to enable direct image loading."
            )
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

    @staticmethod
    def _count_directory_images(directory: Path) -> int:
        """Count image files within a directory tree."""
        if not directory.is_dir():
            return 0

        image_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".gif"}
        count = 0

        for class_dir in directory.iterdir():
            if not class_dir.is_dir():
                continue
            for file in class_dir.iterdir():
                if file.is_file() and file.suffix.lower() in image_extensions:
                    count += 1

        return count


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
