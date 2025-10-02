"""
Create synthetic chest X-ray dataset for testing the pipeline.
This creates a small dataset to demonstrate the full ML workflow.
"""

import os
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import random

def create_synthetic_xray(img_size=(224, 224), has_pneumonia=False):
    """Create a synthetic chest X-ray image."""
    # Create base image (grayscale chest X-ray appearance)
    img = Image.new('RGB', img_size, color=(20, 20, 20))
    draw = ImageDraw.Draw(img)

    # Add ribcage pattern
    for i in range(8):
        y = 30 + i * 20
        x_offset = random.randint(-10, 10)
        draw.arc([30 + x_offset, y, img_size[0] - 30 + x_offset, y + 40],
                 start=0, end=180, fill=(60, 60, 60), width=2)

    # Add lung fields
    left_lung = [50, 60, img_size[0]//2 - 10, img_size[1] - 60]
    right_lung = [img_size[0]//2 + 10, 60, img_size[0] - 50, img_size[1] - 60]

    draw.ellipse(left_lung, fill=(40, 40, 40), outline=(70, 70, 70))
    draw.ellipse(right_lung, fill=(40, 40, 40), outline=(70, 70, 70))

    if has_pneumonia:
        # Add cloudy infiltrates (pneumonia pattern)
        for _ in range(random.randint(5, 10)):
            x = random.randint(60, img_size[0] - 60)
            y = random.randint(70, img_size[1] - 70)
            size = random.randint(15, 40)
            opacity = random.randint(80, 120)
            draw.ellipse([x, y, x + size, y + size],
                        fill=(opacity, opacity, opacity))

        # Add some texture
        img = img.filter(ImageFilter.GaussianBlur(radius=2))
    else:
        # Normal: clearer lung fields
        img = img.filter(ImageFilter.GaussianBlur(radius=1))

    # Add some noise
    pixels = np.array(img)
    noise = np.random.normal(0, 5, pixels.shape)
    pixels = np.clip(pixels + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(pixels)

    return img

def create_dataset(data_dir, num_train=200, num_test=50):
    """Create synthetic dataset with train/test split."""
    print("Creating synthetic chest X-ray dataset...")
    print("=" * 60)

    # Create directory structure
    splits = {
        'train': num_train,
        'test': num_test
    }

    classes = ['NORMAL', 'PNEUMONIA']

    for split, total_images in splits.items():
        for class_name in classes:
            dir_path = os.path.join(data_dir, split, class_name)
            os.makedirs(dir_path, exist_ok=True)

            # Create images (60% pneumonia, 40% normal to simulate imbalance)
            num_images = total_images // 2
            if class_name == 'PNEUMONIA':
                num_images = int(num_images * 1.5)  # More pneumonia cases

            print(f"Creating {num_images} {class_name} images for {split}...")

            for i in range(num_images):
                has_pneumonia = (class_name == 'PNEUMONIA')
                img = create_synthetic_xray(has_pneumonia=has_pneumonia)

                filename = f"{class_name}_{split}_{i+1:04d}.jpeg"
                filepath = os.path.join(dir_path, filename)
                img.save(filepath, 'JPEG', quality=85)

            print(f"  Created {num_images} images in {dir_path}")

    print("\n" + "=" * 60)
    print("Dataset creation complete!")
    print("\nDataset structure:")
    for split in splits.keys():
        for class_name in classes:
            dir_path = os.path.join(data_dir, split, class_name)
            count = len(os.listdir(dir_path))
            print(f"  {split}/{class_name}: {count} images")

if __name__ == "__main__":
    data_dir = "data"
    create_dataset(data_dir, num_train=200, num_test=50)
    print("\nReady to train models!")
