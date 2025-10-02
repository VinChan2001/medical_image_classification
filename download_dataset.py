"""
Download chest X-ray pneumonia dataset using kagglehub.
"""

import kagglehub
import os
import shutil

print("=" * 80)
print("DOWNLOADING CHEST X-RAY PNEUMONIA DATASET")
print("=" * 80)

# Download latest version
print("\nDownloading dataset from Kaggle...")
path = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")

print(f"\nPath to dataset files: {path}")

# Copy to our data directory
data_dir = "data"
print(f"\nCopying dataset to {data_dir}/...")

# The downloaded dataset structure should have chest_xray folder
source_data = os.path.join(path, "chest_xray")
if os.path.exists(source_data):
    # Copy train and test directories
    for split in ['train', 'test', 'val']:
        src = os.path.join(source_data, split)
        if os.path.exists(src):
            dst = os.path.join(data_dir, split)
            if os.path.exists(dst):
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
            print(f"  Copied {split}/ directory")
else:
    print(f"Warning: Expected structure not found at {source_data}")
    print(f"Contents of {path}:")
    for item in os.listdir(path):
        print(f"  - {item}")

print("\n" + "=" * 80)
print("Dataset download and setup complete!")
print("=" * 80)

# Verify dataset
print("\nVerifying dataset structure:")
for split in ['train', 'test']:
    split_path = os.path.join(data_dir, split)
    if os.path.exists(split_path):
        for class_name in os.listdir(split_path):
            class_path = os.path.join(split_path, class_name)
            if os.path.isdir(class_path):
                count = len(os.listdir(class_path))
                print(f"  {split}/{class_name}: {count} images")
