#!/usr/bin/env python3
"""Quick training script to generate actual model files"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

print("TensorFlow version:", tf.__version__)
print("GPU available:", len(tf.config.list_physical_devices('GPU')) > 0)

tf.keras.utils.set_random_seed(42)
np.random.seed(42)

# Paths
DATA_DIR = 'data'
MODEL_DIR = 'models'
os.makedirs(MODEL_DIR, exist_ok=True)

# Image parameters
IMG_SIZE = (150, 150)
BATCH_SIZE = 32
EPOCHS = 10

print(f"\nLoading data from {DATA_DIR}...")

# Data generators with augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

test_datagen = ImageDataGenerator(rescale=1./255)

# Load data
train_generator = train_datagen.flow_from_directory(
    os.path.join(DATA_DIR, 'train'),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training'
)

val_generator = train_datagen.flow_from_directory(
    os.path.join(DATA_DIR, 'train'),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation'
)

test_generator = test_datagen.flow_from_directory(
    os.path.join(DATA_DIR, 'test'),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

print(f"Training samples: {train_generator.samples}")
print(f"Validation samples: {val_generator.samples}")
print(f"Test samples: {test_generator.samples}")

# Calculate class weights
class_counts = np.bincount(train_generator.classes)
total = sum(class_counts)
class_weight = {
    0: total / (2 * class_counts[0]),
    1: total / (2 * class_counts[1])
}
print(f"Class weights: {class_weight}")

# Model 1: Simple CNN
print("\n" + "="*50)
print("Training Model 1: Simple CNN")
print("="*50)

model1 = keras.Sequential([
    keras.layers.Input(shape=(150, 150, 3)),
    keras.layers.Conv2D(32, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D(2, 2),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D(2, 2),
    keras.layers.Conv2D(128, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D(2, 2),
    keras.layers.Flatten(),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

model1.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
)

history1 = model1.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator,
    class_weight=class_weight,
    verbose=1
)

# Evaluate and save
test_loss1, test_acc1, test_prec1, test_rec1 = model1.evaluate(test_generator, verbose=0)
print(f"\nSimple CNN - Test Accuracy: {test_acc1:.4f}, Precision: {test_prec1:.4f}, Recall: {test_rec1:.4f}")

model1.save(os.path.join(MODEL_DIR, 'simple_cnn.keras'))
print(f"Saved: {MODEL_DIR}/simple_cnn.keras")

# Model 2: VGG16 Transfer Learning
print("\n" + "="*50)
print("Training Model 2: VGG16 Transfer Learning")
print("="*50)

base_model = keras.applications.VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=(150, 150, 3)
)

# Freeze base layers
for layer in base_model.layers:
    layer.trainable = False

model2 = keras.Sequential([
    base_model,
    keras.layers.GlobalAveragePooling2D(),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(1, activation='sigmoid')
])

model2.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
)

history2 = model2.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator,
    class_weight=class_weight,
    verbose=1
)

# Evaluate and save
test_loss2, test_acc2, test_prec2, test_rec2 = model2.evaluate(test_generator, verbose=0)
print(f"\nVGG16 - Test Accuracy: {test_acc2:.4f}, Precision: {test_prec2:.4f}, Recall: {test_rec2:.4f}")

model2.save(os.path.join(MODEL_DIR, 'vgg16_transfer.keras'))
print(f"Saved: {MODEL_DIR}/vgg16_transfer.keras")

# Model 3: ResNet50 Transfer Learning
print("\n" + "="*50)
print("Training Model 3: ResNet50 Transfer Learning")
print("="*50)

base_model3 = keras.applications.ResNet50(
    weights='imagenet',
    include_top=False,
    input_shape=(150, 150, 3)
)

for layer in base_model3.layers:
    layer.trainable = False

model3 = keras.Sequential([
    base_model3,
    keras.layers.GlobalAveragePooling2D(),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(1, activation='sigmoid')
])

model3.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
)

history3 = model3.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator,
    class_weight=class_weight,
    verbose=1
)

# Evaluate and save
test_loss3, test_acc3, test_prec3, test_rec3 = model3.evaluate(test_generator, verbose=0)
print(f"\nResNet50 - Test Accuracy: {test_acc3:.4f}, Precision: {test_prec3:.4f}, Recall: {test_rec3:.4f}")

model3.save(os.path.join(MODEL_DIR, 'resnet50_transfer.keras'))
print(f"Saved: {MODEL_DIR}/resnet50_transfer.keras")

# Summary
print("\n" + "="*50)
print("TRAINING COMPLETE - MODELS SAVED")
print("="*50)
print(f"\n1. Simple CNN: {test_acc1:.2%} accuracy -> models/simple_cnn.keras")
print(f"2. VGG16:      {test_acc2:.2%} accuracy -> models/vgg16_transfer.keras")
print(f"3. ResNet50:   {test_acc3:.2%} accuracy -> models/resnet50_transfer.keras")
print(f"\nAll models saved in '{MODEL_DIR}/' folder")
