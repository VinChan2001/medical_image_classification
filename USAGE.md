# Usage Guide - Medical Image Classification

## Table of Contents
1. [Setup](#setup)
2. [Data Preparation](#data-preparation)
3. [Training Models](#training-models)
4. [Making Predictions](#making-predictions)
5. [Evaluation](#evaluation)
6. [Python API](#python-api)
7. [Troubleshooting](#troubleshooting)

---

## Setup

### 1. Clone and Navigate
```bash
cd medical-image-classification
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Verify Installation
```bash
python test_project.py
```

---

## Data Preparation

### Download Dataset

**Option 1: Kaggle API (Recommended)**
```bash
# Install Kaggle CLI
pip install kaggle

# Download dataset
kaggle datasets download -d paultimothymooney/chest-xray-pneumonia

# Extract
unzip chest-xray-pneumonia.zip -d data/
```

**Option 2: Manual Download**
1. Visit: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
2. Download ZIP file
3. Extract to `./data/` directory

### Verify Data Structure
```bash
data/
 train/
    NORMAL/
    PNEUMONIA/
 test/
     NORMAL/
     PNEUMONIA/
```

---

## Training Models

### Custom CNN

**Basic Training:**
```bash
python src/train.py --model_type custom --epochs 50
```

**With Custom Parameters:**
```bash
python src/train.py \
    --model_type custom \
    --epochs 50 \
    --batch_size 32 \
    --img_size 224
```

### Transfer Learning Models

**VGG16:**
```bash
python src/train.py \
    --model_type transfer \
    --base_model VGG16 \
    --epochs 30 \
    --batch_size 32
```

**ResNet50:**
```bash
python src/train.py \
    --model_type transfer \
    --base_model ResNet50 \
    --epochs 30
```

**DenseNet121:**
```bash
python src/train.py \
    --model_type transfer \
    --base_model DenseNet121 \
    --epochs 25
```

**EfficientNetB0:**
```bash
python src/train.py \
    --model_type transfer \
    --base_model EfficientNetB0 \
    --epochs 25
```

**InceptionV3:**
```bash
python src/train.py \
    --model_type transfer \
    --base_model InceptionV3 \
    --epochs 30
```

### Training Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--data_dir` | Path to dataset directory | `data` |
| `--model_type` | Model type: `custom` or `transfer` | `custom` |
| `--base_model` | Base model for transfer learning | `VGG16` |
| `--epochs` | Number of training epochs | `50` |
| `--batch_size` | Batch size | `32` |
| `--img_size` | Image size (square) | `224` |

### Output Files

After training, you'll get:
- `models/{model_name}_best.h5` - Best model checkpoint
- `models/{model_name}_final.h5` - Final trained model
- `results/{model_name}_results.json` - Training metrics
- `results/{model_name}_training_log.csv` - Epoch-by-epoch logs

---

## Making Predictions

### Command Line Interface

**Basic Prediction:**
```bash
python src/predict.py \
    --model models/vgg16_transfer_best.h5 \
    --image path/to/xray.jpg
```

**With Visualization:**
```bash
python src/predict.py \
    --model models/vgg16_transfer_best.h5 \
    --image path/to/xray.jpg \
    --visualize
```

**Save Visualization:**
```bash
python src/predict.py \
    --model models/vgg16_transfer_best.h5 \
    --image path/to/xray.jpg \
    --visualize \
    --output results/prediction.png
```

**Generate Detailed Report:**
```bash
python src/predict.py \
    --model models/vgg16_transfer_best.h5 \
    --image path/to/xray.jpg \
    --report
```

### Expected Output

**Basic Prediction:**
```
Prediction: PNEUMONIA
Confidence: 94.32%
```

**Detailed Report:**
```
PREDICTION REPORT

Predicted Class: PNEUMONIA
Confidence: 94.32%

Probabilities:
  Normal: 5.68%
  Pneumonia: 94.32%

Recommendation:
  High confidence pneumonia detection.
  Immediate medical consultation recommended.
```

---

## Evaluation

### Plot Training History

```python
from src.evaluate import plot_training_history

plot_training_history(
    'results/vgg16_transfer_results.json',
    save_path='results/training_history.png'
)
```

### Evaluate Model

```python
from src.evaluate import evaluate_model
from src.data_loader import MedicalImageLoader

# Load data
loader = MedicalImageLoader('data')
_, _, test_gen = loader.create_data_generators()

# Evaluate
metrics = evaluate_model(
    'models/vgg16_transfer_best.h5',
    test_gen,
    save_dir='results'
)
```

**Output Files:**
- `results/confusion_matrix.png`
- `results/roc_curve.png`
- `results/precision_recall_curve.png`
- `results/classification_report.txt`
- `results/evaluation_metrics.json`

### Visualize Predictions

```python
from src.evaluate import visualize_predictions

visualize_predictions(
    'models/vgg16_transfer_best.h5',
    test_gen,
    num_samples=16,
    save_path='results/predictions.png'
)
```

---

## Python API

### Training Pipeline

```python
from src.train import train_model

# Train custom CNN
model, history = train_model(
    data_dir='data',
    model_type='custom',
    epochs=50,
    batch_size=32,
    img_size=(224, 224)
)

# Train transfer learning model
model, history = train_model(
    data_dir='data',
    model_type='transfer',
    base_model='VGG16',
    epochs=30,
    batch_size=32
)
```

### Prediction

```python
from src.predict import MedicalImagePredictor

# Initialize predictor
predictor = MedicalImagePredictor('models/vgg16_transfer_best.h5')

# Single prediction
predicted_class, confidence = predictor.predict('path/to/xray.jpg')
print(f"Prediction: {predicted_class} ({confidence:.2%})")

# Batch prediction
image_paths = ['xray1.jpg', 'xray2.jpg', 'xray3.jpg']
results = predictor.predict_batch(image_paths)

for img, (pred, conf) in zip(image_paths, results):
    print(f"{img}: {pred} ({conf:.2%})")

# Generate detailed report
report = predictor.generate_report('path/to/xray.jpg')
print(f"Prediction: {report['predicted_class']}")
print(f"Confidence: {report['confidence']:.2%}")
print(f"Recommendation: {report['recommendation']}")

# Visualize prediction
predictor.visualize_prediction(
    'path/to/xray.jpg',
    save_path='results/prediction_viz.png'
)
```

### Data Loading

```python
from src.data_loader import MedicalImageLoader

# Initialize loader
loader = MedicalImageLoader(
    data_dir='data',
    img_size=(224, 224),
    batch_size=32
)

# Create data generators
train_gen, val_gen, test_gen = loader.create_data_generators(
    validation_split=0.2
)

# Get class weights (for imbalanced data)
class_weights = loader.get_class_weights(train_gen)
print(f"Class weights: {class_weights}")
```

### Model Building

```python
from src.models import CNNModels

# Build custom CNN
custom_model = CNNModels.custom_cnn(
    input_shape=(224, 224, 3),
    num_classes=1
)

# Build transfer learning model
vgg_model = CNNModels.transfer_learning_model(
    base_model_name='VGG16',
    input_shape=(224, 224, 3),
    num_classes=1,
    trainable_layers=0  # Freeze all base layers
)

# Get training callbacks
callbacks = CNNModels.get_callbacks(
    model_name='my_model',
    patience=5
)
```

---

## Troubleshooting

### Issue: Dataset not found

**Error:**
```
FileNotFoundError: [Errno 2] No such file or directory: 'data/train'
```

**Solution:**
Ensure dataset is downloaded and extracted to `data/` directory:
```bash
ls data/
# Should show: train/ test/
```

### Issue: GPU not detected

**Solution:**
```python
import tensorflow as tf
print("GPU Available:", tf.config.list_physical_devices('GPU'))
```

If no GPU, training will use CPU (slower but works fine).

### Issue: Out of memory

**Solution:**
Reduce batch size:
```bash
python src/train.py --model_type custom --batch_size 16
```

### Issue: Model not converging

**Solutions:**
1. Increase epochs: `--epochs 100`
2. Try different model: `--base_model ResNet50`
3. Adjust learning rate in `models.py`
4. Check data quality

### Issue: Import errors

**Solution:**
```bash
pip install --upgrade -r requirements.txt
```

### Issue: TensorFlow compatibility

**Solution:**
```bash
pip install tensorflow>=2.16.0 --upgrade
```

---

## Performance Tips

### For Faster Training:
- Use GPU if available
- Increase batch size (if memory allows)
- Use transfer learning (faster convergence)
- Start with fewer epochs for testing

### For Better Accuracy:
- Use transfer learning (DenseNet121 or EfficientNetB0)
- Train for more epochs
- Fine-tune top layers
- Use data augmentation
- Handle class imbalance

### For Production:
- Save best model with checkpoints
- Log all training metrics
- Validate on separate test set
- Document model version and parameters
- Include confidence thresholds

---

## Quick Reference

### Most Common Commands

```bash
# Download data
kaggle datasets download -d paultimothymooney/chest-xray-pneumonia

# Train best model
python src/train.py --model_type transfer --base_model DenseNet121 --epochs 30

# Make prediction
python src/predict.py --model models/model.h5 --image xray.jpg --report

# Run validation
python test_project.py

# View demo
python demo.py
```

---

For more information, see [README.md](README.md) or [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)
