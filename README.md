# Medical Image Classification - Pneumonia Detection from Chest X-Rays

A deep learning project for automated pneumonia detection from chest X-ray images using Convolutional Neural Networks (CNNs) and transfer learning.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## Project Overview

This project implements state-of-the-art deep learning models to classify chest X-ray images as either **NORMAL** or **PNEUMONIA**. The system uses both custom CNN architectures and transfer learning with pre-trained models (VGG16, ResNet50, InceptionV3, DenseNet121, EfficientNetB0) to achieve high accuracy in medical image classification.

### Key Features

- **Multiple Model Architectures**: Custom CNN and 5 pre-trained models
- **Comprehensive Evaluation**: ROC curves, confusion matrices, precision-recall analysis
- **Data Augmentation**: Advanced augmentation techniques to improve generalization
- **Performance Tracking**: Detailed training logs and metrics visualization
- **Inference Pipeline**: Easy-to-use prediction interface for new images
- **Jupyter Notebooks**: Interactive EDA and analysis

## Project Structure

```
medical-image-classification/
 data/                          # Dataset directory (not included)
    train/
       NORMAL/
       PNEUMONIA/
    test/
        NORMAL/
        PNEUMONIA/
 src/                           # Source code
    data_loader.py            # Data loading and preprocessing
    models.py                 # Model architectures
    train.py                  # Training pipeline
    evaluate.py               # Evaluation and visualization
    predict.py                # Inference script
 notebooks/                     # Jupyter notebooks
    exploratory_analysis.ipynb
 models/                        # Saved models (generated during training)
 results/                       # Training results and visualizations
 requirements.txt              # Python dependencies
 README.md                     # This file
```

## Getting Started

### Prerequisites

- Python 3.8 or higher
- pip package manager
- (Optional) GPU with CUDA support for faster training

### Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/medical-image-classification.git
cd medical-image-classification
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the dataset:

**Option 1: Using kagglehub (Recommended)**
```bash
pip install kagglehub
python download_dataset.py
```

**Option 2: Using Kaggle API**
```bash
pip install kaggle
kaggle datasets download -d paultimothymooney/chest-xray-pneumonia
unzip chest-xray-pneumonia.zip -d data/
```

**Option 3: Manual Download**
- Visit [Chest X-Ray Images (Pneumonia) Dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- Download and extract to `./data/` directory

## Dataset Information

- **Source**: Kaggle - Chest X-Ray Images (Pneumonia)
- **Total Size**: 2.29 GB
- **Training Images**: 5,216 images
  - NORMAL: 1,341 images
  - PNEUMONIA: 3,875 images
- **Test Images**: 624 images
  - NORMAL: 234 images
  - PNEUMONIA: 390 images
- **Classes**: 2 (NORMAL, PNEUMONIA)
- **Image Format**: JPEG
- **Class Imbalance**: ~74% pneumonia cases in training set

## Model Training

### Train Custom CNN

```bash
python src/train.py --model_type custom --epochs 50 --batch_size 32
```

### Train with Transfer Learning

```bash
# VGG16
python src/train.py --model_type transfer --base_model VGG16 --epochs 30

# ResNet50
python src/train.py --model_type transfer --base_model ResNet50 --epochs 30

# DenseNet121
python src/train.py --model_type transfer --base_model DenseNet121 --epochs 30

# EfficientNetB0
python src/train.py --model_type transfer --base_model EfficientNetB0 --epochs 30
```

### Training Arguments

- `--data_dir`: Path to dataset (default: `data`)
- `--model_type`: Model type - `custom` or `transfer`
- `--base_model`: Base model for transfer learning (VGG16, ResNet50, InceptionV3, DenseNet121, EfficientNetB0)
- `--epochs`: Number of training epochs (default: 50)
- `--batch_size`: Batch size (default: 32)
- `--img_size`: Image size in pixels (default: 224)

## Model Evaluation

Evaluate a trained model:

```python
from src.evaluate import evaluate_model, plot_training_history
from src.data_loader import MedicalImageLoader

# Plot training history
plot_training_history('results/vgg16_transfer_results.json')

# Evaluate on test set
loader = MedicalImageLoader('data')
_, _, test_gen = loader.create_data_generators()
metrics = evaluate_model('models/vgg16_transfer_best.h5', test_gen)
```

### Evaluation Metrics

- **Accuracy**: Overall classification accuracy
- **Precision**: Positive predictive value
- **Recall**: Sensitivity/True positive rate
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under ROC curve
- **Confusion Matrix**: True vs predicted classifications

## Making Predictions

### Command Line Interface

```bash
# Basic prediction
python src/predict.py --model models/vgg16_transfer_best.h5 --image path/to/xray.jpg

# With visualization
python src/predict.py --model models/vgg16_transfer_best.h5 --image path/to/xray.jpg --visualize

# Generate detailed report
python src/predict.py --model models/vgg16_transfer_best.h5 --image path/to/xray.jpg --report
```

### Python API

```python
from src.predict import MedicalImagePredictor

# Initialize predictor
predictor = MedicalImagePredictor('models/vgg16_transfer_best.h5')

# Make prediction
predicted_class, confidence = predictor.predict('path/to/xray.jpg')
print(f"Prediction: {predicted_class} (Confidence: {confidence:.2%})")

# Generate detailed report
report = predictor.generate_report('path/to/xray.jpg')
print(report)

# Visualize prediction
predictor.visualize_prediction('path/to/xray.jpg', save_path='results/prediction.png')
```

## Exploratory Data Analysis

Run the Jupyter notebook for interactive data exploration:

```bash
jupyter notebook notebooks/exploratory_analysis.ipynb
```

The notebook includes:
- Dataset overview and class distribution
- Image properties analysis
- Sample visualizations
- Pixel intensity distributions
- Summary statistics

## Expected Performance

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Custom CNN | ~85-90% | ~0.85 | ~0.90 | ~0.87 |
| VGG16 | ~90-93% | ~0.90 | ~0.92 | ~0.91 |
| ResNet50 | ~91-94% | ~0.91 | ~0.93 | ~0.92 |
| DenseNet121 | ~92-95% | ~0.92 | ~0.94 | ~0.93 |
| EfficientNetB0 | ~93-96% | ~0.93 | ~0.95 | ~0.94 |

*Note: Actual performance may vary based on training parameters and hardware*

## Technical Details

### Data Preprocessing
- Image resizing to 224224 pixels
- Normalization to [0, 1] range
- RGB color format

### Data Augmentation
- Rotation (20)
- Width/height shift (20%)
- Shear transformation (20%)
- Zoom (20%)
- Horizontal flip
- Fill mode: nearest

### Training Strategy
- Optimizer: Adam
- Loss function: Binary cross-entropy
- Callbacks:
  - Early stopping (patience=7)
  - Model checkpoint (save best)
  - Learning rate reduction on plateau
  - CSV logging

### Class Imbalance Handling
- Computed class weights
- Applied during training to balance loss

## Dependencies

Main libraries:
- TensorFlow 2.15.0
- Keras 2.15.0
- NumPy 1.24.3
- Pandas 2.0.3
- Matplotlib 3.7.2
- Seaborn 0.12.2
- scikit-learn 1.3.0
- OpenCV 4.8.0

See [requirements.txt](requirements.txt) for complete list.

## Disclaimer

This project is for **educational and research purposes only**. It should not be used as a substitute for professional medical diagnosis. Always consult qualified healthcare professionals for medical decisions.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact

For questions or feedback, please open an issue on GitHub.

## Acknowledgments

- Dataset: [Paul Mooney - Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- Pre-trained models: TensorFlow/Keras Applications
- Inspiration: Medical imaging research community

## References

1. Kermany, D. S., et al. (2018). "Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning." Cell, 172(5), 1122-1131.
2. Rajpurkar, P., et al. (2017). "CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays with Deep Learning." arXiv:1711.05225.

---

Made for advancing medical AI research
