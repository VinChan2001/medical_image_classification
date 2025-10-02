# Medical Image Classification - Pneumonia Detection from Chest X-Rays

A deep learning project for automated pneumonia detection from chest X-ray images using Convolutional Neural Networks (CNNs) and transfer learning.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16+-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## Project Overview

This project implements state-of-the-art deep learning models to classify chest X-ray images as either **NORMAL** or **PNEUMONIA**. The system uses both custom CNN architectures and transfer learning with pre-trained models to achieve high accuracy in medical image classification.

### Key Features

- **Multiple Model Architectures**: Custom CNN and 5 pre-trained models (VGG16, ResNet50, InceptionV3, DenseNet121, EfficientNetB0)
- **Real Dataset Analysis**: Actual analysis of 5,840 chest X-ray images
- **Comprehensive Evaluation**: ROC curves, confusion matrices, precision-recall analysis
- **Data Augmentation**: Advanced augmentation techniques to improve generalization
- **Interactive Notebooks**: Jupyter notebooks with actual results and visualizations
- **Production-Ready Code**: Modular, documented, and tested

## Project Structure

```
medical-image-classification/
├── data/                          # Dataset (2.29 GB)
│   ├── train/                     # 5,216 training images
│   └── test/                      # 624 test images
├── notebooks/                     # Jupyter notebooks
│   ├── 01_data_exploration.ipynb       # EDA with actual results
│   └── 02_model_training_simple.ipynb  # Training pipeline
├── src/                           # Source code
│   ├── data_loader.py            # Data loading & preprocessing
│   ├── models.py                 # 6 CNN architectures
│   ├── train.py                  # Training pipeline
│   ├── evaluate.py               # Evaluation & visualization
│   └── predict.py                # Inference script
├── results/                       # Generated visualizations
│   ├── class_distribution.png    # Class distribution charts
│   ├── sample_images_train.png   # Sample X-ray images
│   ├── image_properties.png      # Image analysis
│   └── dataset_summary.csv       # Dataset statistics
├── models/                        # Saved models directory
├── download_dataset.py           # Dataset download script
├── requirements.txt              # Dependencies
├── README.md                     # Documentation
└── RESULTS_SUMMARY.md           # Project results
```

## Results

### Dataset Statistics (Actual)

| Split | Class | Count | Percentage |
|-------|-------|-------|------------|
| Train | PNEUMONIA | 3,875 | 74.3% |
| Train | NORMAL | 1,341 | 25.7% |
| Test | PNEUMONIA | 390 | 62.5% |
| Test | NORMAL | 234 | 37.5% |

**Total**: 5,840 images | **Size**: 2.29 GB

### Visualizations

See `results/` folder for:
- Class distribution charts
- Sample chest X-ray images
- Image property distributions
- Training history plots (when models are trained)

### Expected Model Performance

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Simple CNN | 80-85% | ~0.80 | ~0.85 | ~0.82 |
| Custom CNN | 85-90% | ~0.85 | ~0.90 | ~0.87 |
| VGG16 | 90-93% | ~0.90 | ~0.92 | ~0.91 |
| ResNet50 | 91-94% | ~0.91 | ~0.93 | ~0.92 |
| DenseNet121 | 92-95% | ~0.92 | ~0.94 | ~0.93 |
| EfficientNetB0 | 93-96% | ~0.93 | ~0.95 | ~0.94 |

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

## Usage

### 1. Exploratory Data Analysis

Run the Jupyter notebook to see actual dataset analysis:

```bash
jupyter notebook notebooks/01_data_exploration.ipynb
```

This notebook contains:
- Actual dataset statistics
- Class distribution visualizations
- Sample X-ray images
- Image property analysis

### 2. Model Training

**Option A: Use Jupyter Notebook** (Recommended)
```bash
jupyter notebook notebooks/02_model_training_simple.ipynb
```

**Option B: Command Line**

Train custom CNN:
```bash
python src/train.py --model_type custom --epochs 30 --batch_size 32
```

Train with transfer learning:
```bash
# VGG16
python src/train.py --model_type transfer --base_model VGG16 --epochs 20

# ResNet50
python src/train.py --model_type transfer --base_model ResNet50 --epochs 20

# DenseNet121 (Best performance)
python src/train.py --model_type transfer --base_model DenseNet121 --epochs 20
```

### 3. Make Predictions

```bash
# Basic prediction
python src/predict.py --model models/model.h5 --image path/to/xray.jpg

# With visualization
python src/predict.py --model models/model.h5 --image path/to/xray.jpg --visualize

# Generate detailed report
python src/predict.py --model models/model.h5 --image path/to/xray.jpg --report
```

### 4. Python API

```python
from src.predict import MedicalImagePredictor

# Initialize predictor
predictor = MedicalImagePredictor('models/model.h5')

# Make prediction
predicted_class, confidence = predictor.predict('path/to/xray.jpg')
print(f"Prediction: {predicted_class} ({confidence:.2%})")

# Generate detailed report
report = predictor.generate_report('path/to/xray.jpg')
print(report)
```

## Technical Details

### Data Preprocessing
- Image resizing to 224x224 pixels
- Normalization to [0, 1] range
- RGB color format

### Data Augmentation
- Rotation (±20°)
- Width/height shift (20%)
- Shear transformation (20%)
- Zoom (20%)
- Horizontal flip
- Fill mode: nearest

### Training Strategy
- Optimizer: Adam
- Loss function: Binary cross-entropy
- Metrics: Accuracy, Precision, Recall
- Callbacks:
  - Early stopping (patience=7)
  - Model checkpoint (save best)
  - Learning rate reduction on plateau
  - CSV logging

### Class Imbalance Handling
- Computed class weights (PNEUMONIA:NORMAL = 2.89:1)
- Applied during training to balance loss
- Validation split stratification

## Dependencies

Main libraries:
- TensorFlow >= 2.16.0
- NumPy >= 1.24.0
- Pandas >= 2.0.0
- Matplotlib >= 3.7.0
- Seaborn >= 0.12.0
- scikit-learn >= 1.3.0
- OpenCV >= 4.8.0

See [requirements.txt](requirements.txt) for complete list.

## Troubleshooting

### TensorFlow Installation Issues

If you encounter TensorFlow errors:

1. Create a clean virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

2. Or use Google Colab (no installation needed):
- Upload notebooks to Colab
- Mount Google Drive with dataset
- Run with free GPU

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
