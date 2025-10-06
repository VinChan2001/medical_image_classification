# Training Advanced Models on Google Colab

This guide shows how to train the advanced models (VGG16, ResNet50, DenseNet121) on Google Colab with free GPU.

## Why Google Colab?

- Free GPU access (Tesla T4)
- No TensorFlow installation issues
- Fast training (10-15 min per model)
- No local environment setup needed

## Steps

### 1. Upload Notebook to Colab

1. Go to [Google Colab](https://colab.research.google.com/)
2. Click **File** → **Upload notebook**
3. Upload `notebooks/03_advanced_models.ipynb`

### 2. Enable GPU

1. Click **Runtime** → **Change runtime type**
2. Select **GPU** (Tesla T4 or better)
3. Click **Save**

### 3. Upload Dataset

Option A: **Mount Google Drive** (Recommended if dataset is on Drive)
```python
from google.colab import drive
drive.mount('/content/drive')

# Copy dataset path
data_dir = '/content/drive/MyDrive/chest-xray-data'
```

Option B: **Upload ZIP file**
```python
from google.colab import files
uploaded = files.upload()  # Upload chest-xray-pneumonia.zip

# Extract
!unzip chest-xray-pneumonia.zip -d data/
```

Option C: **Download directly in Colab**
```python
!pip install kagglehub
import kagglehub
path = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")
print(f"Dataset downloaded to: {path}")
```

### 4. Install Dependencies

```python
!pip install tensorflow numpy pandas matplotlib seaborn scikit-learn pillow opencv-python
```

### 5. Upload Source Code

```python
from google.colab import files

# Upload all files from src/ folder
# data_loader.py, models.py, etc.
uploaded = files.upload()
```

### 6. Run Training

Click **Runtime** → **Run all** or run cells one by one

Training time:
- VGG16: ~10-12 minutes (20 epochs)
- ResNet50: ~12-15 minutes (20 epochs)
- DenseNet121: ~15-18 minutes (20 epochs)

### 7. Download Results

```python
from google.colab import files

# Download trained models
files.download('vgg16_final.h5')
files.download('resnet50_final.h5')

# Download results
files.download('confusion_matrices.png')
files.download('training_history.png')
files.download('model_performance.csv')
```

## Expected Results

### VGG16
- Test Accuracy: 90-93%
- Test Precision: 0.89-0.91
- Test Recall: 0.91-0.94
- F1-Score: 0.90-0.92

### ResNet50
- Test Accuracy: 91-94%
- Test Precision: 0.90-0.92
- Test Recall: 0.92-0.95
- F1-Score: 0.91-0.93

### DenseNet121 (Best)
- Test Accuracy: 92-95%
- Test Precision: 0.91-0.93
- Test Recall: 0.93-0.96
- F1-Score: 0.92-0.94

## Alternative: Kaggle Notebooks

You can also use [Kaggle Notebooks](https://www.kaggle.com/code) which provides:
- Free GPU (30hrs/week)
- Dataset already available
- No upload needed

1. Go to Kaggle
2. Create new notebook
3. Add dataset: chest-xray-pneumonia
4. Enable GPU
5. Upload and run `03_advanced_models.ipynb`

## Troubleshooting

**Out of Memory:**
- Reduce batch_size to 16 or 8
- Use smaller image size (128x128 instead of 224x224)

**Dataset not found:**
- Check data_dir path
- Verify folder structure: data/train/ and data/test/

**Slow training:**
- Verify GPU is enabled (Runtime → Change runtime type)
- Check GPU usage: `!nvidia-smi`

## Complete Colab Notebook Template

```python
# Step 1: Setup
!pip install kagglehub

# Step 2: Download dataset
import kagglehub
path = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")

# Step 3: Upload source files (drag and drop in Colab files panel)
# - data_loader.py
# - models.py

# Step 4: Import and run
import sys
sys.path.append('/content')

from data_loader import MedicalImageLoader
from models import CNNModels

# ... rest of training code from notebook
```

---

After training on Colab, download the models and results, then place them in:
- `models/` folder (trained .h5 files)
- `results/` folder (confusion matrices, training history, performance CSV)

This will give you real, trained models with actual performance metrics to showcase in your repository!
