# 🔬 Milk Quality Analysis System

A deep learning-based web application that performs real-time classification of milk samples using microscopic images to detect adulteration, contamination, and quality issues.

## 📊 Project Overview

This system uses Convolutional Neural Networks (CNN) to analyze microscopic images of milk samples and classify them into four categories:
- ✅ Pure Milk
- ⚠️ Glucose Adulteration
- ❌ Adulterated Samples
- 🦠 Pathogen Contamination

### Key Features

- **Real-time Monitoring**: Continuously watches a capture folder for new microscope images
- **Dual Input Methods**: 
  - Upload individual images for quick analysis
  - Monitor live microscope captures
- **Advanced Classification**:
  - CNN-based deep learning model
  - 86.81% validation accuracy
  - Real-time preprocessing and prediction
- **Interactive UI**:
  - Color-coded prediction badges
  - Confidence score visualization
  - Historical data tracking
  - Export functionality

## 🗂️ Project Structure

```
PROJECT-MAJOR/
├── app/
│   └── app.py                 # Main Streamlit application
├── data/
│   └── raw/
│       └── real/
│           ├── pure/         # Pure milk samples (2000 images)
│           ├── glucose/      # Glucose adulterated samples (2000 images)
│           ├── adulterated/  # Other adulterants samples (2000 images)
│           ├── pathogens/    # Contaminated samples (2000 images)
│           └── CAPTURE_FOLDER/ # Live microscope capture directory
│   └── processed/
│           ├── X_train.npy/  # Training images
│           ├── X_test.npy/   # Testing images
│           ├── y_train.npy/  # Training labels
│           ├── y_test.npy/   # Testing labels
├── models/
│   └── milk_classifier_cnn.keras  # Trained CNN model
├── scripts/
│   ├── generate_pathogen_images.py # Synthetic data generation(pathogens)
│   └── train_cnn.py              # Model training script
├── requirements.txt
└── README.md
```

## 🧠 Model Architecture

### CNN Model Details
- **Input Shape**: (128, 128, 3) RGB images
- **Architecture**:
  - 3 Convolutional blocks with batch normalization
  - Max pooling and dropout layers
  - Dense layers with regularization
  - Softmax output for 4 classes
- **Parameters**: 17.1M total parameters
- **Performance**:
  - Training Accuracy: 99.13%
  - Validation Accuracy: 86.81%

### Data Processing
- Image resizing to 128x128
- RGB color normalization
- Real-time data augmentation during training
- Batch processing support

## 🛠️ Technical Stack

### Core Technologies
- **Deep Learning**: TensorFlow/Keras
- **Web Interface**: Streamlit
- **Image Processing**: OpenCV
- **Data Handling**: NumPy
- **Real-time Updates**: Streamlit-autorefresh

### Development Tools
- **Environment**: Python Virtual Environment
- **Version Control**: Git
- **IDE**: Visual Studio Code
- **Package Management**: pip

## 📦 Installation

```bash
# Clone the repository
git clone <repository-url>
cd PROJECT-MAJOR

# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\activate  # Windows
source venv/bin/activate # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### Requirements
```txt
tensorflow>=2.0.0
opencv-python>=4.5.0
streamlit>=1.0.0
numpy>=1.19.0
pillow>=8.0.0
streamlit-autorefresh>=0.0.1
```

## 🚀 Usage

### Running the Application
```bash
streamlit run app/app.py
```

### Features Available
1. **Real-time Monitoring**
   - Auto-detection of new microscope captures
   - Instant classification
   - History tracking

2. **Manual Upload**
   - Drag-and-drop interface
   - Multiple format support
   - Instant results

3. **Analysis Dashboard**
   - Classification statistics
   - Model performance metrics
   - Export functionality

## 📈 Model Training

### Dataset Details
- **Total Images**: 8,000
- **Distribution**: 
  - Pure Milk: 2,000 images
  - Glucose Adulterated: 2,000 images
  - Other Adulterants: 2,000 images
  - Pathogen Contaminated: 2,000 images

### Training Process
```bash
python scripts/train_cnn.py
```
- Data augmentation during training
- Early stopping to prevent overfitting
- Learning rate scheduling
- Model checkpointing

## 🔧 Maintenance

### Adding New Samples
1. Place new images in appropriate class folders
2. Retrain model if needed using train_cnn.py
3. Model automatically updates in the application

### Monitoring
- Real-time capture folder monitoring
- Automatic error logging
- Performance statistics tracking

## 👥 Contributors
- Nithin R Poojary
- Vikas Pawar
- Rahul Vasnt Gunaga
- Mahendra Kummar P


---
*Last Updated: October 5, 2025*