# 🍼 Milk Adulteration Detector

A Streamlit-based web app that classifies microscope images of milk samples in real time to detect adulteration and contamination using a trained machine learning model.

## 🔍 Features

- Upload microscope images or monitor a live capture folder for instant classification
- Color-coded prediction badges for easy interpretation
- Real-time classification of:
  - Pure milk
  - Adulterated milk
  - Glucose adulteration
  - Pathogen presence
- Shows prediction confidence (optional)
- Tracks and displays classification history

## 🧠 Model

- Trained scikit-learn model (`milk_classifier.pkl`)
- Input size: 128x128 (image resized automatically)
- Uses OpenCV for preprocessing and feature extraction

## 📊 Datasets

- Milk sample images: Custom dataset (microscope images of pure, adulterated, glucose, and pathogen samples)
- Suitable for both simulated and real-world lab data

## 🖥️ Installation & Usage

```bash
# Clone the repository
git clone https://github.com/Nithin3302/Milk-Adulteration-Detection-New-.git
cd Milk-Adulteration-Detection-New-

# Create and activate virtual environment
python -m venv venv

# On Windows:
.\venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app/app.py
```

## 📂 Project Structure

```
app/                # Streamlit app code
├── app.py         # Main application
data/
├── raw/
    └── real/
        └── CAPTURE_FOLDER/  # Live microscope captures
models/
└── milk_classifier.pkl      # Trained model
scripts/
├── train_model.py          # Model training script
└── predict.py             # Single image prediction script
requirements.txt
README.md
```

## 📦 Requirements

- 🧪 `streamlit` – For the web interface
- 🧠 `scikit-learn`, `joblib` – For loading and running the trained model
- 🔢 `numpy` – For numerical computations
- 📷 `opencv-python` – For image preprocessing
- 🕒 `streamlit-autorefresh`, `watchdog` – For real-time updates

```bash
pip install -r requirements.txt
```

## 👥 Contributors

-Nithin R Poojary