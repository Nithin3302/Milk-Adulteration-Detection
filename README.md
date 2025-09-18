Milk Adulteration Detector:
A Streamlit-based web app that classifies microscope images of milk samples in real time to detect adulteration and contamination using a trained machine learning model.
---

🔍 Features:

1. Upload microscope images or monitor a live capture folder for instant classification
2. Color-coded prediction badges for easy interpretation
3. Predicts:
-Pure milk
-Adulterated milk
-Glucose adulteration
-Pathogen presence
4. Shows prediction confidence (optional)
5. Tracks and displays classification history
---

🧠 Model:

1. Trained scikit-learn model (milk_classifier.pkl)
2. Input size: 128x128 (image resized automatically)
3. Uses OpenCV for preprocessing and feature extraction
---

📊 Datasets:

1. Milk sample images: Custom dataset (microscope images of pure, adulterated, glucose, and pathogen samples)
2. Suitable for both simulated and real-world lab data
---

🖥️ Run Locally

git clone https://github.com/Nithin3302/Milk-Adulteration-Detection-New-.git
cd Milk-Adulteration-Detection-New-
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate
pip install -r requirements.txt
streamlit run app/app.py
---

📂 Project Structure
app/                # Streamlit app code
  app.py
data/
  raw/real/CAPTURE_FOLDER/   # Folder for live microscope captures
models/
  milk_classifier.pkl        # Trained model
scripts/
  train_model.py             # Model training script
  predict.py                 # Single image prediction script
requirements.txt
README.md
---

📦 Requirements Make sure you have the following Python packages installed:

🧪 streamlit – For the web interface
🧠 scikit-learn, joblib – For loading and running the trained model
🔢 numpy – For numerical computations
📷 opencv-python – For image preprocessing
🕒 streamlit-autorefresh, watchdog – For real-time updates

To install all at once:
pip install -r requirements.txt

👥 Contributors:
Nithin R Poojary