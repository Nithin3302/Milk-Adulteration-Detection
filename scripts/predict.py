import cv2
import numpy as np
import joblib  # ✅ Use joblib instead of pickle
import os

# Load the trained model using joblib
model = joblib.load('models/milk_classifier.pkl')

# Class labels — must match training order!
labels = {
    0: 'Adulterated',
    1: 'Glucose',
    2: 'Pathogens',
    3: 'Pure'
}

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (128, 128))  # ✅ Match training size
    img = img / 255.0  # Normalize
    return img.flatten().reshape(1, -1)

def predict(image_path):
    try:
        features = preprocess_image(image_path)
        prediction = model.predict(features)[0]
        print(f"🧠 Prediction: {labels[prediction]}")
    except Exception as e:
        print(f"❌ Error: {e}")

# Example usage:
if __name__ == "__main__":
    image_path = input("📂 Enter path to test image: ")
    if os.path.exists(image_path):
        predict(image_path)
    else:
        print("🚫 File not found.")
