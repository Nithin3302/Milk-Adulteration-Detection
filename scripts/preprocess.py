import os
import cv2
import numpy as np

# Constants
RAW_DIR = 'data/raw/simulated'
PROCESSED_DIR = 'data/processed'
IMAGE_SIZE = (128, 128)

# Labels for each class
CLASSES = {
    "pure": 0,
    "adulterated": 1,
    "glucose": 2,
    "pathogens": 3
}

def load_images_from_folder(folder_path, label):
    data = []
    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, IMAGE_SIZE)
            img = img / 255.0  # Normalize
            data.append((img, label))
    return data

def load_all_data():
    dataset = []
    for class_name, label in CLASSES.items():
        class_path = os.path.join(RAW_DIR, class_name)
        dataset += load_images_from_folder(class_path, label)
    return dataset

def preprocess_and_save():
    data = load_all_data()
    np.random.shuffle(data)

    images, labels = zip(*data)
    images = np.array(images)
    labels = np.array(labels)

    # Train/test split (80% train, 20% test)
    split_idx = int(len(images) * 0.8)
    X_train, X_test = images[:split_idx], images[split_idx:]
    y_train, y_test = labels[:split_idx], labels[split_idx:]

    # Save to processed directory
    np.save(os.path.join(PROCESSED_DIR, 'X_train.npy'), X_train)
    np.save(os.path.join(PROCESSED_DIR, 'y_train.npy'), y_train)
    np.save(os.path.join(PROCESSED_DIR, 'X_test.npy'), X_test)
    np.save(os.path.join(PROCESSED_DIR, 'y_test.npy'), y_test)

    print("✅ Preprocessing complete! Files saved to data/processed")

if __name__ == "__main__":
    preprocess_and_save()
