import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TensorFlow logging

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

# Set random seed for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

def create_cnn_model(input_shape=(128, 128, 3)):
    """Create a CNN model with stronger regularization"""
    model = models.Sequential([
        # Input layer with explicit input shape
        layers.Input(shape=input_shape),
        
        # First Convolutional Block
        layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Second Convolutional Block
        layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Third Convolutional Block
        layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Dense Layers
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(4, activation='softmax')
    ])
    
    # Simplified compilation with fixed initial learning rate
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def test_environment():
    """Test all components before training"""
    try:
        print("Testing model creation...")
        model = create_cnn_model()
        
        print("Testing data loading...")
        data_dir = os.path.join("data", "raw", "real")
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Data directory not found: {data_dir}")
            
        print("Testing data generator...")
        test_datagen = ImageDataGenerator(rescale=1./255)
        test_generator = test_datagen.flow_from_directory(
            data_dir,
            target_size=(128, 128),
            batch_size=1,
            class_mode='categorical',
            shuffle=False
        )
        
        print("Testing model prediction...")
        test_batch = next(test_generator)
        pred = model.predict(test_batch[0])
        
        print("Testing model saving...")
        save_path = 'models'
        os.makedirs(save_path, exist_ok=True)
        model.save(os.path.join(save_path, 'test.keras'))
        os.remove(os.path.join(save_path, 'test.keras'))
        
        print("\n✅ All tests passed! Safe to proceed with training.")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        return False

def main():
    # Test environment first
    if not test_environment():
        print("Environment test failed. Please fix issues before training.")
        return
        
    # Data directories
    data_dir = os.path.join("data", "raw", "real")
    classes = ['pure', 'glucose', 'adulterated', 'pathogens']
    
    # Create data generators with augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        validation_split=0.2,
        fill_mode='nearest'
    )
    
    print("Loading and preprocessing data...")
    
    # Load training data
    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=(128, 128),
        batch_size=32,
        class_mode='categorical',
        subset='training',
        classes=classes,
        shuffle=True
    )
    
    # Load validation data
    validation_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=(128, 128),
        batch_size=32,
        class_mode='categorical',
        subset='validation',
        classes=classes,
        shuffle=True
    )
    
    # Create and train model
    model = create_cnn_model()
    print("\nModel Architecture:")
    model.summary()
    
    # Callbacks for model training
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath='models/milk_classifier_cnn.keras',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
    ]
    
    # Train the model
    print("\nTraining CNN model...")
    model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=50,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save final model
    model.save('models/milk_classifier_cnn.keras', save_format='keras')
    print("\nTraining complete! Model saved as 'models/milk_classifier_cnn.keras'")

if __name__ == "__main__":
    main()