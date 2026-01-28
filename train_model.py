import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.utils import class_weight
import numpy as np
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
DATASET_DIR = "processed_images"
IMG_SIZE = (64, 64)     
BATCH_SIZE = 32
EPOCHS = 50             # Increased epochs, but EarlyStopping will handle it
LEARNING_RATE = 0.001

def build_custom_cnn(num_classes):
    """
    An improved CNN with BatchNormalization and more capacity.
    """
    model = Sequential([
        Input(shape=(64, 64, 3)),
        
        # Layer 1: Basic Edges
        Conv2D(32, (3, 3), activation='relu', kernel_regularizer=l2(0.0001)),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        
        # Layer 2: Textures
        Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2(0.0001)),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        
        # Layer 3: Complex Patterns
        Conv2D(128, (3, 3), activation='relu', kernel_regularizer=l2(0.0001)),
        BatchNormalization(),
        MaxPooling2D(2, 2),

        # Layer 4: High-level Features (New)
        Conv2D(256, (3, 3), activation='relu', kernel_regularizer=l2(0.0001)),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        
        # Flatten & Classify
        Flatten(),
        Dense(256, activation='relu', kernel_regularizer=l2(0.0001)),
        Dropout(0.5),  # Prevent overfitting
        Dense(num_classes, activation='softmax')
    ])
    return model

def train_custom_cnn():
    print(f"🚀 Detected {os.cpu_count()} CPU cores. Configuring TensorFlow...")
    
    # 1. Image Generators (With Enhanced Augmentation)
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,
        width_shift_range=0.15,  # Increased slightly
        height_shift_range=0.0,
        zoom_range=0.15,
        shear_range=0.1,        # Added shear
        fill_mode='nearest'
    )

    print("   Loading Training Data (Resizing to 64x64)...")
    train_generator = train_datagen.flow_from_directory(
        DATASET_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )

    val_generator = train_datagen.flow_from_directory(
        DATASET_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation'
    )

    if train_generator.samples == 0:
        print("❌ No images found! Please check 'processed_images' folder.")
        return

    num_classes = len(train_generator.class_indices)
    
    # 2. Calculate Class Weights
    print("⚖️  Calculating Class Weights...")
    class_weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_generator.classes),
        y=train_generator.classes
    )
    class_weights_dict = dict(enumerate(class_weights))
    print(f"   Weights: {class_weights_dict}")

    # 3. Build & Compile
    model = build_custom_cnn(num_classes)
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # 4. Callbacks (Smarter Training)
    callbacks = [
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=0.00001, verbose=1),
        EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True, verbose=1)
    ]

    # 5. Train
    print("🧠 Starting Improved Training...")
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=val_generator,
        class_weight=class_weights_dict,
        callbacks=callbacks
    )

    # 6. Save & Plot
    model.save("ycry_custom_cnn.h5")
    print("🎉 Model saved as 'ycry_custom_cnn.h5'")
    
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Improved CNN Performance')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(loc='lower right')
    plt.show()

if __name__ == "__main__":
    train_custom_cnn()