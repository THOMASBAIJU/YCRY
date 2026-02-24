import sys
import os
import traceback
import matplotlib

print("--- Starting Debug Script ---")

try:
    print("Trying import numpy...")
    import numpy as np
    print("✅ numpy imported")
except:
    print("❌ numpy failed")
    traceback.print_exc()

try:
    print("Trying import librosa...")
    import librosa
    print("✅ librosa imported")
except:
    print("❌ librosa failed")
    traceback.print_exc()

try:
    print("Trying import matplotlib...")
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    print("✅ matplotlib imported")
except:
    print("❌ matplotlib failed")
    traceback.print_exc()

try:
    print("Trying import tensorflow...")
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    print("✅ tensorflow imported")
except:
    print("❌ tensorflow failed")
    traceback.print_exc()

MODEL_PATH = os.path.join(os.getcwd(), "model_brain.h5")
print(f"Checking model at: {MODEL_PATH}")

if os.path.exists(MODEL_PATH):
    print("✅ Model file found. Attempting to load...")
    try:
        model = load_model(MODEL_PATH)
        print("✅ Model loaded successfully")
        
        # Test prediction (warmup)
        try:
            print("⏳ Testing prediction...")
            dummy_input = np.zeros((1, 64, 64, 3))
            model.predict(dummy_input, verbose=0)
            print("✅ Model prediction test passed")
        except:
            print("❌ Model prediction failed")
            traceback.print_exc()

    except:
        print("❌ Model load failed")
        traceback.print_exc()
else:
    print(f"❌ Model file not found at {MODEL_PATH}")

print("--- End Debug Script ---")
