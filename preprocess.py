import os
import sys
os.environ["PATH"] += os.pathsep + r"C:\ffmpeg\bin"

import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import cv2

# --- CONFIGURATION ---
SOURCE_FOLDER = "raw_dataset"       # Folder containing your audio files (e.g., raw_dataset/hunger)
OUTPUT_FOLDER = "processed_images"  # Where we will save the spectrograms
SAMPLE_RATE = 22050
DURATION = 7                        # 7 seconds as per your report
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION

def save_spectrogram(signal, label, index, augmentation_name="orig"):
    """Generates and saves a Mel-Spectrogram image."""
    
    # Extract Mel Spectrogram
    mel_spectrogram = librosa.feature.melspectrogram(y=signal, sr=SAMPLE_RATE, n_mels=128, fmax=8000)
    mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)

    # Plot and save (without axes/labels to keep it pure image data)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mel_spectrogram, sr=SAMPLE_RATE, fmax=8000)
    plt.axis('off')
    
    # Create class folder if not exists
    save_path = os.path.join(OUTPUT_FOLDER, label)
    os.makedirs(save_path, exist_ok=True)
    
    # Save file: e.g., processed_images/hunger/hunger_001_noise.png
    filename = f"{label}_{index}_{augmentation_name}.png"
    plt.savefig(os.path.join(save_path, filename), bbox_inches='tight', pad_inches=0)
    plt.close()

def augment_audio(signal, sample_rate):
    """Returns a list of augmented versions of the signal."""
    augmented_signals = []
    
    # 1. Original
    augmented_signals.append(("original", signal))
    
    # 2. Add Noise (Simulating real-world environment)
    noise = np.random.randn(len(signal))
    noise_factor = 0.005
    signal_noise = signal + noise_factor * noise
    augmented_signals.append(("noise", signal_noise))
    
    # 3. Time Stretch (Slightly faster/slower)
    # Note: Time stretch changes duration, so we must fix length after
    signal_stretch = librosa.effects.time_stretch(signal, rate=0.9)
    if len(signal_stretch) > len(signal):
        signal_stretch = signal_stretch[:len(signal)]
    else:
        signal_stretch = np.pad(signal_stretch, (0, len(signal) - len(signal_stretch)))
    augmented_signals.append(("stretch", signal_stretch))
    
    return augmented_signals

def process_dataset():
    if not os.path.exists(SOURCE_FOLDER):
        print(f"❌ Error: Source folder '{SOURCE_FOLDER}' not found. Please create it and add your audio folders.")
        return

    print("🚀 Starting Data Pipeline...")
    
    # Loop through each class folder (Hunger, Pain, etc.)
    for label in os.listdir(SOURCE_FOLDER):
        class_path = os.path.join(SOURCE_FOLDER, label)
        
        if os.path.isdir(class_path):
            print(f"   Processing Class: {label}...")
            
            for i, filename in enumerate(os.listdir(class_path)):
                file_path = os.path.join(class_path, filename)
                
                try:
                    # Load Audio
                    audio, _ = librosa.load(file_path, sr=SAMPLE_RATE)
                    
                    # Pad or Truncate to exact duration (7s)
                    if len(audio) > SAMPLES_PER_TRACK:
                        audio = audio[:SAMPLES_PER_TRACK]
                    else:
                        padding = SAMPLES_PER_TRACK - len(audio)
                        audio = np.pad(audio, (0, padding))
                    
                    # Augment Data
                    signals = augment_audio(audio, SAMPLE_RATE)
                    
                    # Generate Spectrograms for each version
                    for aug_name, signal_ver in signals:
                        save_spectrogram(signal_ver, label, i, aug_name)
                        
                except Exception as e:
                    print(f"   ⚠️ Could not process {filename}: {e}")

    print("✅ Processing Complete! Images ready for MobileNetV2.")

if __name__ == "__main__":
    process_dataset()
