# YCRY - AI Cry Translator & Parenting Assistant 🧸

YCRY is a smart parenting assistant that uses AI to analyze baby cries and determine their needs (Hunger, Pain, Tiredness, etc.). It also features growth tracking, vaccine scheduling, and health monitoring.

## ✨ Features

- **🎙️ AI Cry Translator**: Upload audio of your baby's cry to understand why they are crying.
- **💉 Vaccine Tracker**: Automatic schedule based on birth date with "Overdue" alerts.
- **📏 Growth Journey**: Track height, weight, and milestones (Motor, Social, Communication).
- **👶 Profile Management**: personalized dashbaord for your baby.
- **🔐 Secure & Private**: User authentication with encrypted data.

## 🛠️ Tech Stack

- **Backend**: Flask (Python)
- **AI/ML**: TensorFlow / Keras (CNN Model)
- **Frontend**: HTML, TailwindCSS, JavaScript
- **Database**: SQLite
- **Audio Processing**: Librosa

## 🚀 How to Run

1.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Run the App**
    ```bash
    python app.py
    ```

3.  **Access**
    Open `http://127.0.0.1:5000` in your browser.

## 🧠 ML Model

The AI model is a Convolutional Neural Network (CNN) trained on Mel-spectrograms of baby cries.
To retain the model:
```bash
python train_model.py
```
