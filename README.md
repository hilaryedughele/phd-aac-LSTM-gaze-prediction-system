# Assistive Augmentative Communication (AAC) System

This project implements a next-generation **Assistive Augmentative Communication (AAC)** system designed for individuals with neurological disorders. It combines **eye tracking**, **speech recognition**, **LSTM-based gaze prediction**, **blink detection**, and **gamification** into a real-time, multimodal communication interface using a `Tkinter` GUI.

---

## 🔍 Features

- 👁️ **Real-time Eye Tracking** using MediaPipe
- 🗣️ **Speech Recognition** for GUI control and object selection
- 🧠 **LSTM-Based Gaze Prediction** to highlight likely targets
- 👀 **Blink Detection** and **Dwell-Based Target Selection**
- 🎮 **Gamification Mode** with scoring and feedback
- 📁 **Custom Model Support** – Upload your own LSTM model via the GUI

---

## 🖥️ System Requirements

- Python 3.8 or later
- Windows 10/11 (tested on Windows 11 Home)
- Webcam for eye tracking
- Microphone for voice input

---

## 📦 Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/YOUR_USERNAME/AAC03062025v2.git
   cd AAC03062025v2
   
2. **Install dependencies**:
 
pip install -r requirements.txt

3. **(Windows Only) – If pyaudio fails to install**:
pip install pipwin
pipwin install pyaudio

🚀 Usage

1. Run the application:
python testingAAC.py

🧠 LSTM Gaze Prediction
2-layer LSTM trained on:

EEGET-ALS dataset (ALS vs. healthy users)

Inputs: Normalized gaze vectors with sliding window

Outputs: Predicted target positions

Evaluation: MAE, RMSE

🎮 Game Mode

Interactive game mode with:

Visual + audio feedback

Real-time scoring

Adaptive difficulty based on user performance



📚 Citation
If you use this project in your research or publication, please cite:

@misc{aac2025,
  author       = {Hilary O. Edughele},
  title        = {Eye-Tracking and Speech-Based AAC System with LSTM Gaze Prediction},
  year         = {2025},
  howpublished = {\url{https://github.com/hilaryedughele/phd-aac-LSTM-gaze-prediction-system}}
}

🙋 Acknowledgments

MediaPipe by Google

TensorFlow/Keras team

EEGET-ALS Dataset contributors

OpenCV and SpeechRecognition libraries
