# 🧠 MoodGuard 2.0 — AI Mental State Intelligence System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.38.0-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.9.0-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10-00897B?style=for-the-badge)
![Gemini](https://img.shields.io/badge/Google_Gemini-1.5_Flash-4285F4?style=for-the-badge&logo=google&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**A complete AI-powered platform for real-time emotion detection, focus tracking, and behavioral intelligence.**

[Live Demo](https://moodguard-v-2.streamlit.app/) · [Report Bug](https://github.com/Hamna-Munir/MoodGuard_2.0/issues) · [Request Feature](https://github.com/Hamna-Munir/MoodGuard_2.0/issues)

</div>

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Problem Statement](#-problem-statement)
- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Module Details](#-module-details)
- [Screenshots](#-screenshots)
- [Developer](#-developer)

---

## 🔍 Overview

**MoodGuard 2.0** is a 7-module AI Mental State Intelligence System built with Python and Streamlit. It detects human emotions in real-time using a Convolutional Neural Network (CNN), tracks attention levels using eye landmark analysis, generates behavioral insights through a rule-based agent, and delivers personalized productivity advice using Google Gemini 1.5 Flash.


---

## 🚨 Problem Statement

Most productivity tools tell you *what* to do — but none understand *how you feel* while doing it.

Students and professionals lose focus without realizing it. By the time they notice, valuable time is already lost. There was no single tool that combined:

- 👁️ Real-time emotional awareness
- 🎯 Attention and focus scoring
- 🧠 Behavioral pattern analysis
- ✦ AI-powered personalized advice

**MoodGuard 2.0 solves this.**

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| 😊 **Emotion Detection** | Detects 7 emotions (Happy, Sad, Angry, Neutral, Fear, Surprise, Disgust) in real-time using CNN + ONNX Runtime |
| 🎯 **Focus Scoring** | Generates a 0–100 attention score using MediaPipe eye landmarks + RandomForest classifier |
| 🧠 **Behavioral Intelligence** | Rule-based agent detects stress, distraction, and peak states — then triggers automatic recommendations |
| ✦ **AI Insights** | Google Gemini 1.5 Flash analyzes your session and generates personalized productivity advice |
| 📊 **Session Analytics** | Emotion timeline, focus trends, mental state frequency charts — all logged and visualized |
| 📷 **Photo Analysis** | Upload any image to detect emotion probabilities |
| 📁 **History Tracking** | All sessions saved to CSV with full download support |

---

## 🏗️ System Architecture

```
Webcam / Photo
      ↓
Face Detection (OpenCV Haar Cascade)
      ↓
┌─────────────────────────────────────┐
│         AI Processing Layer         │
│                                     │
│  Emotion Detection    Focus Scoring │
│  (CNN → ONNX)        (MediaPipe +  │
│                        RandomForest)│
└─────────────────────────────────────┘
      ↓
Behavioral Intelligence Engine
(Rule-based: state classification + alerts)
      ↓
Google Gemini 1.5 Flash
(Personalized insights)
      ↓
Streamlit Dashboard
(8-page interactive UI)
```

---

## 🛠️ Tech Stack

### Core AI/ML
| Component | Technology | Purpose |
|-----------|-----------|---------|
| Emotion Model | CNN trained on FER2013 → ONNX | Classifies 7 emotions |
| Focus Model | RandomForest (scikit-learn) | Predicts attention level |
| Face Tracking | MediaPipe Face Mesh | 468 landmark extraction |
| AI Insights | Google Gemini 1.5 Flash | Natural language productivity advice |

### Backend & Frontend
| Component | Technology |
|-----------|-----------|
| Framework | Streamlit 1.38.0 |
| Computer Vision | OpenCV 4.9.0 (headless) |
| Inference Runtime | ONNX Runtime 1.20.1 |
| Charts | Plotly |
| Data | Pandas, NumPy |

### Deployment
| Component | Technology |
|-----------|-----------|
| Platform | Streamlit Community Cloud |
| Python | 3.11 |
| Version Control | GitHub |

---

## 📁 Project Structure

```
MoodGuard_2.0/
│
├── app.py                      ← Main Streamlit application (8 pages)
├── moodguard_model.onnx        ← Trained CNN emotion model (ONNX format)
├── focus_model.pkl             ← Trained RandomForest focus classifier
│
├── modules/
│   ├── __init__.py
│   ├── emotion_detector.py     ← CNN inference via ONNX Runtime
│   ├── focus_detector.py       ← Eye landmark analysis + ML classification
│   ├── behavior_engine.py      ← Rule-based behavioral state machine
│   └── gemini_insights.py      ← Google Gemini API integration
│
├── utils/
│   ├── __init__.py
│   └── logger.py               ← CSV session logging
│
├── data/
│   └── sessions.csv            ← Auto-generated session history
│
├── requirements.txt            ← Python dependencies
├── packages.txt                ← System-level dependencies
└── .env                        ← API keys (local only, not committed)
```

---

## ⚙️ Installation

### Prerequisites
- Python 3.11+
- pip or conda
- Webcam (for live detection)
- Google Gemini API key ([Get one free](https://aistudio.google.com/app/apikey))

### Clone & Setup

```bash
# 1. Clone the repository
git clone https://github.com/Hamna-Munir/MoodGuard_2.0.git
cd MoodGuard_2.0

# 2. Create virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Create .env file
echo GEMINI_API_KEY=your_api_key_here > .env

# 5. Run the app
streamlit run app.py
```

The app will open at `http://localhost:8501`

---

## 🚀 Usage

### Dashboard
The main page shows real-time emotion + focus analysis. Take a snapshot using the camera input to get instant results.

### Photo Analysis
Upload any image (JPEG/PNG) to detect emotions with probability scores for all 7 emotion classes.

### Live Camera
Continuous snapshot-based analysis with automatic session logging.

### Analytics
View session history as interactive charts — emotion distribution pie chart, focus timeline, mental state frequency bar chart.

### AI Insights
Click "Generate AI Insight" to receive a personalized 3-sentence productivity and wellness report powered by Google Gemini.

### Agent Log
View the rule-based behavioral agent's decision history — what rules fired, what actions were taken.

### History
Download complete session data as CSV for external analysis.

---

## 📦 Module Details

### `emotion_detector.py`
- Loads `moodguard_model.onnx` via ONNX Runtime
- Detects faces using OpenCV Haar Cascade
- Preprocesses face ROI: grayscale → resize 48×48 → normalize
- Outputs: dominant emotion, confidence score, all 7 class probabilities, face bounding box

### `focus_detector.py`
- Uses MediaPipe Face Mesh to extract 468 3D landmarks
- Calculates Eye Aspect Ratio (EAR) from 6 landmark points per eye
- Features: left EAR, right EAR, average EAR, eye width ratios, inter-eye distance
- Classifies focus as Focused / Distracted using RandomForest
- Tracks blink count across session

### `behavior_engine.py`
Rules engine that maps emotion + focus combinations to behavioral states:

| Condition | State | Action |
|-----------|-------|--------|
| Happy + Focus > 70 | Peak State | Recommend deep work |
| Neutral + Focus > 60 | Steady Focus | Continue current task |
| Angry / Fear + any focus | Stress Detected | Breathing exercise |
| Any + Focus < 30 | Low Focus | Pomodoro technique |
| Sad + Focus < 30 | Low Motivation | Take a walk |

### `gemini_insights.py`
- Tries models in order: `gemini-2.0-flash` → `gemini-1.5-flash` → `gemini-1.5-flash-latest`
- Falls back to rule-based insight on quota exhaustion
- Reads API key from Streamlit Secrets (cloud) or `.env` (local)

---

## 👩‍💻 Developer

<table>
  <tr>
    <td align="center">
      <strong>Hamna Munir</strong><br/>
      AI/ML Engineer · Software Engineering Student<br/>
      <a href="https://github.com/Hamna-Munir">GitHub</a> ·
      <a href="https://linkedin.com/in/hamna-munir">LinkedIn</a>
    </td>
  </tr>
</table>

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- [FER2013 Dataset](https://www.kaggle.com/datasets/msambare/fer2013) — emotion training data
- [MediaPipe](https://developers.google.com/mediapipe) — face landmark detection
- [Streamlit](https://streamlit.io) — app framework
- [Google Gemini](https://deepmind.google/technologies/gemini/) — AI insights

---

<div align="center">
  <strong>MoodGuard 2.0</strong> · Built with ❤️ by Hamna Munir · 2026
</div>
