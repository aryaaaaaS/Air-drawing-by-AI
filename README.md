# Air-drawing-by-AI
A real-time AI-powered air drawing system that allows users to draw shapes using their fingers in the air, track movements through webcam using MediaPipe and OpenCV, and predict drawn shapes using a trained Convolutional Neural Network (CNN). 
# ✍️ Air Drawing AI - Real-Time Shape Predictor

Draw shapes in the air using just your fingers, and watch AI detect what you've drawn — all in real time! Powered by Computer Vision and Machine Learning.

---

## 📌 Features

- 🖐️ **Finger-based Air Drawing** using **MediaPipe** + **OpenCV**
- 🧠 **CNN Trained** on custom hand-drawn shape dataset
- 🧽 Eraser mode for precise drawing
- 💾 Save your drawings with a single key
- 🧠 Press `'P'` to predict what you've drawn
- 🌈 After prediction, display an emoji/image of the shape (circle, square, triangle, etc.)
- 🎯 Real-time feedback on webcam canvas
- 💬 Optional: Add voice output (TTS)


## 🛠️ TechStack

- **Python**
- **OpenCV**
- **MediaPipe**
- **TensorFlow / Keras**
- **NumPy, Seaborn, Matplotlib**
- **LabelEncoder, Scikit-learn**



## 🎮 Controls

| Key | Action |
|-----|--------|
| `P` | Predict drawn shape |
| `S` | Save canvas as image |
| `C` | Clear canvas |
| `E` | Toggle eraser mode |
| `D` | Toggle data collection mode |
| `1-9` | Choose shape class in dataset mode |
| `Q` | Quit the app |

---

## 🧪 Training the Model

1. Prepare your shape dataset under `/dataset/shape_name/`
2. Run:
```bash
python model_trainer.py




