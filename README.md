# 🎨 DrawMe — AI Sketch Recognition

An AI-powered drawing recognition app that uses a Convolutional Neural Network (CNN) trained on Google's [Quick, Draw!](https://quickdraw.withgoogle.com/data) dataset to identify hand-drawn doodles in real-time.

![Python](https://img.shields.io/badge/Python-3.10--3.12-3776AB?style=flat-square&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-2.x-000000?style=flat-square&logo=flask&logoColor=white)

---

## ✨ Features

- **Interactive Canvas** — Draw with mouse or touch on a responsive HTML5 Canvas
- **Real-time Recognition** — CNN identifies your doodle from 15 categories
- **Beautiful Dark UI** — Premium glassmorphism design with micro-animations
- **Smart Preprocessing** — Bounding-box crop, downsampling, and color inversion
- **Undo Support** — Full stroke history with Ctrl+Z shortcut

## 🏷️ Recognized Categories

☁️ Cloud · ☀️ Sun · 🌳 Tree · 🚗 Car · 🐟 Fish · 🐱 Cat · 🐶 Dog · 🏠 House · ⭐ Star · 🌸 Flower · 🐦 Bird · 🚲 Bicycle · 🎸 Guitar · 🌙 Moon · 🎩 Hat

## 📂 Project Structure

```
DrawMe/
├── data/                    # Downloaded .npy dataset files (gitignored)
├── model/
│   ├── download_data.py     # Download Quick, Draw! .npy files
│   ├── train.py             # CNN training script
│   └── saved/               # Saved model + category mapping
├── api/
│   ├── app.py               # Flask API server
│   └── utils.py             # Image preprocessing pipeline
├── frontend/
│   ├── index.html           # Drawing canvas UI
│   ├── style.css            # Premium dark theme styles
│   └── script.js            # Canvas logic + API communication
└── requirements.txt
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
python -m venv venv
source venv/bin/activate       # macOS/Linux
pip install -r requirements.txt
```

Use Python 3.10-3.12 (recommended: 3.11). TensorFlow wheels are typically not available yet for very new Python versions.

### 2. Download the Dataset
```bash
python model/download_data.py
```
Downloads 15 categories (~1.5 GB total) from Google Cloud Storage.

### 3. Train the Model
```bash
python model/train.py --profile default
```
This runs the resource-friendly profile for smaller machines.

For a bigger GPU machine:
```bash
python model/train.py --profile high_end
```

You can also override any setting with your own JSON file:
```bash
python model/train.py --profile high_end --config /path/to/your_config.json
```

Model artifacts are saved in `model/saved/`:
- `drawme_model.keras`
- `drawme_model.h5`
- `categories.json`
- `training_history.json`
- `training_metadata.json`
- `resolved_config.json`

Built-in profiles:
- `model/configs/default.json`: smaller batch/data/model defaults
- `model/configs/high_end.json`: larger model + mixed precision for stronger GPUs

Quick runtime check without training:
```bash
python model/train.py --profile high_end --dry-run
```

### 4. Start the App
```bash
python api/app.py
```
Open **http://localhost:5000** in your browser, draw something, and click **Predict**!

## 🧠 Model Architecture

```
Input (28×28×1)
  → Conv2D(32, 3×3, ReLU) → MaxPool(2×2)
  → Conv2D(64, 3×3, ReLU) → MaxPool(2×2)
  → Flatten → Dense(128, ReLU) → Dropout(0.3)
  → Dense(15, Softmax)
```

- **Optimizer**: Adam
- **Loss**: Sparse Categorical Crossentropy
- **Class Balancing**: 10,000 samples per category

The training pipeline now supports profile-based architecture scaling:
- Small machine profile: 3 conv blocks and moderate batch/data size
- Big machine profile: deeper/wider CNN, larger dense head, more samples, and optional mixed precision

## 🔧 Image Preprocessing Pipeline

1. **Decode** Base64 PNG from canvas
2. **Convert** to grayscale
3. **Invert** colors (black-on-white → white-on-black)
4. **Crop** to bounding box of drawn content
5. **Pad** to square aspect ratio
6. **Resize** to 28×28 pixels
7. **Normalize** pixel values to [0, 1]

## ⌨️ Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `Enter` | Submit prediction |
| `Ctrl+Z` | Undo last stroke |
| `Esc` | Clear canvas |

## 📊 Tech Stack

| Component | Technology |
|-----------|------------|
| ML Framework | TensorFlow / Keras |
| Backend | Flask (Python) |
| Frontend | Vanilla HTML5 + CSS + JS |
| Dataset | Google Quick, Draw! |

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.