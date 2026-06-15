# 🎭 AI-Powered Emotion Detection from Text

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-orange?logo=scikit-learn)](https://scikit-learn.org/)
[![NLTK](https://img.shields.io/badge/NLTK-3.x-green)](https://www.nltk.org/)
[![HuggingFace](https://img.shields.io/badge/Dataset-HuggingFace-yellow?logo=huggingface)](https://huggingface.co/datasets/dair-ai/emotion)
[![License: MIT](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)

> An NLP-based machine learning system that classifies text into six human emotions — Sadness, Joy, Love, Anger, Fear, and Surprise — using TF-IDF vectorization and Logistic Regression.

---

## 📌 Table of Contents

- [Project Overview](#-project-overview)
- [Demo](#-demo)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Results](#-results)
- [Tech Stack](#-tech-stack)
- [Future Scope](#-future-scope)
- [License](#-license)

---

## 🧠 Project Overview

Humans naturally understand emotions in text, but machines struggle with this. This project builds an **end-to-end NLP pipeline** that:

1. Loads and explores the `dair-ai/emotion` dataset (20,000 labeled samples)
2. Preprocesses raw text (lowercasing, stopword removal, lemmatization)
3. Converts text to numerical features via **TF-IDF**
4. Trains a **Logistic Regression** classifier
5. Evaluates with accuracy, precision, recall, F1-score, and confusion matrix
6. Exposes a `predict_emotion()` function for real-time inference

**Use cases:** Customer support sentiment analysis, social media monitoring, mental health platforms, conversational AI.

---

## 🎬 Demo

```python
from src.predict import predict_emotion

print(predict_emotion("I got selected for an internship!"))
# → joy

print(predict_emotion("I lost my wallet"))
# → sadness

print(predict_emotion("I am terrified about exam results"))
# → fear
```

---

## 📊 Dataset

| Split      | Samples |
|------------|---------|
| Train      | 16,000  |
| Validation | 2,000   |
| Test       | 2,000   |

**Source:** [`dair-ai/emotion`](https://huggingface.co/datasets/dair-ai/emotion) on Hugging Face

**Emotion Classes:**

| Label | Emotion  |
|-------|----------|
| 0     | Sadness  |
| 1     | Joy      |
| 2     | Love     |
| 3     | Anger    |
| 4     | Fear     |
| 5     | Surprise |

> ⚠️ **Class Imbalance:** Joy (~5,000) and Sadness (~4,600) dominate; Surprise has fewer than 600 samples. This affects per-class performance.

---

## 📁 Project Structure

```
emotion_detection/
│
├── emotion_detection.ipynb     # Full project notebook (EDA → Training → Evaluation)
├── src/
│   ├── preprocess.py           # Text cleaning and NLP pipeline
│   └── predict.py              # Inference using saved model
├── models/
│   ├── emotion_model.pkl       # Trained Logistic Regression model
│   └── tfidf_vectorizer.pkl    # Fitted TF-IDF vectorizer
├── requirements.txt            # Python dependencies
├── setup.py                    # Package setup (optional)
├── .gitignore                  # Git ignore rules
└── README.md                   # Project documentation
```

> **Note:** `.pkl` model files are generated after running the notebook. They are excluded from version control via `.gitignore` (add them back if you want to ship pre-trained models).

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/emotion-detection.git
cd emotion-detection
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download NLTK resources

```python
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
```

---

## 🚀 Usage

### Option A — Run the Jupyter Notebook

```bash
jupyter notebook emotion_detection.ipynb
```

Run all cells from top to bottom. The notebook covers:
- Data loading and EDA
- Text preprocessing
- TF-IDF feature extraction
- Model training and evaluation
- Saving model artifacts

### Option B — Use the Prediction Script

After running the notebook (which saves the `.pkl` files):

```bash
python src/predict.py
```

Or import in your own script:

```python
from src.predict import predict_emotion

text = "I cannot believe I won the competition!"
print(predict_emotion(text))  # → joy
```

---

## 📈 Results

| Metric    | Score  |
|-----------|--------|
| Accuracy  | ~89%   |
| Precision | ~89%   |
| Recall    | ~89%   |
| F1-Score  | ~88%   |

> Exact scores may vary slightly depending on dataset version and environment. Run the notebook to reproduce results.

**Per-class Observations:**
- **Joy** and **Sadness** — highest F1 scores (most training data)
- **Surprise** — lowest recall (severe class imbalance, < 600 samples)
- **Love**, **Anger**, **Fear** — moderate performance

---

## 🛠️ Tech Stack

| Tool / Library     | Purpose                        |
|--------------------|--------------------------------|
| Python 3.8+        | Core language                  |
| Pandas & NumPy     | Data manipulation              |
| NLTK               | Text preprocessing (tokenize, lemmatize, stopwords) |
| Scikit-learn       | TF-IDF, Logistic Regression, metrics |
| Matplotlib & Seaborn | Visualization (EDA, confusion matrix) |
| Hugging Face `datasets` | Dataset loading            |
| Pickle             | Model serialization            |

---

## 🔭 Future Scope

- [ ] Handle class imbalance with SMOTE or class weighting
- [ ] Experiment with LSTM / GRU deep learning models
- [ ] Fine-tune pre-trained transformers (BERT, DistilBERT, RoBERTa)
- [ ] Build a Gradio / Streamlit web app for live demo
- [ ] Add multilingual support
- [ ] REST API deployment with FastAPI

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## 🙋 Author

**Your Name**
- LinkedIn: [linkedin.com/in/yourprofile](https://linkedin.com/in/yourprofile)
- GitHub: [github.com/yourusername](https://github.com/yourusername)

---

> ⭐ If you found this project helpful, please consider giving it a star!
