# Politician Ideology Prediction (NLP + ML Pipeline)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0+-orange.svg)](https://www.tensorflow.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-latest-green.svg)](https://scikit-learn.org/)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Transformers-yellow)](https://huggingface.co/)

This project builds an **NLP pipeline** to **scrape U.S. politicians' websites**, process their text, and predict their **ideology score** (DW-NOMINATE) using **Machine Learning & Deep Learning** models.

**Goal:** Predict **political ideology** based on text data and compare different NLP models.

<p align="center">
  <img src="https://via.placeholder.com/800x400?text=NLP+Pipeline+Visualization" alt="NLP Pipeline Visualization" width="700"/>
</p>

---

## **Table of Contents**
- [Project Structure](#project-structure)
- [Installation](#%EF%B8%8F-installation)
- [Quick Start](#-quick-start)
- [Pipeline Workflow](#pipeline-workflow)
- [Dataset Details](#-dataset-details)
- [Model Descriptions](#model-descriptions)
- [Results](#-results)
- [Key Files & Their Purpose](#key-files--their-purpose)
- [Future Work](#-future-work)
- [Contributing](#-contributing)
- [License](#license)

---

## **Project Structure**

```bash
nlp-predicting-ideology/
│── data/                      # 📂 Stores datasets
│   ├── final_dataset.csv      # Cleaned dataset with processed text & DW-NOMINATE scores
│   ├── merged_politicians.csv # Merged data before preprocessing
│   ├── politicians.json       # Scraped website text
│
│── models/                    # 📂 Trained models
│   ├── ridge_regression.pkl   # Ridge Regression model
│   ├── bert_model/            # BERT fine-tuned model
│   ├── metrics/               # Model performance metrics
│
│── src/                       # 📂 Core code (training & preprocessing)
│   ├── embedding_utils.py     # Handles text embeddings (TF-IDF, BERT)
│   ├── preprocess.py          # Preprocesses text (cleans data, removes NaN, filters word count)
│   ├── train_ridge.py         # Trains Ridge Regression model
│   ├── train_bert.py          # Fine-tunes BERT model
│   ├── evaluate.py            # Compares all trained models
│
│── politician_scraper/        # 📂 Web scraping (Scrapy framework)
│   ├── spiders/
│   │   ├── politicians.py     # Scrapy spider script to scrape website text
│
│── requirements.txt           # 🔧 Project dependencies
│── README.md                  # 📚 Project documentation
```

---

## **🛠️ Installation**
### **Clone the Repository**
```bash
git clone https://github.com/yourusername/nlp-predicting-ideology.git
cd nlp-predicting-ideology
```

### **Create a Virtual Environment (Optional)**
```bash
python -m venv venv
source venv/bin/activate  # On Mac/Linux
venv\Scripts\activate     # On Windows
```

### **Install Dependencies**
```bash
pip install -r requirements.txt
```

---

## **🚀 Quick Start**

To get started quickly with a pre-trained model:

```python
from src.embedding_utils import preprocess_text
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import numpy as np

# Load pre-trained BERT model and tokenizer
model_name = "models/bert_model"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)

# Example text from politician's website
text = "We must ensure healthcare is accessible to all Americans while reducing costs..."

# Preprocess text
processed_text = preprocess_text(text)

# Get predictions
inputs = tokenizer(processed_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
with torch.no_grad():
    outputs = model(**inputs)

# Load normalization parameters and denormalize prediction
norm_params = np.load('models/bert_norm_params.npz')
prediction = outputs.logits.item() * norm_params['std'] + norm_params['mean']
print(f"Predicted ideology score: {prediction:.2f}")  # Scale: -1 (liberal) to 1 (conservative)
```

---

## **Pipeline Workflow**
The pipeline follows these steps:

<p align="center">
  <img src="https://via.placeholder.com/700x300?text=Pipeline+Workflow+Diagram" alt="Pipeline Workflow" width="600"/>
</p>

1. **Scrape Politicians' Websites** (`politician_scraper/`)
```bash
cd politician_scraper
python -m scrapy crawl politicians -o ../data/politicians.json
```

2. **Preprocess Text** (`preprocess.py`)
```bash
python src/preprocess.py
```

3. **Train Different Models**
```bash
python src/train_ridge.py
python src/train_bert.py
```

4. **Evaluate All Models** (`evaluate.py`)
```bash
python src/evaluate.py
```

---

## **📊 Dataset Details**

The dataset combines:

- **Website Text Data**: Scraped from official websites of members of the U.S. Congress
- **DW-NOMINATE Scores**: Standard political science measure of ideology
  - Range from -1 (most liberal) to 1 (most conservative)
  - First dimension captures economic ideology
  - Second dimension captures social/cultural issues

**Data Statistics:**
- 435 House representatives + 100 Senators
- ~2,500 words average per website
- Balanced representation across political spectrum

**Data Split:**
- Training: 70% (374 politicians)
- Validation: 15% (80 politicians)
- Testing: 15% (81 politicians)

---

## **Model Descriptions**
| **Model** | **Embeddings Used** | **Pros** | **Cons** |
|-----------|---------------------|----------|----------|
| **Ridge Regression** | TF-IDF | Fast & interpretable | Doesn't capture context |
| **BERT Fine-Tuning** | BERT | Strong contextual understanding | Computationally intensive |

---

## **📈 Results**

### Performance Metrics (Test Set)

| **Model** | **MSE** | **RMSE** | **MAE** | **R²** |
|-----------|---------|----------|---------|--------|
| Ridge Regression | 0.130 | 0.361 | 0.288 | 0.381 |
| BERT | 0.120 | 0.346 | 0.241 | 0.432 |

<p align="center">
  <img src="https://via.placeholder.com/600x400?text=Model+Comparison+Chart" alt="Model Comparison" width="500"/>
</p>

### Key Findings

- **BERT** achieves better performance with an R² score of 0.432, but requires significant computational resources
- Ridge Regression offers decent performance (R² = 0.381) with much faster training
- Most common misclassifications occur for moderate politicians
- The relatively modest R² scores indicate the challenging nature of predicting political ideology from text alone

---

## **Key Files & Their Purpose**
| **File** | **Description** |
|----------|---------------|
| `preprocess.py` | Cleans and filters the dataset before training |
| `embedding_utils.py` | Converts text into TF-IDF or BERT embeddings |
| `train_ridge.py` | Trains Ridge Regression with TF-IDF |
| `train_bert.py` | Fine-tunes BERT on our dataset |
| `evaluate.py` | Compares all trained models and prints evaluation metrics |

---

## **License**
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## **Author**
**Aryah Rao** - [GitHub]([https](https://github.com/aryah-rao))