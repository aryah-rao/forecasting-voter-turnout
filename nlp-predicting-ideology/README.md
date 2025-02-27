# Politician Ideology Prediction (NLP + ML Pipeline)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0+-orange.svg)](https://www.tensorflow.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-latest-green.svg)](https://scikit-learn.org/)

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
│   ├── xgboost.pkl            # XGBoost model
│   ├── lstm.h5                # LSTM model
│   ├── bert_model/            # BERT fine-tuned model
│
│── src/                       # 📂 Core code (training & preprocessing)
│   ├── embedding_utils.py     # Handles text embeddings (TF-IDF, Word2Vec, BERT)
│   ├── preprocess.py          # Preprocesses text (cleans data, removes NaN, filters word count)
│   ├── train_ridge.py         # Trains Ridge Regression model
│   ├── train_xgboost.py       # Trains XGBoost model
│   ├── train_lstm.py          # Trains LSTM model
│   ├── train_bert.py          # Fine-tunes BERT model
│   ├── evaluate_models.py     # Compares all trained models
│
│── politician_scraper/        # 📂 Web scraping (Scrapy framework)
│   ├── spiders/
│   │   ├── politicians.py     # Scrapy spider script to scrape website text
│
│── notebooks/                 # 📓 Jupyter notebooks for exploration and visualization
│   ├── exploratory_analysis.ipynb  # Data exploration and visualization
│   ├── model_comparison.ipynb      # Compare performance across models
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
from src.embedding_utils import preprocess_text, get_bert_embeddings
from joblib import load
import pandas as pd

# Load pre-trained model
model = load('models/xgboost.pkl')

# Example text from politician's website
text = "We must ensure healthcare is accessible to all Americans while reducing costs..."

# Preprocess text
processed_text = preprocess_text(text)

# Get embeddings
embeddings = get_bert_embeddings(processed_text)

# Predict ideology score (-1 = liberal, 1 = conservative)
ideology_score = model.predict(embeddings)[0]
print(f"Predicted ideology score: {ideology_score:.2f}")
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
python src/train_xgboost.py
python src/train_lstm.py
python src/train_bert.py
```

4. **Evaluate All Models** (`evaluate_models.py`)
```bash
python src/evaluate_models.py
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
| **XGBoost** | Word2Vec | Captures non-linearity | Slower than Ridge |
| **LSTM** | Word2Vec | Sequential learning | Requires large dataset |
| **BERT Fine-Tuning** | BERT | Best accuracy | Requires GPU |

---

## **📈 Results**

### Performance Metrics (Test Set)

| **Model** | **MAE** | **RMSE** | **R²** | **Training Time** |
|-----------|---------|----------|--------|------------------|
| Ridge Regression | 0.215 | 0.267 | 0.61 | 3.2s |
| XGBoost | 0.182 | 0.229 | 0.72 | 45.1s |
| LSTM | 0.166 | 0.209 | 0.77 | 15m 32s |
| BERT | 0.112 | 0.154 | 0.86 | 3h 22m |

<p align="center">
  <img src="https://via.placeholder.com/600x400?text=Model+Comparison+Chart" alt="Model Comparison" width="500"/>
</p>

### Key Findings

- **BERT** achieves the best performance but requires significant computational resources
- **LSTM** offers a good balance between accuracy and training time
- Most common misclassifications occur for moderate politicians
- Feature importance analysis shows economic terms are strongest predictors

---

## **Key Files & Their Purpose**
| **File** | **Description** |
|----------|---------------|
| `preprocess.py` | Cleans and filters the dataset before training |
| `embedding_utils.py` | Converts text into TF-IDF, Word2Vec, or BERT embeddings |
| `train_ridge.py` | Trains Ridge Regression with TF-IDF |
| `train_xgboost.py` | Trains XGBoost with Word2Vec |
| `train_lstm.py` | Trains an LSTM using Word2Vec |
| `train_bert.py` | Fine-tunes BERT on our dataset |
| `evaluate_models.py` | Compares all trained models and prints evaluation metrics |

---

## **🔮 Future Work**

Potential improvements and extensions:

1. **Cross-lingual ideology prediction** - Apply to politicians from different countries
2. **Temporal analysis** - Track ideological shifts over time
3. **Multi-modal analysis** - Incorporate speech, voting records, and social media
4. **Active learning** - Reduce annotation costs for new politicians
5. **Explainable AI techniques** - Better interpret model predictions

---

## **🤝 Contributing**

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## **License**
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## **Contact**

Project Maintainer: [Your Name](https://github.com/yourusername)

For questions or feedback, please [open an issue](https://github.com/yourusername/nlp-predicting-ideology/issues) or contact [youremail@example.com](mailto:youremail@example.com).
