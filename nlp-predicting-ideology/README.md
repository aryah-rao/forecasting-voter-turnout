# Politician Ideology Prediction (NLP + ML Pipeline)

This project builds an **NLP pipeline** to **scrape U.S. politicians' websites**, process their text, and predict their **ideology score** (DW-NOMINATE) using **Machine Learning & Deep Learning** models.

**Goal:** Predict **political ideology** based on text data and compare different NLP models.

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

## **Pipeline Workflow**
The pipeline follows these steps:

**Scrape Politicians' Websites** (`politician_scraper/`)
```bash
cd politician_scraper
python -m scrapy crawl politicians -o ../data/politicians.json
```

**Preprocess Text** (`preprocess.py`)
```bash
python src/preprocess.py
```

**Train Different Models**
```bash
python src/train_ridge.py
python src/train_xgboost.py
python src/train_lstm.py
python src/train_bert.py
```

**Evaluate All Models** (`evaluate_models.py`)
```bash
python src/evaluate_models.py
```

---

## **Model Descriptions**
| **Model** | **Embeddings Used** | **Pros** | **Cons** |
|-----------|---------------------|----------|----------|
| **Ridge Regression** | TF-IDF | Fast & interpretable | Doesn't capture context |
| **XGBoost** | Word2Vec | Captures non-linearity | Slower than Ridge |
| **LSTM** | Word2Vec | Sequential learning | Requires large dataset |
| **BERT Fine-Tuning** | BERT | Best accuracy | Requires GPU |

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

## **License**
MIT License. Free to use and modify.
