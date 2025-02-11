# Politician Ideology Analysis Tool

This project is an **NLP pipeline** that scrapes U.S. politicians' websites, processes their text, and assigns them an **ideology score** (liberal to conservative) based on **DW-NOMINATE** scores. The goal is to highlight discrepancies between **party labels** and politicians' **actual ideological positions**.

## 📂 Project Structure

```bash
politician_ideology_project/
│── data/                       # Stores scraped data, DW-NOMINATE scores, and processed text
│   ├── politicians.json        # Scraped websites' text data
│   ├── DW-NOMINATE.csv         # Ground truth ideology scores (downloaded from Voteview)
│   ├── processed_data.csv      # Cleaned and preprocessed dataset
│
│── models/                     # Stores trained models
│   ├── ridge_regression.pkl    # 
│
│── notebooks/                  # Jupyter Notebooks for exploration & debugging
│   ├── 01_scraping.ipynb
│   ├── 02_feature_extraction.ipynb
│   ├── 03_model_training.ipynb
│   ├── 04_visualization.ipynb
│
│── politician_scraper/         # Scrapy spider project for web scraping
│   ├── politician_scraper/
│   │   ├── spiders/
│   │   │   ├── politicians.py  # Scrapy spider script
│   │   ├── settings.py
│   │   ├── pipelines.py
│   ├── scrapy.cfg
│
│── src/                        # Main Python scripts for processing & training
│   ├── scraping.py             # Calls Scrapy to scrape websites
│   ├── preprocess.py           # Cleans and processes text data
│   ├── feature_extraction.py   # TF-IDF, BERT embeddings, and topic modeling
│   ├── train_model.py          # Trains Ridge Regression model
│   ├── validate.py             # Evaluates model against DW-NOMINATE
│   ├── visualize.py            # Generates ideology vs. party comparison plots
│
│── requirements.txt            # Dependencies for the project
│── README.md                   # Project documentation
│── main.py                     # Entry point for running the entire pipeline
```

## 🚀 How to Run the Pipeline

### **1️⃣ Install Dependencies**
```bash
pip install -r requirements.txt
```
2️⃣ Scrape Politicians' Websites
```bash
cd politician_scraper
scrapy crawl politicians -o ../data/politicians.json
```
3️⃣ Preprocess and Extract Features
```bash
python src/preprocess.py
python src/feature_extraction.py
```
4️⃣ Train the Model
```bash
python src/train_model.py
```
5️⃣ Validate and Visualize Results
```bash
python src/validate.py
python src/visualize.py
```