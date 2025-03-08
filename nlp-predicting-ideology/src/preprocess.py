"""
Preprocess raw politician text data from web scraping into a clean dataset for NLP analysis.

This script:
1. Loads scraped politician website text
2. Cleans and preprocesses text data
3. Merges with DW-NOMINATE ideology scores
4. Saves the final dataset for model training
"""
import os
import json
import pandas as pd
import numpy as np
import re
import nltk
import logging
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from bs4 import BeautifulSoup
from typing import List, Dict, Any
from pathlib import Path

# Define the base directory based on script location
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LOGS_DIR = os.path.join(BASE_DIR, "logs")

# Create logs directory if it doesn't exist
os.makedirs(LOGS_DIR, exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(LOGS_DIR, "preprocessing.log"), mode='a')
    ]
)
logger = logging.getLogger("preprocess")

# Ensure all necessary NLTK resources are downloaded
try:
    nltk.download("stopwords", quiet=True)
    nltk.download("punkt", quiet=True)
    nltk.download("wordnet", quiet=True)
    logger.info("NLTK resources downloaded successfully")
except Exception as e:
    logger.warning(f"Error downloading NLTK resources: {e}")

# Define file paths
DATA_DIR = os.path.join(BASE_DIR, "data")
NOMINATE_PATH = os.path.join(DATA_DIR, "DW-NOMINATE.csv")
WEBSITE_MAPPING_PATH = os.path.join(DATA_DIR, "matching_bionames_and_urls_with_websites.csv")
SCRAPED_DATA_PATH = os.path.join(DATA_DIR, "politicians.json")
OUTPUT_PATH = os.path.join(DATA_DIR, "final_dataset.csv")

def load_data() -> tuple:
    """
    Load required data files: 
    1. DW-NOMINATE ideology scores
    2. Politician-website mapping
    3. Scraped website text
    
    Returns:
        Tuple of (nominate_df, website_mapping_df, scraped_data)
    """
    logger.info("Loading data files...")
    
    try:
        # Load DW-NOMINATE data
        nominate_df = pd.read_csv(NOMINATE_PATH)
        logger.info(f"Loaded {len(nominate_df)} records from DW-NOMINATE data")
        
        # Load website mapping data
        website_mapping_df = pd.read_csv(WEBSITE_MAPPING_PATH)
        logger.info(f"Loaded {len(website_mapping_df)} records from website mapping data")
        
        # Load scraped website data
        with open(SCRAPED_DATA_PATH, 'r', encoding='utf-8') as f:
            scraped_data = json.load(f)
        logger.info(f"Loaded {len(scraped_data)} records from scraped website data")
        
        return nominate_df, website_mapping_df, scraped_data
        
    except FileNotFoundError as e:
        logger.error(f"Data file not found: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def clean_text(text: str) -> str:
    """
    Clean and preprocess text data:
    1. Convert to lowercase
    2. Remove HTML tags if any
    3. Remove special characters and digits
    4. Remove extra whitespace
    5. Remove stopwords
    6. Lemmatize words
    
    Args:
        text: Raw text string
        
    Returns:
        Cleaned text string
    """
    if not isinstance(text, str):
        return ""
        
    try:
        # Convert to lowercase
        text = text.lower()
        
        # Remove HTML tags
        soup = BeautifulSoup(text, "html.parser")
        text = soup.get_text()
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stop_words]
        
        # Lemmatize
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
        
        # Remove extra whitespace
        cleaned_text = " ".join(tokens)
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
        
        return cleaned_text
        
    except Exception as e:
        logger.error(f"Error cleaning text: {e}")
        return ""

def create_dataset():
    """
    Create the final dataset by:
    1. Loading and preprocessing the data
    2. Merging DW-NOMINATE scores with website content
    3. Cleaning the text data
    4. Saving the resulting dataset
    """
    try:
        # Load data
        nominate_df, website_mapping_df, scraped_data = load_data()
        
        # Keep only necessary columns from DW-NOMINATE
        if 'bioname' in nominate_df.columns:
            nominate_df = nominate_df[['bioname', 'nominate_dim1']]
            
        # Convert scraped data to DataFrame
        scraped_df = pd.DataFrame(scraped_data)
        logger.info(f"Scraped data shape: {scraped_df.shape}")
        
        # Merge website mapping with scraped content
        merged_df = pd.merge(
            website_mapping_df, 
            scraped_df, 
            left_on='website', 
            right_on='url', 
            how='inner'
        )
        logger.info(f"Merged website mapping with scraped content: {len(merged_df)} records")
        
        # Merge with DW-NOMINATE data
        final_df = pd.merge(
            merged_df,
            nominate_df,
            on='bioname',
            how='inner'
        )
        logger.info(f"Merged with DW-NOMINATE data: {len(final_df)} records")
        
        # Clean text
        logger.info("Cleaning text data...")
        final_df['clean_text'] = final_df['text'].apply(clean_text)
        
        # Remove rows with empty text after cleaning
        initial_count = len(final_df)
        final_df = final_df.dropna(subset=['clean_text'])
        final_df = final_df[final_df['clean_text'].str.strip() != '']
        if initial_count - len(final_df) > 0:
            logger.warning(f"Removed {initial_count - len(final_df)} rows with empty text")
        
        # Select final columns
        final_df = final_df[['bioname', 'nominate_dim1', 'website', 'clean_text']]
        
        # Save final dataset
        final_df.to_csv(OUTPUT_PATH, index=False)
        logger.info(f"Final dataset saved with {len(final_df)} records to {OUTPUT_PATH}")
        
        # Print dataset statistics
        print("\nDataset Statistics:")
        print(f"Total politicians: {len(final_df)}")
        print(f"Average text length: {final_df['clean_text'].str.len().mean():.1f} characters")
        print(f"Average word count: {final_df['clean_text'].str.split().str.len().mean():.1f} words")
        
        return final_df
        
    except Exception as e:
        logger.error(f"Error creating dataset: {e}")
        raise

if __name__ == "__main__":
    logger.info("Starting data preprocessing...")
    create_dataset()
    logger.info("Preprocessing complete!")
