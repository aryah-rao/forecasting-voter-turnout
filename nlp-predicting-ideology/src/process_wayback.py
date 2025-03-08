"""
Process wayback machine archived content for presidential candidates.

This script:
1. Loads wayback machine archived website content (JSON files)
2. Cleans and preprocesses the text data
3. Merges with 2016 presidential candidates data
4. Saves the final dataset for analysis
"""
import os
import json
import glob
import pandas as pd
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import re
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
import sys

# Add the parent directory to the path if needed
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

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
        logging.FileHandler(os.path.join(LOGS_DIR, "wayback_processing.log"), mode='a')
    ]
)
logger = logging.getLogger("wayback_processor")

# Import clean_text from preprocess.py
try:
    # First try to import directly (if running from src directory)
    from preprocess import clean_text
    logger.info("Successfully imported clean_text function")
except ImportError:
    # If that fails, try with full module path
    try:
        from src.preprocess import clean_text
        logger.info("Successfully imported clean_text function using src prefix")
    except ImportError:
        logger.error("Could not import clean_text function - defining it directly")
        
        # Define the clean_text function directly if import fails
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

# Ensure all necessary NLTK resources are downloaded
try:
    nltk.download("stopwords", quiet=True)
    nltk.download("punkt", quiet=True)
    nltk.download("wordnet", quiet=True)
    logger.info("NLTK resources downloaded successfully")
except Exception as e:
    logger.warning(f"Error downloading NLTK resources: {e}")

# Define file paths
WAYBACK_DIR = os.path.join(BASE_DIR, "data", "wayback_content")
CANDIDATES_PATH = os.path.join(BASE_DIR, "data", "2016_presidential_candidates.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "data", "presidential_candidates_with_text.csv")

def load_candidates_data() -> pd.DataFrame:
    """
    Load presidential candidates data from CSV.
    
    Returns:
        DataFrame with candidate data
    """
    logger.info(f"Loading candidates data from {CANDIDATES_PATH}")
    
    try:
        if not os.path.exists(CANDIDATES_PATH):
            raise FileNotFoundError(f"Candidates file not found: {CANDIDATES_PATH}")
            
        candidates_df = pd.read_csv(CANDIDATES_PATH)
        logger.info(f"Loaded {len(candidates_df)} candidate records")
        return candidates_df
        
    except Exception as e:
        logger.error(f"Error loading candidates data: {e}")
        raise

def get_wayback_files() -> List[str]:
    """
    Get a list of all JSON files in the wayback_content directory.
    
    Returns:
        List of file paths to JSON files
    """
    if not os.path.exists(WAYBACK_DIR):
        logger.error(f"Wayback content directory not found: {WAYBACK_DIR}")
        raise FileNotFoundError(f"Directory not found: {WAYBACK_DIR}")
        
    json_files = glob.glob(os.path.join(WAYBACK_DIR, "*.json"))
    logger.info(f"Found {len(json_files)} JSON files in wayback content directory")
    return json_files

def extract_candidate_info_from_filename(filename: str) -> Dict[str, Any]:
    """
    Extract candidate information from the wayback JSON filename.
    
    Args:
        filename: Name of the JSON file
    
    Returns:
        Dictionary with candidate name and other extracted info
    """
    # Extract the candidate name and other info from filename
    # Assuming filename format like: "firstname_lastname_wayback_content.json"
    base_name = os.path.basename(filename)
    
    # Parse the filename to extract the candidate's name
    name_parts = base_name.split('_wayback_content.json')[0].split('_')
    
    # Convert to title case and join with space
    if len(name_parts) >= 1:
        # Convert snake_case to Title Case
        candidate_name = ' '.join(part.capitalize() for part in name_parts)
        return {"Name": candidate_name}  # Use "Name" to match the CSV column
    
    logger.warning(f"Could not extract candidate info from filename: {filename}")
    return {"Name": "unknown"}

def process_wayback_file(filepath: str) -> Dict[str, Any]:
    """
    Process a single wayback JSON file to extract and clean text.
    
    Args:
        filepath: Path to the JSON file
    
    Returns:
        Dictionary with candidate info and cleaned text
    """
    try:
        # Extract candidate info from filename
        candidate_info = extract_candidate_info_from_filename(filepath)
        
        # Load JSON data
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Extract text content from JSON
        raw_text = ""
        if isinstance(data, dict):
            if "text" in data:
                raw_text = data["text"]
            elif "content" in data:
                raw_text = data["content"]
        
        # Clean the text using the function from preprocess.py
        cleaned_text = clean_text(raw_text)
        
        result = {
            **candidate_info,
            "raw_text_length": len(raw_text),
            "clean_text": cleaned_text,
            "clean_text_length": len(cleaned_text),
            "word_count": len(cleaned_text.split())
        }
        
        return result
        
    except Exception as e:
        logger.error(f"Error processing wayback file {filepath}: {e}")
        return {
            "Name": os.path.basename(filepath),
            "clean_text": "",
            "error": str(e)
        }

def process_all_wayback_files() -> pd.DataFrame:
    """
    Process all wayback JSON files and create a DataFrame.
    
    Returns:
        DataFrame with candidate names and cleaned text
    """
    wayback_files = get_wayback_files()
    processed_data = []
    
    for i, file_path in enumerate(wayback_files):
        logger.info(f"Processing file {i+1}/{len(wayback_files)}: {os.path.basename(file_path)}")
        result = process_wayback_file(file_path)
        processed_data.append(result)
    
    # Convert to DataFrame
    wayback_df = pd.DataFrame(processed_data)
    logger.info(f"Processed {len(wayback_df)} wayback files")
    
    # Remove entries with empty text
    initial_count = len(wayback_df)
    wayback_df = wayback_df.dropna(subset=["clean_text"])
    wayback_df = wayback_df[wayback_df["clean_text"].str.strip() != ""]
    if initial_count - len(wayback_df) > 0:
        logger.warning(f"Removed {initial_count - len(wayback_df)} entries with empty text")
    
    return wayback_df

def merge_with_candidates(wayback_df: pd.DataFrame, candidates_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge wayback text data with presidential candidates data.
    
    Args:
        wayback_df: DataFrame with processed wayback content
        candidates_df: DataFrame with presidential candidates data
    
    Returns:
        Merged DataFrame
    """
    logger.info("Merging wayback text with candidates data")
    
    # Log column names for debugging
    logger.info(f"Candidates DataFrame columns: {candidates_df.columns.tolist()}")
    logger.info(f"Wayback DataFrame columns: {wayback_df.columns.tolist()}")
    
    # Merge on the "Name" column which should be present in both DataFrames
    merged_df = pd.merge(
        candidates_df,
        wayback_df,
        on="Name",  # Use "Name" column for merging
        how="left"
    )
    
    # Log merge statistics
    total_candidates = len(candidates_df)
    matched_candidates = merged_df["clean_text"].notna().sum()
    logger.info(f"Matched {matched_candidates} out of {total_candidates} candidates with website text")
    
    return merged_df

def main():
    """
    Main function to run the wayback content processing pipeline.
    """
    try:
        # Load presidential candidates data
        candidates_df = load_candidates_data()
        
        # Process all wayback JSON files
        wayback_df = process_all_wayback_files()
        
        # Merge the data
        merged_df = merge_with_candidates(wayback_df, candidates_df)
        
        # Save the final dataset
        merged_df.to_csv(OUTPUT_PATH, index=False)
        logger.info(f"Final dataset saved to {OUTPUT_PATH}")
        
        # Print some statistics
        print("\nDataset Statistics:")
        print(f"Total candidates: {len(candidates_df)}")
        print(f"Candidates with website text: {merged_df['clean_text'].notna().sum()}")
        print(f"Average word count: {merged_df['word_count'].mean():.1f} words")
        
    except Exception as e:
        logger.error(f"Error in main process: {e}")
        raise

if __name__ == "__main__":
    logger.info("Starting wayback content processing...")
    main()
    logger.info("Processing complete!")
