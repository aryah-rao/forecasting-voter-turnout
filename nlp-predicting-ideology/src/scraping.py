"""
Scrape politician websites using URLs from the mapping file.

This script:
1. Loads the website mapping data
2. Extracts URLs for scraping
3. Initiates the Scrapy spider to fetch website content
4. Saves the scraped data for further processing
"""
import os
import subprocess
import sys
import logging
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join("..", "logs", "scraping.log"), mode='a')
    ]
)
logger = logging.getLogger("scraping")

# Create logs directory if it doesn't exist
os.makedirs(os.path.join("..", "logs"), exist_ok=True)

# Get absolute path of the project root
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# File paths
DW_NOMINATE_PATH = os.path.join(BASE_DIR, "data", "DW-NOMINATE.csv")
WEBSITE_MAPPING_PATH = os.path.join(BASE_DIR, "data", "matching_bionames_and_urls_with_websites.csv")
SCRAPER_PATH = os.path.join(BASE_DIR, "politician_scraper")
SCRAPED_DATA_OUTPUT = os.path.join(BASE_DIR, "data", "politicians.json")
MERGED_DATA_OUTPUT = os.path.join(BASE_DIR, "data", "merged_politicians.csv")

def load_website_mapping() -> pd.DataFrame:
    """
    Load the website mapping data and extract URLs for scraping.
    
    Returns:
        DataFrame containing politician names and websites
    """
    logger.info(f"Loading website mapping from {WEBSITE_MAPPING_PATH}")
    
    try:
        # Check if file exists
        if not os.path.exists(WEBSITE_MAPPING_PATH):
            raise FileNotFoundError(f"Website mapping file not found: {WEBSITE_MAPPING_PATH}")
        
        # Load website mapping data
        website_mapping = pd.read_csv(WEBSITE_MAPPING_PATH)
        
        # Ensure required columns exist
        if 'bioname' not in website_mapping.columns or 'website' not in website_mapping.columns:
            raise ValueError("Website mapping file must contain 'bioname' and 'website' columns")
            
        # Filter out records with missing websites
        initial_count = len(website_mapping)
        website_mapping = website_mapping.dropna(subset=['website'])
        if initial_count - len(website_mapping) > 0:
            logger.warning(f"Dropped {initial_count - len(website_mapping)} records with missing websites")
            
        logger.info(f"Loaded {len(website_mapping)} politician websites")
        return website_mapping
        
    except Exception as e:
        logger.error(f"Error loading website mapping: {e}")
        raise

def prepare_url_list(website_mapping: pd.DataFrame) -> List[str]:
    """
    Extract and validate URLs from the website mapping.
    
    Args:
        website_mapping: DataFrame with politician names and websites
        
    Returns:
        List of valid URLs for scraping
    """
    # Extract unique URLs
    urls = website_mapping["website"].dropna().unique().tolist()
    
    # Validate URLs (basic check)
    valid_urls = []
    for url in urls:
        if not isinstance(url, str):
            logger.warning(f"Skipping non-string URL: {url}")
            continue
            
        url = url.strip()
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
            
        valid_urls.append(url)
    
    logger.info(f"Prepared {len(valid_urls)} URLs for scraping")
    return valid_urls

def run_spider(urls: List[str]):
    """
    Run the Scrapy spider to scrape the politician websites.
    
    Args:
        urls: List of URLs to scrape
    """
    logger.info(f"Starting web scraping for {len(urls)} URLs...")
    
    # Create a temporary JSON file with URLs for the spider
    temp_urls_path = os.path.join(BASE_DIR, "data", "urls_to_scrape.json")
    with open(temp_urls_path, 'w') as f:
        json.dump(urls, f)
    
    # Find Python executable
    python_executable = sys.executable
    
    # Prepare the scraper directory
    if not os.path.exists(SCRAPER_PATH):
        logger.error(f"Scraper directory not found: {SCRAPER_PATH}")
        raise FileNotFoundError(f"Scraper directory not found: {SCRAPER_PATH}")
    
    try:
        # Create data directory if it doesn't exist
        os.makedirs(os.path.join(BASE_DIR, "data"), exist_ok=True)
        
        # Run the spider
        logger.info(f"Running scrapy spider from {SCRAPER_PATH}")
        subprocess.run(
            [python_executable, "-m", "scrapy", "crawl", "politicians", 
             "-o", SCRAPED_DATA_OUTPUT, 
             "-a", f"url_file={temp_urls_path}"],
            cwd=SCRAPER_PATH,
            check=True
        )
        logger.info(f"Scraping completed. Data saved to {SCRAPED_DATA_OUTPUT}")
        
        # Clean up temp file
        if os.path.exists(temp_urls_path):
            os.remove(temp_urls_path)
            
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running scrapy spider: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during scraping: {e}")
        raise

def merge_data():
    """
    Merge the DW-NOMINATE data with the website mapping data for reference.
    """
    try:
        # Load DW-NOMINATE data
        if not os.path.exists(DW_NOMINATE_PATH):
            logger.warning(f"DW-NOMINATE file not found: {DW_NOMINATE_PATH}")
            return
            
        dw_nominate = pd.read_csv(DW_NOMINATE_PATH)
        dw_nominate = dw_nominate[['bioname', 'nominate_dim1']]
        
        # Load website mapping
        website_mapping = pd.read_csv(WEBSITE_MAPPING_PATH)
        website_mapping = website_mapping[['bioname', 'website']]
        
        # Merge data
        merged_data = pd.merge(dw_nominate, website_mapping, on='bioname', how='inner')
        logger.info(f"Total politicians matched: {len(merged_data)}")
        
        # Save merged data for reference
        merged_data.to_csv(MERGED_DATA_OUTPUT, index=False)
        
    except Exception as e:
        logger.error(f"Error merging data: {e}")
        raise

if __name__ == "__main__":
    try:
        website_mapping = load_website_mapping()
        urls = prepare_url_list(website_mapping)
        run_spider(urls)
        merge_data()
    except Exception as e:
        logger.error(f"Script failed: {e}")
        sys.exit(1)
