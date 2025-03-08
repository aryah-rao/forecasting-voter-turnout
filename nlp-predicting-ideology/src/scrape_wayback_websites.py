import requests
import pandas as pd
from bs4 import BeautifulSoup
import os
import time
import json
from datetime import datetime
import re

# Configure paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
OUTPUT_DIR = os.path.join(BASE_DIR, 'data', 'wayback_content')

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Function to get Wayback Machine URL for a website for the end of October 2016
def get_wayback_url(url):
    # Target date: End of October 2016 (October 31, 2016)
    target_date = "20161031"
    
    # Format the Wayback Machine URL
    wayback_url = f"https://web.archive.org/web/{target_date}/{url}"
    return wayback_url

# Function to scrape content from a URL
def scrape_website(url, candidate_name):
    try:
        print(f"Scraping {url} for {candidate_name}...")
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=30)
        
        if response.status_code == 200:
            # Parse HTML
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract text content (remove script and style elements)
            for script_or_style in soup(["script", "style"]):
                script_or_style.decompose()
            
            text = soup.get_text(separator=' ', strip=True)
            
            # Clean up text (remove extra whitespace)
            text = re.sub(r'\s+', ' ', text).strip()
            
            # Create result dictionary
            result = {
                'candidate_name': candidate_name,
                'url': url,
                'scraped_date': datetime.now().isoformat(),
                'content': text,
                'html': response.text
            }
            
            return result
        else:
            print(f"Failed to fetch {url}, status code: {response.status_code}")
            return None
    except Exception as e:
        print(f"Error scraping {url}: {str(e)}")
        return None

# Main function
def main():
    # Load the dataset
    candidates_file = os.path.join(DATA_DIR, '2016_presidential_candidates.csv')
    candidates_df = pd.read_csv(candidates_file)
    
    # Iterate through each candidate and scrape their website
    results = []
    
    for index, candidate in candidates_df.iterrows():
        name = candidate['Name']
        website = candidate['Website']
        
        # Get Wayback Machine URL
        wayback_url = get_wayback_url(website)
        
        # Scrape the website
        result = scrape_website(wayback_url, name)
        
        if result:
            # Save individual result to a JSON file
            filename = f"{name.lower().replace(' ', '_')}_wayback_content.json"
            with open(os.path.join(OUTPUT_DIR, filename), 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=4)
            
            # Add to results list (without HTML content to save memory)
            result_without_html = result.copy()
            result_without_html.pop('html', None)
            results.append(result_without_html)
            
            # Be nice to the Wayback Machine and avoid rate limiting
            time.sleep(3)
    
    # Create a summary DataFrame and save it
    if results:
        summary_df = pd.DataFrame(results)
        summary_df.to_csv(os.path.join(OUTPUT_DIR, 'wayback_scraping_summary.csv'), index=False)
        print(f"Successfully scraped {len(results)} websites. Data saved to {OUTPUT_DIR}")
    else:
        print("No websites were successfully scraped.")

if __name__ == "__main__":
    main()
