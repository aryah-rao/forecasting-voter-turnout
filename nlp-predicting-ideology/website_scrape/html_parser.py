import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin
import time
import hashlib

def normalize_url(url):
    """Removes fragments and query parameters for consistency."""
    parsed_url = urlparse(url)
    return f"{parsed_url.scheme}://{parsed_url.netloc}{parsed_url.path.rstrip('/')}"

def get_page_hash(html):
    """Returns a hash of the page content to detect duplicates."""
    return hashlib.md5(html.encode("utf-8")).hexdigest()

def scrape_page(driver):
    """Extracts and parses the full HTML and text content of the current page."""
    html = driver.page_source
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text(separator=' ', strip=False)
    return str(soup), text, get_page_hash(str(soup))

def visit_and_scrape(driver, visited_urls, visited_hashes, base_url, base_domain, depth=0, max_depth=3):
    """Recursively scrapes HTML and text from the website, staying within the original domain."""
    if depth > max_depth:
        return [], []
    
    html_data, text_data = [], []
    page_html, page_text, page_hash = scrape_page(driver)
    
    # Avoid duplicate content based on hash
    if page_hash not in visited_hashes:
        visited_hashes.add(page_hash)
        html_data.append(page_html)
        text_data.append(page_text)

    # Wait for the <a> tags to be loaded
    try:
        links = WebDriverWait(driver, 10).until(
            EC.presence_of_all_elements_located((By.TAG_NAME, 'a'))
        )
    except Exception as e:
        print(f"Error waiting for links: {e}")
        return html_data, text_data
    
    for link in links:
        try:
            href = link.get_attribute("href")
            if href and "javascript:" not in href and "mailto:" not in href:
                normalized_href = normalize_url(urljoin(base_url, href))  # Normalize URL
                
                # Ensure the URL is within the original domain
                if base_domain in urlparse(normalized_href).netloc and normalized_href not in visited_urls:
                    visited_urls.add(normalized_href)
                    driver.execute_script("window.open(arguments[0]);", normalized_href)
                    driver.switch_to.window(driver.window_handles[-1])
                    time.sleep(2)  # Wait for dynamic content to load
                    
                    new_html, new_text = visit_and_scrape(driver, visited_urls, visited_hashes, base_url, base_domain, depth + 1, max_depth)
                    html_data.extend(new_html)
                    text_data.extend(new_text)
                    
                    driver.close()
                    driver.switch_to.window(driver.window_handles[0])
        except Exception as e:
            print(f"Skipping link due to error: {e}")
    
    return html_data, text_data

def save_scraped_content_in_csv(df, index, text_content):
    """Add the scraped content as a new column to the DataFrame."""
    df.at[index, 'scraped_content'] = text_content

def main(csv_file):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_file)
    
    # Initialize a headless browser to avoid opening a GUI window
    options = webdriver.ChromeOptions()
    options.headless = True
    driver = webdriver.Chrome(options=options)

    # Loop through the DataFrame and scrape each politician's website
    for index, row in df.iterrows():
        name = row['bioname']
        website = row['website']
        
        if pd.notnull(website):  # Check if there's a valid website URL
            print(f"Scraping website for {name}: {website}")
            try:
                # Visit and scrape the website
                driver.get(website)
                time.sleep(2)  # Allow initial page to load
                
                base_url = f"{urlparse(website).scheme}://{urlparse(website).netloc}"
                base_domain = urlparse(website).netloc  # Extract base domain (e.g., "sewell.house.gov")
                visited_urls = set([normalize_url(website)])
                visited_hashes = set()  # Store page content hashes to detect duplicates
                all_html, all_text = visit_and_scrape(driver, visited_urls, visited_hashes, base_url, base_domain)

                # Join all scraped text into one large string and add it to the DataFrame
                full_text_content = " ".join(all_text)
                save_scraped_content_in_csv(df, index, full_text_content)

                print(f"Successfully scraped content for {name}")
            except Exception as e:
                print(f"Error scraping {name}: {e}")
        else:
            print(f"No website for {name}")

    # Save the updated DataFrame to a new CSV file
    df.to_csv("updated_merged_politicians.csv", index=False)
    
    # Close the browser after scraping
    driver.quit()

if __name__ == "__main__":
    csv_file = "../data/merged_politicians.csv"  # Path to your CSV file
    main(csv_file)
