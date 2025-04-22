import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.webdriver import WebDriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

# Function to get website URL for a given Congress member URL
def get_website_url(member_url: str):
    # Set up Chrome options to run headless (no UI)
    chrome_options = Options()
    # chrome_options.add_argument("--headless")  # Run in headless mode (no browser window)

    # Initialize the WebDriver with the Service class
    service = Service(ChromeDriverManager().install())  # Automatically installs the right ChromeDriver version
    driver = webdriver.Chrome(service=service, options=chrome_options)

    try:
        # Open the member's page
        driver.get(member_url)

        # Wait for the <a> element containing the website link to be visible
        website_link = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.XPATH, "/html[1]/body[1]/div[2]/div[1]/main[1]/div[1]/div[3]/div[1]/div[1]/div[2]/table[1]/tbody[1]/tr[1]/td[1]/a[1]"))
        )

        # Extract the URL from the <a> element
        website_url = website_link.get_attribute('href')
        return website_url

    except Exception as e:
        print(f"Error fetching website for {member_url}: {e}")
        return None

    finally:
        # Close the driver
        driver.quit()

# Read the CSV file containing names and URLs
csv_file = "../data/matching_bionames_and_urls.csv"  # Update path if necessary
df = pd.read_csv(csv_file)

# Create an empty list to store website URLs for each member
website_urls = []

# Loop through each row in the dataframe to fetch the website URL
for index, row in df.iterrows():
    bioname = row['bioname']
    member_url = row['url']
    
    # Get the website URL for the member page
    website_url = get_website_url(member_url)
    
    # Add the website URL to the list
    website_urls.append(website_url)

# Add the website URLs to the dataframe
df['website'] = website_urls

# Save the updated dataframe with websites to a new CSV file
df.to_csv("../data/matching_bionames_and_urls_with_websites.csv", index=False)

print("Website URLs saved to 'matching_bionames_and_urls_with_websites.csv'")
