import scrapy
import pandas as pd
import os
import logging

# Get absolute path of the project root
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

csv_path = os.path.join(BASE_DIR, "data", "merged_politicians.csv")

# Read politician websites
df = pd.read_csv(csv_path)
urls = df["website"].dropna().unique().tolist()

print(f"Loaded {len(urls)} politician URLs from {csv_path}")

class PoliticianSpider(scrapy.Spider):
    name = "politicians"
    allowed_domains = [url.replace("https://", "").replace("http://", "").split("/")[0] for url in urls]
    start_urls = urls

    def parse(self, response):
        if response.status != 200:  # Log failed requests
            logging.warning(f"⚠️ Failed to scrape: {response.url} | Status: {response.status}")
            return

        # Extract visible text
        page_text = response.xpath("//body//text()").getall()
        cleaned_text = " ".join(page_text).strip()

        if not page_text or len(page_text) < 50:
            logging.warning(f"⚠️ Possibly JavaScript-rendered page: {response.url}")

        yield {
            "url": response.url,
            "text": cleaned_text
        }