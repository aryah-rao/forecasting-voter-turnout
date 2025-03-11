import pandas as pd
import xml.etree.ElementTree as ET

# Function to preprocess bioname and lowercase firstname, second name, and suffix
def preprocess_name(bioname):
    # Split by comma to separate LASTNAME from firstname secondname suffix
    parts = bioname.split(",")
    lastname = parts[0].strip().lower()  # Last name part, convert to lowercase
    first_middle = parts[1].strip()  # First name, middle name, and suffix part
    name_parts = first_middle.split()
    
    # Extract first name (always the first element)
    firstname = name_parts[0].lower() 
    
    name = f"{firstname} {lastname}".strip()

    # Return the name in the format: firstname secondname suffix (if applicable)
    return name

# Step 1: Read the CSV file to get the names
csv_file = "../data/DW-NOMINATE.csv"  # Update the path to your CSV file
df = pd.read_csv(csv_file)
df['bioname_preprocess'] = df['bioname'].apply(lambda x: preprocess_name(x.strip()))  # Clean and lowercase the bioname column

# Step 2: Parse the XML file to extract the URLs
xml_file = "../data/congress_members.xml"  # Update the path to your XML file
tree = ET.parse(xml_file)
root = tree.getroot()

# Step 3: Create a dictionary for matching bionames to URLs
namespace = {'ns': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
name_to_url = {}

for url_elem in root.findall('ns:url', namespace):
    loc = url_elem.find('ns:loc', namespace).text
    # Extract name from the URL (last part of the URL path)
    url_parts = loc.split("/")
    member_name = url_parts[-2].replace("-", " ").lower()  # Convert to lowercase and format "ROGERS, Mike Dennis"
    name_to_url[member_name] = loc

# Step 4: Create a list to store results
results = []

# Step 5: Find matching names and extract URLs
for bioname, bio in zip(df['bioname'], df["bioname_preprocess"]):
    url = name_to_url.get(bio, 'No matching URL found')
    results.append([bioname, url])

# Step 6: Create a DataFrame from the results and save to CSV
results_df = pd.DataFrame(results, columns=['bioname', 'url'])
results_df.to_csv("../data/matching_bionames_and_urls.csv", index=False)

print("Results saved to 'matching_bionames_and_urls.csv'")