import pandas as pd
import numpy as np
import os
import embedding_utils

# Load dataset
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
dataset_path = os.path.join(BASE_DIR, "data", "final_dataset.csv")
df = embedding_utils.load_data()

# Compute word lengths
df["word_count"] = df["clean_text"].apply(lambda text: len(str(text).split()))

# Calculate statistics
mean_length = df["word_count"].mean()
std_dev = df["word_count"].std()
percentiles = np.percentile(df["word_count"], [25, 50, 75, 90, 95, 99])

# Print results
print(f"Mean Word Length: {mean_length:.2f}")
print(f"Standard Deviation: {std_dev:.2f}")
print(f"Percentiles:")
print(f"  25th Percentile: {percentiles[0]}")
print(f"  50th Percentile (Median): {percentiles[1]}")
print(f"  75th Percentile: {percentiles[2]}")
print(f"  90th Percentile: {percentiles[3]}")
print(f"  95th Percentile: {percentiles[4]}")
print(f"  99th Percentile: {percentiles[5]}")

import seaborn as sns
import matplotlib.pyplot as plt

# Plot histogram of word lengths
plt.figure(figsize=(12, 6))
sns.histplot(df["word_count"][df["word_count"] < 5000], bins=50, kde=True)
plt.xlabel("Word Length")
plt.ylabel("Frequency")
plt.title("Histogram of Word Lengths")
plt.show()