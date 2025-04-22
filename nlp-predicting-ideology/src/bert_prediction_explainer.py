import os
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
PREDICTIONS_PATH = os.path.join(DATA_DIR, "presidential_ideology_predictions.csv")
OUTPUT_DIR = os.path.join(DATA_DIR, "bert_prediction_explainer")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_data():
    df = pd.read_csv(PREDICTIONS_PATH)
    # Filter out missing text
    df = df.dropna(subset=["clean_text", "bert_prediction"])
    return df

def split_by_bert(df, threshold=0.0):
    left_df = df[df["bert_prediction"] < threshold]
    right_df = df[df["bert_prediction"] >= threshold]
    return left_df, right_df

def generate_wordcloud(texts, title, filename):
    text = " ".join(texts)
    stopwords = set(STOPWORDS)
    wc = WordCloud(width=800, height=400, background_color="white", stopwords=stopwords, max_words=100, colormap="coolwarm")
    wc.generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=200)
    plt.close()

def get_top_terms(texts, n=20):
    from collections import Counter
    import re
    words = []
    for t in texts:
        tokens = re.findall(r'\b[a-zA-Z]{3,}\b', t.lower())
        words.extend([w for w in tokens if w not in STOPWORDS])
    return Counter(words).most_common(n)

def main():
    df = load_data()
    left_df, right_df = split_by_bert(df)
    print(f"Left-predicted: {len(left_df)}, Right-predicted: {len(right_df)}")

    # Word clouds
    generate_wordcloud(left_df["clean_text"], "Word Cloud: BERT-Predicted Left", "bert_left_wordcloud.png")
    generate_wordcloud(right_df["clean_text"], "Word Cloud: BERT-Predicted Right", "bert_right_wordcloud.png")

    # Top terms
    print("\nTop terms for BERT-predicted LEFT:")
    for word, count in get_top_terms(left_df["clean_text"]):
        print(f"{word}: {count}")
    print("\nTop terms for BERT-predicted RIGHT:")
    for word, count in get_top_terms(right_df["clean_text"]):
        print(f"{word}: {count}")

    print(f"\nWord clouds saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
