import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "final_dataset.csv")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "wordclouds")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Define DW-NOMINATE bins and labels
bins = [-1, -0.5, 0, 0.5, 1]
labels = ["Far Left", "Left", "Right", "Far Right"]

def main():
    df = pd.read_csv(DATA_PATH)
    # Drop missing or empty text
    df = df.dropna(subset=["clean_text", "nominate_dim1"])
    df = df[df["clean_text"].str.strip() != ""]

    # Compute word counts
    df["word_count"] = df["clean_text"].apply(lambda t: len(str(t).split()))

    # Bin ideology scores
    df["ideology_bin"] = pd.cut(df["nominate_dim1"], bins=bins, labels=labels, include_lowest=True, right=False)

    # Store word count stats for each bin
    wc_stats = []
    for label in labels:
        bin_df = df[df["ideology_bin"] == label]
        texts = bin_df["clean_text"].tolist()
        all_text = " ".join(texts)
        wordcloud = WordCloud(
            width=900, height=400, background_color="white",
            stopwords=STOPWORDS, max_words=100, colormap="viridis"
        ).generate(all_text)
        plt.figure(figsize=(12, 5))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.title(f"Word Cloud: {label} ({len(bin_df)} samples)")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"wordcloud_{label.replace(' ', '_').lower()}.png"), dpi=200)
        plt.close()

        wc = bin_df["word_count"]
        wc_stats.append({
            "label": label,
            "mean": wc.mean(),
            "median": wc.median(),
            "q25": np.percentile(wc, 25),
            "q75": np.percentile(wc, 75),
            "count": len(bin_df)
        })
        print(f"\n{label} ({len(bin_df)} samples):")
        print(f"  Mean word count: {wc.mean():.1f}")
        print(f"  Median word count: {wc.median():.1f}")
        print(f"  25th percentile: {np.percentile(wc, 25):.1f}")
        print(f"  75th percentile: {np.percentile(wc, 75):.1f}")

    # Also print overall stats
    print("\nOverall:")
    wc = df["word_count"]
    print(f"  Mean word count: {wc.mean():.1f}")
    print(f"  Median word count: {wc.median():.1f}")
    print(f"  25th percentile: {np.percentile(wc, 25):.1f}")
    print(f"  75th percentile: {np.percentile(wc, 75):.1f}")

    # Plot and save word count stats
    stats_df = pd.DataFrame(wc_stats)
    plt.figure(figsize=(9, 6))
    x = stats_df["label"]
    plt.bar(x, stats_df["mean"], width=0.4, label="Mean", color="#3498db", alpha=0.7)
    plt.bar(x, stats_df["median"], width=0.2, label="Median", color="#e67e22", alpha=0.7)
    plt.errorbar(x, stats_df["mean"], 
                 yerr=[stats_df["mean"] - stats_df["q25"], stats_df["q75"] - stats_df["mean"]],
                 fmt='o', color='k', capsize=6, label="IQR (25th-75th)")
    plt.ylabel("Word Count")
    plt.title("Word Count Statistics by Ideology Bin")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "wordcount_stats_by_bin.png"), dpi=200)
    plt.close()

if __name__ == "__main__":
    main()
