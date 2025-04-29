import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "presidential_ideology_predictions.csv")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def main():
    df = pd.read_csv(DATA_PATH)
    # Drop missing values for required columns
    df = df.dropna(subset=["bert_prediction", "Party", "word_count", "Name"])

    # Set up color palette for parties
    party_palette = {
        "Democratic": "#3498db",
        "Republican": "#e74c3c",
        "Green": "#2ecc71",
        "Libertarian": "#f39c12",
        # Add more if needed
    }
    # Assign colors, fallback to gray if not found
    df["color"] = df["Party"].map(party_palette).fillna("#888888")

    # --- Scatter plot ---
    plt.figure(figsize=(12, 7))
    plt.scatter(
        df["bert_prediction"],
        df["word_count"],
        c=df["color"],
        s=120,
        edgecolor='k',
        alpha=0.85
    )
    for _, row in df.iterrows():
        plt.text(
            row["bert_prediction"], row["word_count"] + 50,
            row["Name"], fontsize=9, ha='center', va='bottom'
        )
    handles = []
    for party, color in party_palette.items():
        handles.append(plt.Line2D([0], [0], marker='o', color='w', label=party,
                                  markerfacecolor=color, markersize=10, markeredgecolor='k'))
    plt.legend(handles=handles, title="Party", loc="best")
    plt.xlabel("BERT Ideology Prediction (-1 = Liberal, 1 = Conservative)")
    plt.ylabel("Website Word Count")
    plt.title("BERT Ideology Predictions by Candidate\n(Colored by Party, Word Count Shown)")
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "bert_scatter_plot.png"), dpi=300)
    #plt.show()

    # --- Horizontal bar plot ---
    # Prepare candidate labels with word count
    df_sorted = df.sort_values("bert_prediction")
    df_sorted["label"] = df_sorted.apply(
        lambda r: f"{r['Name']} ({int(r['word_count'])})", axis=1
    )
    plt.figure(figsize=(13, max(7, 0.45 * len(df_sorted))))
    bars = plt.barh(
        df_sorted["label"],
        df_sorted["bert_prediction"],
        color=df_sorted["color"],
        edgecolor='k',
        alpha=0.85
    )
    plt.axvline(0, color='black', linestyle='-', alpha=0.5)
    plt.xlabel("BERT Ideology Prediction (-1 = Liberal, 1 = Conservative)")
    plt.title("BERT Ideology Predictions by Candidate\n(Bar: Word Count in Parentheses)")
    plt.grid(axis='x', linestyle='--', alpha=0.3)
    # Add value labels at end of bars
    for bar, val in zip(bars, df_sorted["bert_prediction"]):
        xpos = bar.get_width() + (0.02 if val >= 0 else -0.02)
        ha = 'left' if val >= 0 else 'right'
        plt.text(xpos, bar.get_y() + bar.get_height()/2, f"{val:.2f}", va='center', ha=ha, fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "bert_bar_plot.png"), dpi=300)
    #plt.show()

if __name__ == "__main__":
    main()
