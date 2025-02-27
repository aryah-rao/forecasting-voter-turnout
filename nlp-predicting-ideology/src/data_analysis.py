"""
Comprehensive data analysis tool for political ideology prediction dataset.

This script performs in-depth analysis of:
1. Dataset statistics and distributions
2. Text properties and patterns
3. Ideology score characteristics
4. Visualization of key insights
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter
from wordcloud import WordCloud, STOPWORDS
import nltk
from nltk.util import ngrams
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import embedding_utils

# Download necessary NLTK resources
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

# Configure visualization settings
plt.style.use('seaborn-v0_8-whitegrid')
sns.set(font_scale=1.1)
COLORS = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12", "#9b59b6", "#1abc9c"]

# Create results directory for saving figures
RESULTS_DIR = Path("../results/data_analysis")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def load_clean_data():
    """Load and clean the dataset, returning both original and processed versions."""
    print("Loading dataset...")
    df = embedding_utils.load_data()
    
    # Drop rows with missing values in key columns
    df_clean = df.dropna(subset=["clean_text", "nominate_dim1"]).copy()
    
    # Calculate word counts
    df_clean["word_count"] = df_clean["clean_text"].apply(lambda x: len(str(x).split()))
    
    # Add ideological categories
    bins = [-1, -0.5, -0.2, 0.2, 0.5, 1]
    labels = ["Far Left", "Left", "Center", "Right", "Far Right"]
    df_clean["ideology_category"] = pd.cut(df_clean["nominate_dim1"], bins=bins, labels=labels)
    
    print(f"Dataset loaded: {len(df)} total records, {len(df_clean)} complete records")
    return df, df_clean

def dataset_overview(df, df_clean):
    """Generate high-level overview of the dataset."""
    print("\n==== DATASET OVERVIEW ====")
    
    # Basic statistics
    print(f"Total records: {len(df)}")
    print(f"Complete records: {len(df_clean)}")
    print(f"Missing ideology scores: {df['nominate_dim1'].isna().sum()}")
    print(f"Missing text data: {df['clean_text'].isna().sum()}")
    
    # Ideology score statistics
    ideology_scores = df_clean["nominate_dim1"]
    print("\nIdeology Score Statistics:")
    print(f"  Mean: {ideology_scores.mean():.4f}")
    print(f"  Median: {ideology_scores.median():.4f}")
    print(f"  Min: {ideology_scores.min():.4f}")
    print(f"  Max: {ideology_scores.max():.4f}")
    print(f"  Standard Deviation: {ideology_scores.std():.4f}")
    
    # Text length statistics
    word_counts = df_clean["word_count"]
    print("\nText Length Statistics (word count):")
    print(f"  Mean: {word_counts.mean():.2f}")
    print(f"  Median: {word_counts.median():.2f}")
    print(f"  Min: {word_counts.min()}")
    print(f"  Max: {word_counts.max()}")
    print(f"  Standard Deviation: {word_counts.std():.2f}")
    
    # Ideological distribution
    print("\nIdeological Categories Distribution:")
    ideology_dist = df_clean["ideology_category"].value_counts().sort_index()
    for category, count in ideology_dist.items():
        print(f"  {category}: {count} ({count/len(df_clean)*100:.1f}%)")
    
    return ideology_dist

def visualize_ideology_distribution(df_clean):
    """Create visualizations of ideology score distributions."""
    print("\nGenerating ideology distribution visualizations...")
    
    # Create figure with multiple plots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Histogram of ideology scores
    ax1 = axes[0, 0]
    sns.histplot(df_clean["nominate_dim1"], bins=30, kde=True, ax=ax1, color=COLORS[0])
    ax1.axvline(x=0, color='black', linestyle='-', alpha=0.7)
    ax1.axvline(x=df_clean["nominate_dim1"].mean(), color='red', linestyle='--', 
                label=f'Mean: {df_clean["nominate_dim1"].mean():.2f}')
    ax1.axvline(x=df_clean["nominate_dim1"].median(), color='green', linestyle='--', 
                label=f'Median: {df_clean["nominate_dim1"].median():.2f}')
    ax1.set_title("Distribution of Ideology Scores")
    ax1.set_xlabel("NOMINATE Dimension 1 Score")
    ax1.set_ylabel("Count")
    ax1.legend()
    
    # Plot 2: Boxplot of ideology scores
    ax2 = axes[0, 1]
    sns.boxplot(y=df_clean["nominate_dim1"], ax=ax2, color=COLORS[1])
    ax2.set_title("Boxplot of Ideology Scores")
    ax2.set_ylabel("NOMINATE Dimension 1 Score")
    
    # Plot 3: KDE plot by ideology category
    ax3 = axes[1, 0]
    sns.kdeplot(data=df_clean, x="nominate_dim1", hue="ideology_category", 
                fill=True, common_norm=False, palette=COLORS, ax=ax3)
    ax3.set_title("Density by Ideology Category")
    ax3.set_xlabel("NOMINATE Dimension 1 Score")
    
    # Plot 4: Count by ideology category
    ax4 = axes[1, 1]
    sns.countplot(x="ideology_category", data=df_clean, palette=COLORS, ax=ax4)
    ax4.set_title("Count by Ideology Category")
    ax4.set_xlabel("Ideology Category")
    ax4.set_ylabel("Count")
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "ideology_distribution.png", dpi=300)
    
    return fig

def visualize_text_properties(df_clean):
    """Analyze and visualize key text properties."""
    print("\nAnalyzing text properties...")
    
    # Create figure with multiple plots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Distribution of text lengths (word count)
    ax1 = axes[0, 0]
    sns.histplot(df_clean["word_count"].clip(upper=5000), bins=40, kde=True, ax=ax1, color=COLORS[0])
    ax1.set_title("Distribution of Text Lengths")
    ax1.set_xlabel("Word Count (clipped at 5000)")
    ax1.set_ylabel("Count")
    
    # Plot 2: Boxplot of text lengths by ideology category
    ax2 = axes[0, 1]
    sns.boxplot(x="ideology_category", y="word_count", data=df_clean, ax=ax2, palette=COLORS)
    ax2.set_ylim(0, 5000)  # Clip for better visibility
    ax2.set_title("Text Length by Ideology Category")
    ax2.set_xlabel("Ideology Category")
    ax2.set_ylabel("Word Count")
    
    # Plot 3: Scatter plot of ideology score vs text length
    ax3 = axes[1, 0]
    sns.scatterplot(x="nominate_dim1", y="word_count", data=df_clean, 
                    hue="ideology_category", palette=COLORS, alpha=0.6, ax=ax3)
    ax3.set_ylim(0, 5000)  # Clip for better visibility
    ax3.set_title("Ideology Score vs Text Length")
    ax3.set_xlabel("NOMINATE Dimension 1 Score")
    ax3.set_ylabel("Word Count")
    
    # Plot 4: Text length percentiles
    ax4 = axes[1, 1]
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    values = np.percentile(df_clean["word_count"], percentiles)
    ax4.bar(range(len(percentiles)), values, color=COLORS[2])
    ax4.set_xticks(range(len(percentiles)))
    ax4.set_xticklabels([f"{p}th" for p in percentiles])
    ax4.set_title("Text Length Percentiles")
    ax4.set_xlabel("Percentile")
    ax4.set_ylabel("Word Count")
    
    # Add value labels to the bars
    for i, v in enumerate(values):
        ax4.text(i, v + 50, f"{v:.0f}", ha='center')
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "text_properties.png", dpi=300)
    
    return fig

def analyze_vocabulary(df_clean):
    """Analyze vocabulary usage across ideological categories."""
    print("\nAnalyzing vocabulary differences by ideology...")
    
    # Create modified stopwords list
    stop_words = set(stopwords.words('english'))
    # Add additional common political stopwords
    political_stopwords = {
        'would', 'one', 'also', 'us', 'may', 'many', 'must', 
        'congress', 'senate', 'house', 'representative', 'representative',
        'constituent', 'district', 'state', 'states', 'united', 
        'america', 'american', 'americans'
    }
    stop_words.update(political_stopwords)
    
    # Function to extract top terms for each category
    def get_top_terms(texts, n=30):
        # Tokenize and clean
        all_words = []
        for text in texts:
            # Convert to lowercase and tokenize
            tokens = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
            # Filter out stopwords
            filtered_tokens = [word for word in tokens if word not in stop_words]
            all_words.extend(filtered_tokens)
        
        # Count and return top N
        word_counts = Counter(all_words)
        return word_counts.most_common(n)
    
    # Dictionary to store results
    category_terms = {}
    
    # Create plots for word clouds and top terms by category
    fig, axes = plt.subplots(len(df_clean["ideology_category"].unique()), 2, 
                            figsize=(16, 5 * len(df_clean["ideology_category"].unique())))
    
    # Process each ideological category
    for i, category in enumerate(sorted(df_clean["ideology_category"].unique())):
        # Filter texts for this category
        category_texts = df_clean[df_clean["ideology_category"] == category]["clean_text"]
        
        # Extract top terms
        top_terms = get_top_terms(category_texts)
        category_terms[category] = top_terms
        
        # Create word cloud
        text = " ".join(category_texts)
        wordcloud = WordCloud(width=800, height=400, 
                             background_color='white', 
                             stopwords=stop_words,
                             max_words=100,
                             colormap='viridis').generate(text)
        
        # Plot word cloud
        axes[i, 0].imshow(wordcloud, interpolation='bilinear')
        axes[i, 0].set_title(f"Word Cloud: {category}")
        axes[i, 0].axis('off')
        
        # Plot bar chart of top terms
        words, counts = zip(*top_terms[:15])  # Unzip top 15 terms
        axes[i, 1].barh(range(len(words)), counts, color=COLORS[i % len(COLORS)])
        axes[i, 1].set_yticks(range(len(words)))
        axes[i, 1].set_yticklabels(words)
        axes[i, 1].invert_yaxis()  # Most frequent at the top
        axes[i, 1].set_title(f"Top Terms: {category}")
        axes[i, 1].set_xlabel("Frequency")
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "vocabulary_analysis.png", dpi=300)
    
    return category_terms, fig

def topic_modeling(df_clean):
    """Perform topic modeling on the dataset."""
    print("\nPerforming topic modeling...")
    
    # Process by ideology category
    categories = sorted(df_clean["ideology_category"].unique())
    n_topics = 3  # Number of topics to extract per category
    
    # Initialize results storage
    results = {}
    
    # Create figure for visualization
    fig, axes = plt.subplots(len(categories), 1, figsize=(14, 5 * len(categories)))
    
    # Configure text vectorizer
    vectorizer = CountVectorizer(
        max_features=5000,
        min_df=5,
        max_df=0.7,
        stop_words='english'
    )
    
    # Process each category
    for i, category in enumerate(categories):
        # Filter texts for this category
        texts = df_clean[df_clean["ideology_category"] == category]["clean_text"].tolist()
        
        if len(texts) < 10:  # Skip if too few documents
            print(f"  Skipping {category} - insufficient data")
            results[category] = None
            continue
            
        print(f"  Modeling topics for {category} ({len(texts)} documents)")
        
        # Vectorize the texts
        X = vectorizer.fit_transform(texts)
        
        # Apply LDA
        lda = LatentDirichletAllocation(
            n_components=n_topics,
            random_state=42,
            max_iter=20
        )
        lda.fit(X)
        
        # Extract the most important words for each topic
        feature_names = vectorizer.get_feature_names_out()
        topics = []
        
        for topic_idx, topic in enumerate(lda.components_):
            top_words_idx = topic.argsort()[:-11:-1]  # Get indices of top 10 words
            top_words = [feature_names[i] for i in top_words_idx]
            topic_dict = {
                "topic_id": topic_idx,
                "words": top_words,
                "weights": topic[top_words_idx].tolist()
            }
            topics.append(topic_dict)
        
        results[category] = topics
        
        # Visualize top words for each topic
        ax = axes[i] if len(categories) > 1 else axes
        ax.set_title(f"Top Topics for {category}")
        ax.set_xlabel("Weight")
        ax.set_ylabel("Word")
        
        # Plot top words for each topic side by side
        width_offset = 0
        bar_width = 0.8 / n_topics
        
        for t, topic in enumerate(topics):
            words = topic["words"]
            weights = topic["weights"]
            
            # Ensure weights sum to 1 for fair comparison
            normalized_weights = np.array(weights) / sum(weights)
            
            # Create positions for side-by-side bars
            positions = np.arange(len(words)) + width_offset
            ax.barh(positions, normalized_weights, 
                   height=bar_width, 
                   color=COLORS[t % len(COLORS)],
                   label=f"Topic {t+1}")
            
            width_offset += bar_width
        
        # Add word labels
        ax.set_yticks(np.arange(len(words)) + (bar_width * (n_topics - 1) / 2))
        ax.set_yticklabels(topics[0]["words"])  # Use words from first topic
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "topic_modeling.png", dpi=300)
    
    return results, fig

def comparative_analysis(df_clean):
    """Compare key linguistic features across ideological categories."""
    print("\nPerforming comparative linguistic analysis...")
    
    # Initialize metrics to collect
    categories = sorted(df_clean["ideology_category"].unique())
    metrics = {
        "avg_word_length": [],
        "sentence_count": [],
        "avg_sentence_length": [],
        "vocabulary_size": [],
        "lexical_diversity": []
    }
    
    # Helper function to calculate metrics for a text
    def calculate_text_metrics(text):
        # Word metrics
        words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        avg_word_length = sum(len(word) for word in words) / len(words) if words else 0
        
        # Sentence metrics
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        sentence_count = len(sentences)
        
        words_per_sentence = [len(re.findall(r'\b[a-zA-Z]+\b', s.lower())) for s in sentences]
        avg_sentence_length = sum(words_per_sentence) / sentence_count if sentence_count else 0
        
        # Vocabulary and diversity
        vocab = set(words)
        vocabulary_size = len(vocab)
        lexical_diversity = vocabulary_size / len(words) if words else 0
        
        return {
            "avg_word_length": avg_word_length,
            "sentence_count": sentence_count,
            "avg_sentence_length": avg_sentence_length,
            "vocabulary_size": vocabulary_size,
            "lexical_diversity": lexical_diversity
        }
    
    # Calculate metrics for each category
    category_metrics = {}
    
    for category in categories:
        texts = df_clean[df_clean["ideology_category"] == category]["clean_text"].tolist()
        
        # Process each text and average the results
        all_metrics = [calculate_text_metrics(text) for text in texts]
        
        # Average each metric
        avg_metrics = {
            metric: sum(item[metric] for item in all_metrics) / len(all_metrics)
            for metric in metrics.keys()
        }
        
        category_metrics[category] = avg_metrics
    
    # Convert to DataFrame for easier visualization
    metrics_df = pd.DataFrame.from_dict(category_metrics, orient='index')
    
    # Create visualization
    fig, axes = plt.subplots(3, 2, figsize=(15, 14))
    axes = axes.flatten()
    
    # Plot each metric
    for i, metric in enumerate(metrics_df.columns):
        if i >= len(axes):
            break
            
        sns.barplot(x=metrics_df.index, y=metrics_df[metric], ax=axes[i], palette=COLORS)
        axes[i].set_title(f"{metric.replace('_', ' ').title()} by Ideology")
        axes[i].set_xlabel("Ideology Category")
        axes[i].set_ylabel(metric.replace('_', ' ').title())
        
        # Add value labels
        for j, v in enumerate(metrics_df[metric]):
            axes[i].text(j, v * 1.01, f"{v:.2f}", ha='center')
    
    # Add a summary table in the last subplot
    if len(axes) > len(metrics_df.columns):
        axes[-1].axis('off')
        table_data = [[f"{v:.2f}" for v in row] for row in metrics_df.values]
        table = axes[-1].table(
            cellText=table_data,
            rowLabels=metrics_df.index,
            colLabels=[col.replace('_', ' ').title() for col in metrics_df.columns],
            cellLoc='center',
            loc='center',
            bbox=[0.1, 0.1, 0.8, 0.8]
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        axes[-1].set_title("Summary of Linguistic Metrics")
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "comparative_linguistics.png", dpi=300)
    
    return metrics_df, fig

def generate_report(df, df_clean):
    """Generate a comprehensive analysis report."""
    print("\nGenerating comprehensive analysis report...")
    
    # Create HTML report - fixed by escaping curly braces in CSS with double braces {{ }}
    html_content = """
    <html>
    <head>
        <title>Political Ideology Dataset Analysis</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
            h1 {{ color: #2c3e50; }}
            h2 {{ color: #3498db; margin-top: 30px; }}
            h3 {{ color: #2980b9; }}
            img {{ max-width: 100%; margin: 20px 0; border: 1px solid #ddd; }}
            table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
            th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
            th {{ background-color: #f2f2f2; }}
            tr:hover {{ background-color: #f5f5f5; }}
            .section {{ margin: 40px 0; }}
            .stats {{ background-color: #f9f9f9; padding: 20px; border-radius: 5px; }}
        </style>
    </head>
    <body>
        <h1>Political Ideology Dataset Analysis</h1>
        
        <div class="section">
            <h2>Dataset Overview</h2>
            <div class="stats">
                <p><strong>Total Records:</strong> {total_records}</p>
                <p><strong>Complete Records:</strong> {complete_records}</p>
                <p><strong>Missing Ideology Scores:</strong> {missing_ideology}</p>
                <p><strong>Missing Text Data:</strong> {missing_text}</p>
            </div>
            
            <h3>Ideology Score Statistics</h3>
            <div class="stats">
                <p><strong>Mean:</strong> {ideology_mean:.4f}</p>
                <p><strong>Median:</strong> {ideology_median:.4f}</p>
                <p><strong>Min:</strong> {ideology_min:.4f}</p>
                <p><strong>Max:</strong> {ideology_max:.4f}</p>
                <p><strong>Standard Deviation:</strong> {ideology_std:.4f}</p>
            </div>
            
            <h3>Text Length Statistics</h3>
            <div class="stats">
                <p><strong>Mean:</strong> {text_mean:.2f} words</p>
                <p><strong>Median:</strong> {text_median:.2f} words</p>
                <p><strong>Min:</strong> {text_min} words</p>
                <p><strong>Max:</strong> {text_max} words</p>
                <p><strong>Standard Deviation:</strong> {text_std:.2f} words</p>
            </div>
            
            <h3>Ideological Categories Distribution</h3>
            <table>
                <tr>
                    <th>Category</th>
                    <th>Count</th>
                    <th>Percentage</th>
                </tr>
                {category_rows}
            </table>
        </div>
        
        <div class="section">
            <h2>Ideology Distribution</h2>
            <p>This section shows the distribution of ideology scores in the dataset.</p>
            <img src="ideology_distribution.png" alt="Ideology Distribution">
        </div>
        
        <div class="section">
            <h2>Text Properties</h2>
            <p>Analysis of text length and other properties across different ideology categories.</p>
            <img src="text_properties.png" alt="Text Properties">
        </div>
        
        <div class="section">
            <h2>Vocabulary Analysis</h2>
            <p>Word clouds and top terms for each ideological category.</p>
            <img src="vocabulary_analysis.png" alt="Vocabulary Analysis">
        </div>
        
        <div class="section">
            <h2>Topic Modeling</h2>
            <p>Latent Dirichlet Allocation (LDA) results showing main topics for each ideological category.</p>
            <img src="topic_modeling.png" alt="Topic Modeling">
        </div>
        
        <div class="section">
            <h2>Comparative Linguistic Analysis</h2>
            <p>Comparison of linguistic features across ideological categories.</p>
            <img src="comparative_linguistics.png" alt="Comparative Linguistics">
        </div>
    </body>
    </html>
    """
    
    # The rest of the function stays the same...
    # Generate category rows for the table
    category_rows = ""
    ideology_dist = df_clean["ideology_category"].value_counts().sort_index()
    for category, count in ideology_dist.items():
        percentage = count / len(df_clean) * 100
        category_rows += f"<tr><td>{category}</td><td>{count}</td><td>{percentage:.1f}%</td></tr>\n"
    
    # Format with actual values
    html_content = html_content.format(
        total_records=len(df),
        complete_records=len(df_clean),
        missing_ideology=df["nominate_dim1"].isna().sum(),
        missing_text=df["clean_text"].isna().sum(),
        ideology_mean=df_clean["nominate_dim1"].mean(),
        ideology_median=df_clean["nominate_dim1"].median(),
        ideology_min=df_clean["nominate_dim1"].min(),
        ideology_max=df_clean["nominate_dim1"].max(),
        ideology_std=df_clean["nominate_dim1"].std(),
        text_mean=df_clean["word_count"].mean(),
        text_median=df_clean["word_count"].median(),
        text_min=df_clean["word_count"].min(),
        text_max=df_clean["word_count"].max(),
        text_std=df_clean["word_count"].std(),
        category_rows=category_rows
    )
    
    # Save HTML report
    with open(RESULTS_DIR / "data_analysis_report.html", "w") as f:
        f.write(html_content)
    
    print(f"Report saved to {RESULTS_DIR / 'data_analysis_report.html'}")

def main():
    """Execute the full data analysis pipeline."""
    # Load and prepare data
    df, df_clean = load_clean_data()
    
    # Generate basic overview
    dataset_overview(df, df_clean)
    
    # Generate visualizations and analyses
    visualize_ideology_distribution(df_clean)
    visualize_text_properties(df_clean)
    analyze_vocabulary(df_clean)
    topic_modeling(df_clean)
    comparative_analysis(df_clean)
    
    # Generate comprehensive report
    generate_report(df, df_clean)
    
    print("\nAnalysis complete. All results saved to:", RESULTS_DIR)

if __name__ == "__main__":
    main()
