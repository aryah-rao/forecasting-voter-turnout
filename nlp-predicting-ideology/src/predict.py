"""
Predict ideology scores for 2016 presidential candidates using trained NLP models.

This script:
1. Loads presidential candidate data with processed text
2. Uses pre-trained models (BERT and Ridge Regression) to predict ideology scores
3. Compares predictions with known political alignments
4. Visualizes results
"""
import os
import sys
import pandas as pd
import numpy as np
import joblib
import torch
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.linear_model import Ridge

# Import utility functions
from embedding_utils import (
    get_tfidf_embeddings,
    evaluate_regression_model,
    print_metrics,
    plot_predictions
)

# Define paths
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "..", "models")
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
PRESIDENTIAL_DATA_PATH = os.path.join(DATA_DIR, "presidential_candidates_with_text.csv")
RIDGE_MODEL_PATH = os.path.join(MODELS_DIR, "ridge_regression.pkl")
TFIDF_VECTORIZER_PATH = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")
BERT_MODEL_PATH = os.path.join(MODELS_DIR, "bert_model")
BERT_NORM_PARAMS_PATH = os.path.join(MODELS_DIR, "bert_norm_params.npz")
OUTPUT_PATH = os.path.join(DATA_DIR, "presidential_ideology_predictions.csv")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger("ideology_predictor")

def load_candidate_data() -> pd.DataFrame:
    """
    Load presidential candidate data.
    
    Returns:
        DataFrame with candidate data and text
    """
    logger.info(f"Loading presidential candidate data from {PRESIDENTIAL_DATA_PATH}")
    
    if not os.path.exists(PRESIDENTIAL_DATA_PATH):
        raise FileNotFoundError(f"Cannot find presidential candidate data at {PRESIDENTIAL_DATA_PATH}")
    
    df = pd.read_csv(PRESIDENTIAL_DATA_PATH)
    
    # Filter out candidates without clean text
    initial_count = len(df)
    df = df.dropna(subset=["clean_text"])
    df = df[df["clean_text"].str.strip() != ""]
    
    if initial_count - len(df) > 0:
        logger.warning(f"Filtered out {initial_count - len(df)} candidates without text data")
    
    logger.info(f"Loaded {len(df)} candidates with text data")
    return df

def load_models() -> Tuple:
    """
    Load the trained models for prediction.
    
    Returns:
        Tuple of (ridge_model, tfidf_vectorizer, bert_model, bert_tokenizer, bert_norm_params)
    """
    logger.info("Loading trained models...")
    
    # Check if models exist
    if not os.path.exists(RIDGE_MODEL_PATH):
        raise FileNotFoundError(f"Ridge model not found at {RIDGE_MODEL_PATH}")
    
    if not os.path.exists(TFIDF_VECTORIZER_PATH):
        raise FileNotFoundError(f"TF-IDF vectorizer not found at {TFIDF_VECTORIZER_PATH}")
    
    if not os.path.exists(BERT_MODEL_PATH):
        raise FileNotFoundError(f"BERT model not found at {BERT_MODEL_PATH}")
    
    if not os.path.exists(BERT_NORM_PARAMS_PATH):
        raise FileNotFoundError(f"BERT normalization parameters not found at {BERT_NORM_PARAMS_PATH}")
    
    # Load Ridge model and TF-IDF vectorizer
    ridge_model = joblib.load(RIDGE_MODEL_PATH)
    tfidf_vectorizer = joblib.load(TFIDF_VECTORIZER_PATH)
    
    # Load BERT model and tokenizer
    bert_model = BertForSequenceClassification.from_pretrained(BERT_MODEL_PATH)
    bert_tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_PATH)
    
    # Load BERT normalization parameters
    bert_norm_params = np.load(BERT_NORM_PARAMS_PATH)
    
    logger.info("Models loaded successfully")
    
    return ridge_model, tfidf_vectorizer, bert_model, bert_tokenizer, bert_norm_params

def predict_with_ridge(
    texts: List[str], 
    ridge_model: Ridge, 
    tfidf_vectorizer
) -> np.ndarray:
    """
    Predict ideology scores using Ridge Regression model.
    
    Args:
        texts: List of preprocessed text
        ridge_model: Trained Ridge Regression model
        tfidf_vectorizer: Fitted TF-IDF vectorizer
        
    Returns:
        Array of predicted ideology scores
    """
    logger.info("Making predictions with Ridge Regression model...")
    
    # Transform texts to TF-IDF features
    X_tfidf = tfidf_vectorizer.transform(texts)
    
    # Make predictions
    predictions = ridge_model.predict(X_tfidf)
    
    logger.info(f"Made {len(predictions)} predictions with Ridge Regression")
    return predictions

def predict_with_bert(
    texts: List[str], 
    bert_model: BertForSequenceClassification, 
    bert_tokenizer: BertTokenizer, 
    norm_params: Dict[str, float]
) -> np.ndarray:
    """
    Predict ideology scores using fine-tuned BERT model.
    
    Args:
        texts: List of preprocessed text
        bert_model: Fine-tuned BERT model
        bert_tokenizer: BERT tokenizer
        norm_params: Normalization parameters for denormalization
        
    Returns:
        Array of predicted ideology scores
    """
    logger.info("Making predictions with BERT model...")
    
    # Set model to evaluation mode
    bert_model.eval()
    
    # Initialize array for predictions
    predictions = []
    
    # Process in batches to avoid memory issues
    batch_size = 8
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        
        # Tokenize and prepare inputs
        inputs = bert_tokenizer(
            batch_texts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=512
        )
        
        # Make predictions
        with torch.no_grad():
            outputs = bert_model(**inputs)
        
        # Extract and denormalize predictions
        batch_preds = outputs.logits.numpy().flatten()
        batch_preds = batch_preds * norm_params["std"] + norm_params["mean"]
        
        predictions.extend(batch_preds.tolist())
    
    logger.info(f"Made {len(predictions)} predictions with BERT")
    return np.array(predictions)

def analyze_predictions(df: pd.DataFrame) -> None:
    """
    Analyze the model predictions and create visualizations.
    
    Args:
        df: DataFrame with candidate data and predictions
    """
    logger.info("Analyzing predictions...")
    
    # Calculate mean absolute error for each model, if actual ideology is available
    if "Ideology" in df.columns:
        # Map text ideology to numeric values (approximate)
        ideology_map = {
            "Far-Left (Progressive)": -0.8,
            "Far-Left (Democratic Socialist)": -0.8,
            "Center-Left": -0.4,
            "Center-Right": 0.4,
            "Right": 0.7,
            "Right-Populist": 0.7,
            "Far-Right": 0.9,
            "Libertarian-Right": 0.5,
            "Libertarian-Center": 0.1
        }
        
        # Create numeric ideology column if text values are provided
        if df["Ideology"].dtype == object:
            df["ideology_numeric"] = df["Ideology"].map(ideology_map)
            logger.info("Mapped text ideology to numeric values")
    
    # Create visualizations
    plt.figure(figsize=(12, 8))
    
    # Sort by BERT predictions
    df_sorted = df.sort_values(by="bert_prediction")
    
    # Plot candidates by ideology score
    plt.barh(df_sorted["Name"], df_sorted["bert_prediction"], label="BERT", alpha=0.7)
    plt.barh(df_sorted["Name"], df_sorted["ridge_prediction"], label="Ridge", alpha=0.7)
    
    
    plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    plt.grid(axis='x', linestyle='--', alpha=0.3)
    plt.xlabel("Ideology Score (-1 = Liberal, 1 = Conservative)")
    plt.title("Predicted Ideology Scores for 2016 Presidential Candidates")
    plt.legend()
    plt.tight_layout()
    
    # Save plot
    plt.savefig(os.path.join(DATA_DIR, "presidential_ideology_predictions.png"), dpi=300)
    logger.info(f"Visualization saved to {os.path.join(DATA_DIR, 'presidential_ideology_predictions.png')}")
    
    # Calculate agreement between models
    corr = df["bert_prediction"].corr(df["ridge_prediction"])
    logger.info(f"Correlation between BERT and Ridge predictions: {corr:.4f}")
    
    # Print party-wise average predictions
    if "Party" in df.columns:
        print("\nAverage Predictions by Party:")
        party_means = df.groupby("Party")[["bert_prediction", "ridge_prediction"]].mean()
        print(party_means)

def main():
    """
    Main function to run the ideology prediction pipeline.
    """
    try:
        # Load candidate data
        df = load_candidate_data()
        
        # Load trained models
        ridge_model, tfidf_vectorizer, bert_model, bert_tokenizer, bert_norm_params = load_models()
        
        # Extract cleaned texts
        texts = df["clean_text"].tolist()
        
        # Make predictions with Ridge Regression
        ridge_predictions = predict_with_ridge(texts, ridge_model, tfidf_vectorizer)
        df["ridge_prediction"] = ridge_predictions
        
        # Make predictions with BERT
        bert_predictions = predict_with_bert(
            texts, 
            bert_model, 
            bert_tokenizer, 
            {"mean": bert_norm_params["mean"], "std": bert_norm_params["std"]}
        )
        df["bert_prediction"] = bert_predictions
        
        # Save predictions
        logger.info(f"Saving predictions to {OUTPUT_PATH}")
        df.to_csv(OUTPUT_PATH, index=False)
        
        # Analyze and visualize predictions
        analyze_predictions(df)
        
        # Print summary
        print("\nPrediction Results:")
        print(f"{'Name':<20} {'Party':<12} {'BERT':<8} {'Ridge':<8}")
        print("-" * 50)
        for _, row in df.iterrows():
            name = row["Name"][:20]
            party = row.get("Party", "Unknown")[:12]
            bert_pred = f"{row['bert_prediction']:.2f}"
            ridge_pred = f"{row['ridge_prediction']:.2f}"
            print(f"{name:<20} {party:<12} {bert_pred:<8} {ridge_pred:<8}")
        
    except Exception as e:
        logger.error(f"Error in ideology prediction pipeline: {e}")
        raise

if __name__ == "__main__":
    logger.info("Starting presidential candidate ideology prediction")
    main()
    logger.info("Prediction complete!")
