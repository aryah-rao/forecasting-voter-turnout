"""
Utility functions for text embeddings and model evaluation.

This module provides standardized functions for:
1. Data loading and preprocessing
2. Text embedding generation (TF-IDF, Word2Vec, BERT, RoBERTa)
3. Model evaluation and metrics reporting
4. Common configuration settings

All model training scripts should use these utilities for consistency.
"""

import os
import json
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict, Any, List, Union, Optional
from pathlib import Path

# ML libraries
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from gensim.models import Word2Vec
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join("..", "logs", "processing.log"), mode='a')
    ]
)
logger = logging.getLogger("embedding_utils")

# Create logs directory if it doesn't exist
os.makedirs(os.path.join("..", "logs"), exist_ok=True)

# Project configuration
CONFIG = {
    "data_path": "../data/final_dataset.csv",
    "models_dir": "../models",
    "min_words": 500,
    "max_words": 2700,
    "random_seed": 42,
    "test_size": 0.2,
    "tfidf_max_features": 5000,
    "word2vec_dim": 100,
    "word2vec_window": 5,
    "word2vec_min_count": 2,
    "bert_model_name": "all-MiniLM-L6-v2",
    "max_sequence_length": 256,
}

def load_data(filter_by_length: bool = False) -> pd.DataFrame:
    """
    Load and preprocess dataset from the data directory.
    
    Args:
        filter_by_length: Whether to filter text by min/max word count
    
    Returns:
        DataFrame with cleaned and filtered data
    """
    logger.info(f"Loading data from {CONFIG['data_path']}...")
    
    try:
        df = pd.read_csv(CONFIG['data_path'])
        
        # Drop rows with NaN values in critical columns
        initial_count = len(df)
        df = df.dropna(subset=["clean_text"])
        if initial_count - len(df) > 0:
            logger.warning(f"Dropped {initial_count - len(df)} rows with missing text data.")
        
        # Ensure nominate_dim1 is numeric and drop NaN values
        df["nominate_dim1"] = pd.to_numeric(df["nominate_dim1"], errors="coerce")
        
        initial_count = len(df)
        df = df.dropna(subset=["nominate_dim1"])
        if initial_count - len(df) > 0:
            logger.warning(f"Dropped {initial_count - len(df)} rows with missing ideology scores.")
        
        # Calculate word count and filter by length if required
        if filter_by_length:
            df["word_count"] = df["clean_text"].apply(lambda text: len(str(text).split()))
            
            initial_count = len(df)
            df = df[(df["word_count"] >= CONFIG["min_words"]) & 
                    (df["word_count"] <= CONFIG["max_words"])]
            
            if initial_count - len(df) > 0:
                logger.info(f"Filtered out {initial_count - len(df)} rows based on word count.")
            
            # Drop word_count column after filtering
            df = df.drop(columns=["word_count"])
        
        logger.info(f"Successfully loaded {len(df)} valid samples.")
        return df
        
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def get_train_test_split(
    df: pd.DataFrame, 
    text_col: str = "clean_text", 
    label_col: str = "nominate_dim1",
    normalize: bool = False
) -> Tuple:
    """
    Split data into training and testing sets with optional normalization.
    
    Args:
        df: DataFrame containing data
        text_col: Column name containing text
        label_col: Column name containing labels
        normalize: Whether to normalize the target variable
    
    Returns:
        Tuple of (X_train, X_test, y_train, y_test, normalization_params)
        normalization_params is None if normalize=False
    """
    from sklearn.model_selection import train_test_split
    
    X = df[text_col]
    y = df[label_col].values
    
    norm_params = None
    if normalize:
        # Normalize y values for better training stability
        y_mean, y_std = y.mean(), y.std()
        y = (y - y_mean) / y_std
        norm_params = {"mean": y_mean, "std": y_std}
        logger.info(f"Normalized target variable: mean={y_mean:.4f}, std={y_std:.4f}")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=CONFIG["test_size"], 
        random_state=CONFIG["random_seed"]
    )
    
    logger.info(f"Data split: train={len(X_train)} samples, test={len(X_test)} samples")
    return X_train, X_test, y_train, y_test, norm_params

def get_tfidf_embeddings(texts: pd.Series, max_features: int = None) -> Tuple:
    """
    Generate TF-IDF embeddings for texts.
    
    Args:
        texts: Series of text documents
        max_features: Maximum number of features for TF-IDF
    
    Returns:
        Tuple of (embedding matrix, vectorizer)
    """
    if max_features is None:
        max_features = CONFIG["tfidf_max_features"]
        
    logger.info(f"Generating TF-IDF embeddings with {max_features} features...")
    
    try:
        vectorizer = TfidfVectorizer(max_features=max_features)
        embeddings = vectorizer.fit_transform(texts)
        
        logger.info(f"Generated TF-IDF embeddings with shape: {embeddings.shape}")
        return embeddings, vectorizer
    except Exception as e:
        logger.error(f"Error generating TF-IDF embeddings: {e}")
        raise

def get_word2vec_embeddings(
    texts: pd.Series, 
    vector_size: int = None,
    window: int = None,
    min_count: int = None
) -> Tuple:
    """
    Generate Word2Vec embeddings by averaging word vectors for each document.
    
    Args:
        texts: Series of text documents
        vector_size: Dimensionality of word vectors
        window: Context window size
        min_count: Minimum word frequency threshold
    
    Returns:
        Tuple of (embedding matrix, word2vec model)
    """
    if vector_size is None:
        vector_size = CONFIG["word2vec_dim"]
    if window is None:
        window = CONFIG["word2vec_window"]
    if min_count is None:
        min_count = CONFIG["word2vec_min_count"]
        
    logger.info(f"Training Word2Vec model with {vector_size} dimensions...")
    
    try:
        # Tokenize text into sentences of words
        sentences = [text.split() for text in texts]
        
        # Train Word2Vec model
        model = Word2Vec(
            sentences, 
            vector_size=vector_size, 
            window=window, 
            min_count=min_count, 
            workers=4,
            seed=CONFIG["random_seed"]
        )
        
        # Convert texts into averaged word vectors
        def average_word_vectors(text):
            words = text.split()
            vectors = [model.wv[word] for word in words if word in model.wv]
            return np.mean(vectors, axis=0) if vectors else np.zeros(vector_size)
        
        X_word2vec = np.array([average_word_vectors(text) for text in texts])
        
        logger.info(f"Generated Word2Vec embeddings with shape: {X_word2vec.shape}")
        return X_word2vec, model
    except Exception as e:
        logger.error(f"Error generating Word2Vec embeddings: {e}")
        raise

def get_transformer_embeddings(
    texts: pd.Series, 
    model_name: str = None,
    batch_size: int = 32
) -> Tuple:
    """
    Generate embeddings using a pre-trained transformer model.
    
    Args:
        texts: Series of text documents
        model_name: Name of the pre-trained model
        batch_size: Batch size for encoding
    
    Returns:
        Tuple of (embedding matrix, model)
    """
    if model_name is None:
        model_name = CONFIG["bert_model_name"]
    
    logger.info(f"Generating embeddings using {model_name}...")
    
    try:
        model = SentenceTransformer(model_name)
        # Convert to list as SentenceTransformer expects list input
        text_list = texts.tolist()
        
        embeddings = model.encode(
            text_list,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        logger.info(f"Generated transformer embeddings with shape: {embeddings.shape}")
        return embeddings, model
    except Exception as e:
        logger.error(f"Error generating transformer embeddings: {e}")
        raise

def evaluate_regression_model(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Evaluate regression model performance using multiple metrics.
    
    Args:
        y_true: True target values
        y_pred: Predicted target values
    
    Returns:
        Dictionary of evaluation metrics
    """
    metrics = {
        "MSE": mean_squared_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAE": mean_absolute_error(y_true, y_pred),
        "R²": r2_score(y_true, y_pred)
    }
    return metrics

def print_metrics(metrics: Dict[str, float], model_name: str = "Model") -> None:
    """
    Print evaluation metrics in a formatted way.
    
    Args:
        metrics: Dictionary of metrics
        model_name: Name of the model being evaluated
    """
    logger.info(f"\n{model_name} Performance:")
    for metric_name, value in metrics.items():
        logger.info(f"  {metric_name}: {value:.4f}")

def save_model_metrics(
    metrics: Dict[str, float], 
    model_name: str,
    save_path: Optional[str] = None
) -> None:
    """
    Save model metrics to a JSON file.
    
    Args:
        metrics: Dictionary of metrics
        model_name: Name of the model
        save_path: Path to save the metrics file (optional)
    """
    if save_path is None:
        save_path = os.path.join(CONFIG["models_dir"], "metrics")
    
    os.makedirs(save_path, exist_ok=True)
    file_path = os.path.join(save_path, f"{model_name.lower().replace(' ', '_')}_metrics.json")
    
    with open(file_path, 'w') as f:
        json.dump({
            "model_name": model_name,
            "metrics": metrics,
            "timestamp": pd.Timestamp.now().isoformat()
        }, f, indent=2)
    
    logger.info(f"Metrics saved to {file_path}")

def plot_predictions(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    model_name: str,
    save_path: Optional[str] = None
) -> None:
    """
    Create and save scatter plot of predicted vs true values.
    
    Args:
        y_true: True target values
        y_pred: Predicted target values
        model_name: Name of the model for title
        save_path: Path to save the plot (optional)
    """
    if save_path is None:
        save_path = os.path.join(CONFIG["models_dir"], "plots")
    
    os.makedirs(save_path, exist_ok=True)
    file_path = os.path.join(save_path, f"{model_name.lower().replace(' ', '_')}_predictions.png")
    
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([-1, 1], [-1, 1], 'r--')
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.xlabel("True Ideology Score")
    plt.ylabel("Predicted Ideology Score")
    plt.title(f"{model_name} Predictions vs True Values")
    
    # Add metrics text box
    metrics = evaluate_regression_model(y_true, y_pred)
    metrics_text = "\n".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
    plt.text(-0.95, 0.7, metrics_text, fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(file_path, dpi=300)
    plt.close()
    
    logger.info(f"Prediction plot saved to {file_path}")

def ensure_dirs_exist() -> None:
    """Create necessary directories for model artifacts if they don't exist."""
    dirs = [
        os.path.join("..", "models"),
        os.path.join("..", "logs"),
        os.path.join("..", "models", "plots"),
        os.path.join("..", "models", "metrics"),
    ]
    
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
        logger.debug(f"Ensured directory exists: {dir_path}")