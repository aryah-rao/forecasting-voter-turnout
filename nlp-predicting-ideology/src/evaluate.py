import pandas as pd
import numpy as np
import joblib
import torch
import logging
from pathlib import Path
from typing import List, Tuple, Union, Dict
from transformers import (
    BertTokenizer, BertForSequenceClassification
)
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import embedding_utils

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def evaluate_models() -> pd.DataFrame:
    """
    Evaluate all models and return their performance metrics.
    Returns:
        pd.DataFrame: DataFrame containing model performance metrics
    """
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load Data
    try:
        df = embedding_utils.load_data()
        
        # Use the same train/test split function as in training scripts
        # This ensures we're evaluating on the exact same test data
        _, X_test_text, _, y_test, _ = embedding_utils.get_train_test_split(df)
        logger.info(f"Loaded test set with {len(X_test_text)} samples")
        
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

    model_results = []

    # Ridge Regression Evaluation
    try:
        logger.info("Evaluating Ridge Regression model...")
        vectorizer = joblib.load("../models/tfidf_vectorizer.pkl")
        X_test_tfidf = vectorizer.transform(X_test_text)
        ridge_model = joblib.load("../models/ridge_regression.pkl")
        ridge_preds = ridge_model.predict(X_test_tfidf)
        model_results.append(evaluate_predictions("Ridge Regression", y_test, ridge_preds))
    except Exception as e:
        logger.error(f"Error in Ridge evaluation: {e}")

    # BERT Evaluation
    try:
        logger.info("Evaluating BERT model...")
        # Load normalization parameters used during BERT training
        bert_norm_params = load_normalization_params("../models/bert_norm_params.npz")
        
        # Get raw predictions from the model
        bert_preds_raw = evaluate_transformer_model(
            model_type="bert",
            model_path="../models/bert_model",
            texts=X_test_text,
            device=device
        )
        
        # Denormalize predictions to match original scale
        bert_preds = bert_preds_raw * bert_norm_params["std"] + bert_norm_params["mean"]
        model_results.append(evaluate_predictions("BERT Fine-Tuning", y_test, bert_preds))
    except Exception as e:
        logger.error(f"Error in BERT evaluation: {e}")

    # Create results DataFrame
    results_df = pd.DataFrame(model_results, columns=["Model", "MSE", "MAE", "R² Score"])
    
    # Save results
    try:
        results_path = Path("../models/model_comparison.csv")
        results_df.to_csv(results_path, index=False)
        logger.info(f"Results saved to {results_path}")
    except Exception as e:
        logger.error(f"Error saving results: {e}")
    
    # Print results
    print("\nModel Comparison Results:")
    print(results_df)

    return results_df

def load_normalization_params(path: str) -> Dict[str, float]:
    """Load normalization parameters from npz file."""
    try:
        loaded = np.load(path)
        return {"mean": loaded["mean"], "std": loaded["std"]}
    except Exception as e:
        logger.error(f"Error loading normalization parameters from {path}: {e}")
        # Return default values as fallback
        return {"mean": 0.0, "std": 1.0}

def evaluate_predictions(model_name: str, y_true: np.ndarray, y_pred: np.ndarray) -> List:
    """Evaluate model predictions using multiple metrics."""
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    logger.info(f"{model_name} - MSE: {mse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
    
    return [
        model_name,
        mse,
        mae,
        r2
    ]

def evaluate_transformer_model(
    model_type: str,
    model_path: str,
    texts: pd.Series,
    device: torch.device
) -> np.ndarray:
    """
    Evaluate a transformer model on given texts.
    
    Args:
        model_type: Type of model ('bert')
        model_path: Path to the model
        texts: Input texts to evaluate
        device: Torch device to use
        
    Returns:
        np.ndarray: Model predictions
    """
    # Select appropriate model and tokenizer classes based on model_type
    if model_type.lower() == "bert":
        tokenizer = BertTokenizer.from_pretrained(model_path)
        model = BertForSequenceClassification.from_pretrained(model_path).to(device)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    logger.info(f"Evaluating {model_type} model from {model_path}")
    model.eval()

    # Process in batches to avoid memory issues
    batch_size = 16
    predictions = []
    texts_list = texts.tolist()
    
    for i in range(0, len(texts_list), batch_size):
        batch_texts = texts_list[i:i+batch_size]
        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            batch_preds = outputs.logits.squeeze().cpu().numpy()
            
            # Handle case where batch size is 1
            if len(batch_texts) == 1:
                batch_preds = np.array([batch_preds])
                
            predictions.append(batch_preds)
    
    # Combine batches
    all_predictions = np.concatenate(predictions) if len(predictions) > 1 else predictions[0]
    return all_predictions

if __name__ == "__main__":
    results = evaluate_models()
