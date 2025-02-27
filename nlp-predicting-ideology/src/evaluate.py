import pandas as pd
import numpy as np
import joblib
import torch
import logging
from pathlib import Path
from typing import List, Tuple, Union
from transformers import (
    BertTokenizer, BertForSequenceClassification,
    RobertaTokenizer, RobertaForSequenceClassification,
    PreTrainedTokenizer, PreTrainedModel
)
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import embedding_utils
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing.text import Tokenizer # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore

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
        y = df["nominate_dim1"]
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

    # Train-Test Split
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        df["clean_text"], y, test_size=0.2, random_state=42
    )

    model_results = []

    # Ridge Regression Evaluation
    try:
        vectorizer = joblib.load("../models/tfidf_vectorizer.pkl")
        X_test_tfidf = vectorizer.transform(X_test_text)
        ridge_model = joblib.load("../models/ridge_regression.pkl")
        ridge_preds = ridge_model.predict(X_test_tfidf)
        model_results.append(evaluate_predictions("Ridge Regression", y_test, ridge_preds))
    except Exception as e:
        logger.error(f"Error in Ridge evaluation: {e}")

    # XGBoost Evaluation
    try:
        X_word2vec, _ = embedding_utils.get_word2vec_embeddings(X_test_text)
        xgb_model = joblib.load("../models/xgboost.pkl")
        xgb_preds = xgb_model.predict(X_word2vec)
        model_results.append(evaluate_predictions("XGBoost", y_test, xgb_preds))
    except Exception as e:
        logger.error(f"Error in XGBoost evaluation: {e}")

    # BERT Evaluation
    try:
        bert_preds = evaluate_transformer_model(
            model_type="bert",
            model_path="../models/bert_model",
            texts=X_test_text,
            device=device
        )
        model_results.append(evaluate_predictions("BERT Fine-Tuning", y_test, bert_preds))
    except Exception as e:
        logger.error(f"Error in BERT evaluation: {e}")
        
    # RoBERTa Evaluation
    try:
        roberta_preds = evaluate_transformer_model(
            model_type="roberta",
            model_path="../models/roberta_model",
            texts=X_test_text,
            device=device
        )
        model_results.append(evaluate_predictions("RoBERTa Fine-Tuning", y_test, roberta_preds))
    except Exception as e:
        logger.error(f"Error in RoBERTa evaluation: {e}")

    # Create results DataFrame
    results_df = pd.DataFrame(model_results, columns=["Model", "MSE", "MAE", "R² Score"])
    
    # Save results
    try:
        results_path = Path("../models/model_comparison.csv")
        results_df.to_csv(results_path, index=False)
        logger.info(f"Results saved to {results_path}")
    except Exception as e:
        logger.error(f"Error saving results: {e}")

    return results_df

def evaluate_predictions(model_name: str, y_true: np.ndarray, y_pred: np.ndarray) -> List:
    """Evaluate model predictions using multiple metrics."""
    return [
        model_name,
        mean_squared_error(y_true, y_pred),
        mean_absolute_error(y_true, y_pred),
        r2_score(y_true, y_pred)
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
        model_type: Type of model ('bert' or 'roberta')
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
    elif model_type.lower() == "roberta":
        tokenizer = RobertaTokenizer.from_pretrained(model_path)
        model = RobertaForSequenceClassification.from_pretrained(model_path).to(device)
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
    print("\nModel Comparison Results:")
    print(results)
