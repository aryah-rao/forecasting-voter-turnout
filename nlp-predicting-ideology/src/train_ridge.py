"""
Train Ridge Regression model for political ideology prediction using TF-IDF features.
"""
import os
import logging
import numpy as np
import joblib
from sklearn.linear_model import Ridge
import embedding_utils

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join("..", "logs", "ridge_training.log"), mode='a')
    ]
)
logger = logging.getLogger("train_ridge")

# Create necessary directories
embedding_utils.ensure_dirs_exist()

def train_ridge(alpha=0.1):
    """
    Train and evaluate Ridge Regression model with TF-IDF features.
    
    Args:
        alpha: Regularization strength for Ridge
    
    Returns:
        Dict containing evaluation metrics
    """
    logger.info(f"Starting Ridge Regression training process with alpha={alpha}...")
    
    # Load and prepare data
    df = embedding_utils.load_data()
    X_train_text, X_test_text, y_train, y_test, _ = embedding_utils.get_train_test_split(df)
    
    # Generate TF-IDF features
    logger.info("Generating TF-IDF features...")
    X_train_tfidf, vectorizer = embedding_utils.get_tfidf_embeddings(X_train_text)
    X_test_tfidf = vectorizer.transform(X_test_text)
    
    logger.info(f"TF-IDF features shape: train={X_train_tfidf.shape}, test={X_test_tfidf.shape}")
    
    # Initialize and train Ridge model
    logger.info("Training Ridge Regression model...")
    ridge_model = Ridge(alpha=alpha, random_state=42)
    ridge_model.fit(X_train_tfidf, y_train)
    
    # Make predictions
    y_pred = ridge_model.predict(X_test_tfidf)
    
    # Evaluate model
    metrics = embedding_utils.evaluate_regression_model(y_test, y_pred)
    embedding_utils.print_metrics(metrics, "Ridge Regression")
    embedding_utils.save_model_metrics(metrics, "Ridge_Regression")
    
    # Plot predictions
    embedding_utils.plot_predictions(y_test, y_pred, "Ridge_Regression")
    
    # Save model and vectorizer
    logger.info("Saving model artifacts...")
    model_path = os.path.join("..", "models", "ridge_regression.pkl")
    vectorizer_path = os.path.join("..", "models", "tfidf_vectorizer.pkl")
    
    joblib.dump(ridge_model, model_path)
    joblib.dump(vectorizer, vectorizer_path)
    
    logger.info(f"Model saved to {model_path}")
    logger.info(f"Vectorizer saved to {vectorizer_path}")
    
    # Print summary
    print("\nRidge Regression Model Training Complete!")
    print(f"MSE: {metrics['MSE']:.4f} | MAE: {metrics['MAE']:.4f} | R² Score: {metrics['R²']:.4f}")
    
    return metrics

if __name__ == "__main__":
    train_ridge(alpha=0.1)
