"""
Train XGBoost model for political ideology prediction using Word2Vec embeddings.
"""
import os
import logging
import numpy as np
import joblib
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import embedding_utils

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join("..", "logs", "xgboost_training.log"), mode='a')
    ]
)
logger = logging.getLogger("train_xgboost")

# Create necessary directories
embedding_utils.ensure_dirs_exist()

def train_xgboost():
    """Train and evaluate XGBoost regression model."""
    logger.info("Starting XGBoost training process...")
    
    # Load and prepare data
    df = embedding_utils.load_data()
    X_train_text, X_test_text, y_train, y_test, _ = embedding_utils.get_train_test_split(df)
    
    # Generate word2vec embeddings
    logger.info("Generating Word2Vec embeddings...")
    X_train_vec, word2vec_model = embedding_utils.get_word2vec_embeddings(X_train_text)
    
    # Process test set using the same Word2Vec model
    def get_test_embeddings(test_texts, model):
        """Convert test texts to embeddings using trained Word2Vec model."""
        vector_size = model.wv.vector_size
        
        def average_word_vectors(text):
            words = text.split()
            vectors = [model.wv[word] for word in words if word in model.wv]
            return np.mean(vectors, axis=0) if vectors else np.zeros(vector_size)
        
        return np.array([average_word_vectors(text) for text in test_texts])
    
    X_test_vec = get_test_embeddings(X_test_text, word2vec_model)
    
    # Define XGBoost parameters
    params = {
        'objective': 'reg:squarederror',
        'learning_rate': 0.1,
        'max_depth': 5,
        'min_child_weight': 1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'eta': 0.3,
        'seed': 42
    }
    
    logger.info("Training XGBoost model...")
    # Train XGBoost model
    dtrain = xgb.DMatrix(X_train_vec, label=y_train)
    dtest = xgb.DMatrix(X_test_vec, label=y_test)
    
    # Set up watchlist for early stopping
    watchlist = [(dtrain, 'train'), (dtest, 'eval')]
    
    # Train model
    model = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=1000,
        evals=watchlist,
        early_stopping_rounds=20,
        verbose_eval=50
    )
    
    # Make predictions
    y_pred = model.predict(dtest)
    
    # Evaluate model
    metrics = embedding_utils.evaluate_regression_model(y_test, y_pred)
    embedding_utils.print_metrics(metrics, "XGBoost")
    embedding_utils.save_model_metrics(metrics, "XGBoost")
    
    # Plot predictions
    embedding_utils.plot_predictions(y_test, y_pred, "XGBoost")
    
    # Save model and vectorizer
    logger.info("Saving model artifacts...")
    model_path = os.path.join("..", "models", "xgboost.pkl")
    word2vec_path = os.path.join("..", "models", "word2vec.pkl")
    
    joblib.dump(model, model_path)
    joblib.dump(word2vec_model, word2vec_path)
    
    logger.info(f"Model saved to {model_path}")
    logger.info(f"Word2Vec model saved to {word2vec_path}")
    
    # Print summary
    print("\nXGBoost Model Training Complete!")
    print(f"MSE: {metrics['MSE']:.4f} | MAE: {metrics['MAE']:.4f} | R² Score: {metrics['R²']:.4f}")
    
    return metrics

if __name__ == "__main__":
    train_xgboost()
