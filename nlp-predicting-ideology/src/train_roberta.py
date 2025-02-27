"""
Train RoBERTa model for political ideology prediction using fine-tuning approach.
"""
import os
import logging
import numpy as np
import pandas as pd
import torch
import random
from datasets import Dataset
from transformers import (
    RobertaTokenizer, 
    RobertaForSequenceClassification, 
    Trainer, 
    TrainingArguments,
    EarlyStoppingCallback,
    DataCollatorWithPadding
)
import embedding_utils

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join("..", "logs", "roberta_training.log"), mode='a')
    ]
)
logger = logging.getLogger("train_roberta")

# Create necessary directories
embedding_utils.ensure_dirs_exist()

def set_seed(seed=42):
    """Set seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True

def compute_metrics(eval_pred):
    """Compute metrics for evaluation during training."""
    predictions, labels = eval_pred
    predictions = predictions.flatten()
    
    return {
        "mse": ((predictions - labels) ** 2).mean().item(),
        "mae": abs(predictions - labels).mean().item(),
    }

def preprocess_data(max_length=256):
    """
    Load and preprocess data for RoBERTa training.
    
    Args:
        max_length: Maximum sequence length for tokenization
        
    Returns:
        Processed train and validation datasets, normalization parameters
    """
    logger.info("Loading and preprocessing data...")
    
    # Load data
    df = embedding_utils.load_data()
    
    # Get train-test split with normalized targets
    X_train_text, X_test_text, y_train, y_test, norm_params = embedding_utils.get_train_test_split(
        df, normalize=True
    )
    
    # Create dataframes for datasets
    train_df = pd.DataFrame({"text": X_train_text, "label": y_train})
    val_df = pd.DataFrame({"text": X_test_text, "label": y_test})
    
    # Load tokenizer
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    
    # Tokenization function
    def tokenize_function(examples):
        return tokenizer(
            examples["text"], 
            truncation=True, 
            padding="max_length", 
            max_length=max_length
        )
    
    # Create and process datasets
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    
    # Tokenize datasets
    train_tokenized = train_dataset.map(tokenize_function, batched=True)
    val_tokenized = val_dataset.map(tokenize_function, batched=True)
    
    return train_tokenized, val_tokenized, norm_params

def train_roberta():
    """
    Train and evaluate RoBERTa model for ideology prediction.
    
    Returns:
        Dict containing evaluation metrics
    """
    logger.info("Starting RoBERTa training process...")
    set_seed(42)
    
    # Check for GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Preprocess data
    train_dataset, val_dataset, norm_params = preprocess_data(max_length=256)
    
    # Save normalization parameters
    np.savez(
        os.path.join("..", "models", "roberta_norm_params.npz"), 
        mean=norm_params["mean"], 
        std=norm_params["std"]
    )
    
    # Load model
    model = RobertaForSequenceClassification.from_pretrained(
    "roberta-large",
    num_labels=1,
    problem_type="regression",
    hidden_dropout_prob=0.2,
    attention_probs_dropout_prob=0.2
)
    
    # Create data collator for dynamic padding
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=os.path.join("..", "models", "roberta_checkpoints"),
        num_train_epochs=15,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        warmup_ratio=0.2,
        weight_decay=0.1,
        learning_rate=1e-5,
        fp16=torch.cuda.is_available(),
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="mse",
        greater_is_better=False,
        logging_dir=os.path.join("..", "logs", "roberta_logs"),
        logging_steps=50,
        report_to="none",
        gradient_accumulation_steps=4,
        max_grad_norm=1.0  # Add gradient clipping
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        data_collator=data_collator,
    )
    
    # Train model
    logger.info("Starting RoBERTa training...")
    trainer.train()
    
    # Evaluate model
    eval_results = trainer.evaluate()
    logger.info(f"Evaluation results: {eval_results}")
    
    # Save best model
    model_path = os.path.join("..", "models", "roberta_model")
    trainer.save_model(model_path)
    tokenizer.save_pretrained(model_path)
    logger.info(f"Model saved to {model_path}")
    
    # Final evaluation on validation set
    predictions = trainer.predict(val_dataset).predictions.flatten()
    labels = np.array(val_dataset["label"])
    
    # Denormalize predictions for reporting
    predictions_original = predictions * norm_params["std"] + norm_params["mean"]
    labels_original = labels * norm_params["std"] + norm_params["mean"]
    
    # Calculate metrics
    metrics = embedding_utils.evaluate_regression_model(labels_original, predictions_original)
    embedding_utils.print_metrics(metrics, "RoBERTa")
    embedding_utils.save_model_metrics(metrics, "RoBERTa")
    
    # Plot predictions
    embedding_utils.plot_predictions(labels_original, predictions_original, "RoBERTa")
    
    # Print summary
    print("\nRoBERTa Fine-Tuning Complete!")
    print(f"MSE: {metrics['MSE']:.4f} | MAE: {metrics['MAE']:.4f} | R² Score: {metrics['R²']:.4f}")
    
    return metrics

if __name__ == "__main__":
    train_roberta()
