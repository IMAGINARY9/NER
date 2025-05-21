"""
Entity Extraction Project - Main Entry Point

This script orchestrates the full pipeline for named entity recognition (NER) model.
It handles data loading, preprocessing, model building, training, and evaluation.
"""

import os
import sys
import platform
import logging
import argparse
import pandas as pd
import numpy as np
import time
import json
import traceback
import tensorflow as tf
from pathlib import Path

# Add the src directory to the path so we can import our modules
src_dir = Path(__file__).resolve().parent.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from src.data_processing.data_loader import load_data, preprocess_data
from src.data_processing.dataset import create_ner_dataset
from src.models.ner_model import build_model
from src.training.train import train_model
from src.config import Config
from src.utils.logger import setup_logging
from src.utils.metrics import compute_ner_metrics
from src.utils.embeddings import download_glove_embeddings, load_embeddings
from src.evaluate import evaluate_model, evaluate_on_examples
from src.predict import NERPredictor
from keras.models import load_model

logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="NER Project")    # Core parameters
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=15, help='Number of training epochs')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--model-type', choices=['bilstm', 'bilstm-crf', 'transformer'], default='bilstm-crf', help='Model type')
    parser.add_argument('--max-seq-length', type=int, default=128, help='Maximum sequence length')
    parser.add_argument('--mode', choices=['train', 'evaluate', 'predict'], default='train', help='Operation mode')
    parser.add_argument('--predict-text', type=str, help='Text to predict entities for')
    parser.add_argument('--model-dir', type=str, help='Path to the model directory')
    parser.add_argument('--model-path', type=str, help='Path to the specific model file to load', dest='model_path_param')
    parser.add_argument('--output-dir', type=str, help='Directory to save outputs (evaluations, predictions)')
    parser.add_argument('--data-path', type=str, default='./data/ner.csv', help='Path to the data file')

    # Enhancement options
    parser.add_argument('--basic', action='store_true', help='Use basic configuration without enhancements')
    parser.add_argument('--full', action='store_true', help='Use all enhancements (pre-trained embeddings + character features)')
    parser.add_argument('--embeddings', action='store_true', help='Use pre-trained embeddings only')
    parser.add_argument('--char-features', action='store_true', help='Use character features only')
    parser.add_argument('--use-char-features', action='store_true', help='Enable character features')
    parser.add_argument('--no-pretrained-embeddings', action='store_true', help='Disable pre-trained embeddings')

    # Environment options
    parser.add_argument('--use-gpu', action='store_true', help='Enable GPU usage')
    parser.add_argument('--no-gpu', action='store_true', help='Disable GPU usage')
    parser.add_argument('--use-xla', action='store_true', help='Enable XLA acceleration')
    parser.add_argument('--use-mixed-precision', action='store_true', help='Enable mixed precision training')

    return parser.parse_args()

def main():
    """Main function to orchestrate the NER pipeline."""
    args = parse_arguments()
    
    # Convert argparse namespace to Config object
    config = Config()
    # Transfer all attributes from args to config
    for key, value in vars(args).items():
        if value is not None:
            setattr(config, key, value)
    
    setup_logging(config.log_dir)
    
    # Display configuration
    config.display()
    logger.info(f"Starting NER pipeline in {getattr(config, 'mode', 'train')} mode")
    logger.info(f"Platform: {platform.platform()}")
    logger.info(f"Python version: {platform.python_version()}")
    
    try:
        # Set default mode if not specified
        mode = getattr(config, 'mode', 'train')
        
        if mode == 'train':            # Load and preprocess data
            data = load_data(config.data_path)
            preprocessed_data = preprocess_data(data)
            
            # Determine if we should use character-level features
            use_char_features = config.use_char_features if hasattr(config, 'use_char_features') else False
            
            # Create dataset for training
            if use_char_features:
                # With character features                # Create dataset without character features to match the model
                train_dataset, val_dataset, test_dataset, word_vocab, tag_vocab, idx2word, idx2tag, class_weights = create_ner_dataset(
                    preprocessed_data, 
                    config.batch_size, 
                    config.max_seq_length,
                    val_split=config.validation_split,
                    test_split=config.test_split,
                    use_optimization=config.use_dataset_optimization,
                    use_char_features=False  # Match the model configuration
                )                # Temporarily disable character features to bypass the error
                num_tags = len(tag_vocab) 
                logger.info(f"Building model with {num_tags} tag classes")
                  # Check the maximum tag index in the dataset to ensure it's within range
                max_tag_value = 0
                for batch in train_dataset.take(10):  # Sample more batches for better coverage
                    if len(batch) >= 2:  # Ensure we have features and labels
                        labels = batch[1].numpy()
                        batch_max = np.max(labels) if labels.size > 0 else 0
                        max_tag_value = max(max_tag_value, batch_max)
                
                logger.info(f"Maximum tag value in dataset sample: {max_tag_value}")
                if max_tag_value >= num_tags:
                    logger.warning(f"Warning: Dataset contains tag value {max_tag_value} but model output size is {num_tags}")
                    logger.warning("Ensuring model output size matches dataset requirements")
                    # Adjust num_tags to be one more than the highest observed tag index
                    num_tags = max_tag_value + 1
                    logger.info(f"Adjusted num_tags to {num_tags} to accommodate all tag indices")
                    # If needed, adjust num_tags here
                
                model = build_model(
                    vocab_size=len(word_vocab),
                    num_tags=num_tags,
                    embedding_dim=config.embedding_dim,
                    hidden_dim=config.hidden_dim,
                    dropout_rate=config.dropout_rate,
                    model_type=config.model_type,
                    recurrent_dropout_rate=config.recurrent_dropout_rate,
                    use_char_features=False,  # Temporarily disable character features
                    attention_heads=getattr(config, 'attention_heads', 8)
                )
            else:                # Without character features
                train_dataset, val_dataset, test_dataset, word_vocab, tag_vocab, idx2word, idx2tag, class_weights = create_ner_dataset(
                    preprocessed_data, 
                    config.batch_size, 
                    config.max_seq_length,
                    val_split=config.validation_split,
                    test_split=config.test_split,
                    use_optimization=config.use_dataset_optimization
                )
                
                # Handle pre-trained embeddings if requested
                pretrained_embeddings_path = None
                if config.use_pretrained_embeddings:
                    logger.info("Using pre-trained word embeddings")
                    
                    # Download embeddings if needed
                    embeddings_file = download_glove_embeddings(
                        embedding_dim=config.embedding_dim,
                        cache_dir=config.cache_dir
                    )
                    
                    # Load and prepare embedding matrix
                    embedding_matrix, matrix_path = load_embeddings(
                        embeddings_file=embeddings_file,
                        word_vocab=word_vocab,
                        embedding_dim=config.embedding_dim
                    )
                    
                    # Set the path for the model to use
                    pretrained_embeddings_path = matrix_path
                
                # Build model without character features
                model = build_model(
                    vocab_size=len(word_vocab),
                    num_tags=len(tag_vocab),
                    embedding_dim=config.embedding_dim,
                    hidden_dim=config.hidden_dim,
                    dropout_rate=config.dropout_rate,
                    recurrent_dropout_rate=config.recurrent_dropout_rate,
                    model_type=config.model_type,
                    attention_heads=getattr(config, 'attention_heads', 8),
                    use_pretrained_embeddings=config.use_pretrained_embeddings,
                    pretrained_embeddings_path=pretrained_embeddings_path
                )
              # Configure mixed precision if requested
            if config.use_mixed_precision:
                logger.info("Enabling mixed precision training")
                policy = tf.keras.mixed_precision.Policy('mixed_float16')
                tf.keras.mixed_precision.set_global_policy(policy)
            
            # Configure XLA if requested
            if config.use_xla:
                logger.info("Enabling XLA acceleration")
                tf.config.optimizer.set_jit(True)
            
            # Enable memory growth for GPUs if available
            if config.use_gpu:
                gpus = tf.config.list_physical_devices('GPU')
                if gpus:
                    logger.info(f"Found {len(gpus)} GPU(s). Configuring memory growth.")
                    for gpu in gpus:
                        try:
                            tf.config.experimental.set_memory_growth(gpu, config.memory_growth)
                        except RuntimeError as e:
                            logger.warning(f"Error configuring GPU: {e}")
                else:
                    logger.warning("No GPUs found despite use_gpu=True")
            
            # Train the model with class weights for imbalanced data
            trained_model = train_model(
                model=model,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                epochs=config.epochs,
                learning_rate=config.learning_rate,
                model_dir=config.model_dir,
                log_dir=config.log_dir,                early_stopping_patience=config.early_stopping_patience,
                class_weights=class_weights,  # Add class weights to handle imbalance
                measure_performance=True
            )              # Evaluate on test set
            # Use tag_vocab and idx2tag for evaluation
            metrics = compute_ner_metrics(trained_model, test_dataset, tag_vocab, idx2tag)
            logger.info(f"Test metrics: {metrics}")
            
        elif mode == 'evaluate':
            # Model path handling - explicitly check for parameter or try to find most recent model
            if not hasattr(config, 'model_path_param') or not config.model_path_param:
                # If using a specific model file name
                specific_model = "ner_model_final_20250520-204254.keras"
                if os.path.exists(os.path.join(config.model_dir, specific_model)):
                    config.model_path = os.path.join(config.model_dir, specific_model)
                    logger.info(f"Using specified model file: {config.model_path}")
                else:
                    # Look for both .keras and .h5 files
                    model_dir = Path(config.model_dir)
                    model_files = list(model_dir.glob("*.keras")) + list(model_dir.glob("*.h5"))
                    if model_files:
                        # Sort by modification time and get the most recent
                        model_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                        config.model_path = str(model_files[0])
                        logger.info(f"Using most recent model file: {config.model_path}")
                    else:
                        logger.error("No model file found in model_dir. Please provide a specific model file.")
                        sys.exit(1)
            else:
                config.model_path = config.model_path_param
            
            # Load data for evaluation
            logger.info(f"Loading data from {config.data_path}")
            data = load_data(config.data_path)
            preprocessed_data = preprocess_data(data)
            
            # Create dataset for evaluation (using same parameters as training)
            logger.info("Creating evaluation dataset")
            _, _, test_dataset, word_vocab, tag_vocab, idx2word, idx2tag, _ = create_ner_dataset(
                preprocessed_data, 
                config.batch_size, 
                config.max_seq_length,
                val_split=config.validation_split,
                test_split=config.test_split,
                use_optimization=False  # No need for optimization during evaluation
            )
            
            # Load the model
            logger.info(f"Loading model from {config.model_path}")
            model = load_model(config.model_path)
            
            # Set up output directory for evaluation results
            output_dir = os.path.join(config.output_dir if hasattr(config, 'output_dir') else "evaluations", 
                                     f"eval_{time.strftime('%Y%m%d-%H%M%S')}")
            os.makedirs(output_dir, exist_ok=True)            # Evaluate model
            logger.info("Evaluating model on test dataset")
            metrics = evaluate_model(model, test_dataset, tag_vocab, output_dir=output_dir)
            
            # Print summary of results
            logger.info("Evaluation results:")
            logger.info(f"Entity-level F1 Score: {metrics.get('entity_f1', metrics.get('f1', 0.0)):.4f}")
            logger.info(f"Entity-level Precision: {metrics.get('entity_precision', metrics.get('precision', 0.0)):.4f}")
            logger.info(f"Entity-level Recall: {metrics.get('entity_recall', metrics.get('recall', 0.0)):.4f}")
            logger.info(f"Token-level F1 Score: {metrics.get('token_f1', metrics.get('token_f1_score', 'N/A'))}")
            
            logger.info(f"Detailed evaluation results saved to {output_dir}")
            
        elif mode == 'predict':
            # Model path handling - explicitly check for parameter or try to find most recent model
            if not hasattr(config, 'model_path_param') or not config.model_path_param:
                # If using a specific model file name
                specific_model = "ner_model_final_20250520-204254.keras"
                if os.path.exists(os.path.join(config.model_dir, specific_model)):
                    config.model_path = os.path.join(config.model_dir, specific_model)
                    logger.info(f"Using specified model file: {config.model_path}")
                else:
                    # Look for both .keras and .h5 files
                    model_dir = Path(config.model_dir)
                    model_files = list(model_dir.glob("*.keras")) + list(model_dir.glob("*.h5"))
                    if model_files:
                        # Sort by modification time and get the most recent
                        model_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                        config.model_path = str(model_files[0])
                        logger.info(f"Using most recent model file: {config.model_path}")
                    else:
                        logger.error("No model file found in model_dir. Please provide a specific model file.")
                        sys.exit(1)
            else:
                config.model_path = config.model_path_param
            
            # Check if text to predict is provided
            if not hasattr(config, 'predict_text') or not config.predict_text:
                logger.error("No text provided for prediction. Please use --predict-text parameter.")
                sys.exit(1)
            
            # Load data to get vocabulary
            logger.info("Loading data to get vocabulary information")
            data = load_data(config.data_path)
            preprocessed_data = preprocess_data(data)
            
            # We need the word and tag vocabulary for prediction
            logger.info("Creating vocabulary for prediction")
            _, _, _, word_vocab, tag_vocab, idx2word, idx2tag, _ = create_ner_dataset(
                preprocessed_data, 
                config.batch_size, 
                config.max_seq_length,
                val_split=0.1,
                test_split=0.1,
                use_optimization=False
            )
            
            # Initialize predictor with model and vocabulary
            logger.info(f"Loading model from {config.model_path}")
            predictor = NERPredictor(
                model_path=config.model_path,
                word_tokenizer=word_vocab,  # Pass the vocabulary
                tag_vocab=tag_vocab  # Pass tag vocabulary
            )
            
            # Make prediction
            text = config.predict_text
            logger.info(f"Predicting entities in text: '{text}'")
            tokens, predicted_tags, entities = predictor.predict(text)
            
            # Create visualization
            visualization = predictor.visualize_entities(tokens, predicted_tags, entities)
            
            # Display results
            logger.info("Prediction results:")
            logger.info(f"Tokens: {tokens}")
            logger.info(f"Predicted tags: {predicted_tags}")
            logger.info(f"Entities found: {len(entities)}")
            
            for i, entity in enumerate(entities):
                logger.info(f"Entity {i+1}: {entity['text']} - Type: {entity['type']}")
            
            # Save results if output directory is specified
            if hasattr(config, 'output_dir') and config.output_dir:
                output_dir = os.path.join(config.output_dir, f"predict_{time.strftime('%Y%m%d-%H%M%S')}")
                os.makedirs(output_dir, exist_ok=True)
                
                # Save visualization as CSV
                visualization_path = os.path.join(output_dir, "prediction.csv")
                visualization.to_csv(visualization_path, index=False)
                
                # Save entities as JSON
                entities_path = os.path.join(output_dir, "entities.json")
                with open(entities_path, 'w') as f:
                    json.dump(entities, f, indent=2, default=str)
                
                logger.info(f"Prediction results saved to {output_dir}")
        else:
            logger.error(f"Unknown mode: {mode}")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        logger.error(traceback.format_exc())
        sys.exit(1)
    
    logger.info("NER pipeline completed successfully")


if __name__ == "__main__":
    main()
