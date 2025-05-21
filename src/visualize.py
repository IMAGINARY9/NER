"""
Visualize and analyze NER model predictions.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import tensorflow as tf
import logging
import json
import time
from tqdm import tqdm

# Set path to project root
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# Set tensorflow threading options early, before any TF operations
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

from src.data_processing.data_loader import load_data, preprocess_data
from src.data_processing.dataset import create_ner_dataset, create_tag_lookup_tables
from src.models.ner_model import build_model
from src.utils.metrics import compute_ner_metrics
from src.utils.model_loader import load_model_with_custom_objects
from src.config import Config

logger = logging.getLogger(__name__)


def setup_logging():
    """Set up logging configuration."""
    log_dir = os.path.join(project_root, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'visualization.log')),
            logging.StreamHandler()
        ]
    )


def configure_tensorflow_verbosity(verbose=False):
    """
    Configure TensorFlow to reduce or increase output verbosity.
    
    Args:
        verbose (bool): If True, show more output. If False, minimize output.
    """
    # Disable TensorFlow logging for less verbosity
    if not verbose:
        # Set environment variable - use level 3 for maximum suppression
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0=all, 1=INFO, 2=WARNING, 3=ERROR
        
        # Configure TensorFlow logger
        tf_logger = tf.get_logger()
        tf_logger.setLevel(logging.ERROR)
        
        # Disable all TensorFlow logging systems
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
        
        # Disable progress bars in Keras
        try:
            import keras
            if hasattr(keras.utils, 'disable_progress_bar'):
                keras.utils.disable_progress_bar()
            elif hasattr(keras.utils, 'set_verbosity'):
                keras.utils.set_verbosity(0)  # Alternative approach
                
            # Also suppress auto-logging in keras callbacks
            if hasattr(keras, 'callbacks'):
                if hasattr(keras.callbacks, 'ProgbarLogger'):
                    keras.callbacks.ProgbarLogger.update_stateful_metrics = lambda *args, **kwargs: None
        except Exception as e:
            logger.debug(f"Non-critical error configuring Keras verbosity: {e}")
            
        # Disable TF auto-logging features
        if hasattr(tf, 'autograph'):
            tf.autograph.set_verbosity(0)
            
        # Set global TensorFlow execution mode to be less verbose
        try:
            tf.keras.backend.set_floatx('float32')  # This can reduce some deprecation warnings
        except:
            pass
    else:
        # Enable verbose logging
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
        tf_logger = tf.get_logger()
        tf_logger.setLevel(logging.INFO)
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
        
        # Re-enable progress bars if previously disabled
        try:
            import keras
            if hasattr(keras.utils, 'enable_progress_bar'):
                keras.utils.enable_progress_bar()
            elif hasattr(keras.utils, 'set_verbosity'):
                keras.utils.set_verbosity(1)  # Default verbosity
        except:
            pass


def visualize_confusion_matrix(confusion_matrix, labels, title='Confusion Matrix', figsize=(12, 10), save_path=None):
    """
    Visualize a confusion matrix for the NER tags.
    """
    fig = plt.figure(figsize=figsize)
    
    # Create a normalized confusion matrix to handle highly imbalanced data
    norm_cm = confusion_matrix.astype('float') / (confusion_matrix.sum(axis=1)[:, np.newaxis] + 1e-10)
    
    # Use a logarithmic color scale for better visualization of imbalanced data
    with np.errstate(divide='ignore'):
        log_cm = np.log10(confusion_matrix + 1)  # Add 1 to avoid log(0)
    
    # Create the heatmap with custom normalization
    sns.heatmap(log_cm, annot=confusion_matrix, fmt='d', 
                cmap='viridis', xticklabels=labels, yticklabels=labels, 
                cbar=True, linewidths=0.5, linecolor='black')
    
    plt.title(title)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Saved confusion matrix to {save_path}")
    
    plt.show()
    
    # Close the figure to prevent it from showing again later
    plt.close(fig)


def visualize_class_distribution(class_distribution, title='Class Distribution', figsize=(12, 8), save_path=None):
    """
    Visualize the distribution of entity classes.
    """
    fig = plt.figure(figsize=figsize)
    
    # Sort by frequency
    labels, values = zip(*sorted(class_distribution.items(), key=lambda x: x[1], reverse=True))
    
    # Define colors for the bars - use a sequential palette to better highlight differences
    colors = sns.color_palette("viridis", len(labels))

    # Create bar chart
    bar = plt.bar(labels, values, color=colors)
    
    # Add values on top of bars
    for rect in bar:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width()/2., height + 5,
                 f'{height:,}', ha='center', va='bottom')
    
    plt.title(title)
    plt.xlabel('Entity Classes')
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Saved class distribution to {save_path}")
    
    plt.show()
    
    # Close the figure to prevent it from appearing again later
    plt.close(fig)


def visualize_entity_metrics(entity_metrics, title='Entity-level Metrics', figsize=(14, 8), save_path=None):
    """
    Visualize precision, recall, and F1-score for each entity type.
    """
    # Check if there are any metrics to visualize
    if not entity_metrics:
        logger.warning("No entity metrics to visualize. Skipping entity metrics visualization.")
        return
    
    # Extract entity types and metrics
    entities = []
    precision = []
    recall = []
    f1_score = []
    support = []
    
    for entity, metrics in entity_metrics.items():
        entities.append(entity)
        
        # Handle the case where metrics might be a dict or a float value
        if isinstance(metrics, dict):
            precision.append(metrics.get('precision', 0))
            recall.append(metrics.get('recall', 0))
            f1_score.append(metrics.get('f1-score', 0))
            support.append(metrics.get('support', 0))
        else:
            # If metrics is a single value (float), use it for f1-score
            # and set other metrics to the same value
            precision.append(float(metrics))
            recall.append(float(metrics))
            f1_score.append(float(metrics))
            support.append(1)  # Default support value
            
    # Check if we have any entities to plot
    if not entities:
        logger.warning("No entities found in metrics. Skipping entity metrics visualization.")
        return
        
    # Sort by support (frequency)
    sorted_indices = np.argsort(support)[::-1]
    entities = [entities[i] for i in sorted_indices]
    precision = [precision[i] for i in sorted_indices]
    recall = [recall[i] for i in sorted_indices]
    f1_score = [f1_score[i] for i in sorted_indices]
    support = [support[i] for i in sorted_indices]
    
    # Create a single figure for the plot
    fig, ax1 = plt.subplots(figsize=figsize)
    ax2 = ax1.twinx()
    
    # Plot metrics on left y-axis
    x = np.arange(len(entities))
    width = 0.25
    
    ax1.bar(x - width, precision, width, label='Precision', color='skyblue')
    ax1.bar(x, recall, width, label='Recall', color='lightgreen')
    ax1.bar(x + width, f1_score, width, label='F1 Score', color='coral')
    ax1.set_ylim(0, 1)
    ax1.set_ylabel('Score')
    
    # Plot support on right y-axis with log scale to handle imbalanced data
    ax2.plot(x, support, 'o-', color='purple', label='Support')
    if max(support) / (min(support) + 1e-10) > 100:  # If highly imbalanced
        ax2.set_yscale('log')
    ax2.set_ylabel('Support (# samples)')
    
    # Set x-axis labels
    ax1.set_xticks(x)
    ax1.set_xticklabels(entities, rotation=45, ha='right')
    
    # Add legends
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    
    plt.title(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved entity metrics to {save_path}")
    
    plt.show()
    
    # Close the figure to ensure it doesn't appear again later
    plt.close(fig)


def visualize_example_predictions(model, test_dataset, idx2tag, num_examples=5, save_dir=None):
    """
    Visualize example predictions for a few samples from the test dataset.
    
    Args:
        model: Trained NER model
        test_dataset: Test dataset
        idx2tag: Mapping from tag indices to tag names
        num_examples: Number of examples to visualize
        save_dir: Directory to save visualizations (optional)
    """
    try:
        logger.info(f"Visualizing {num_examples} example predictions...")
        
        # Limit to num_examples
        example_dataset = test_dataset.take(num_examples)
        
        example_count = 0
        max_attempts = num_examples * 3  # Try more times than needed in case some examples fail
        attempts = 0
        
        for batch in example_dataset:
            try:
                # Handle different dataset structures - the create_ner_dataset function returns 
                # (features, labels, weights) tuples
                if isinstance(batch, tuple):
                    if len(batch) == 3:  # Handle (features, labels, weights)
                        features, labels, _ = batch
                    elif len(batch) == 2:  # Handle (features, labels)
                        features, labels = batch
                    else:
                        logger.warning(f"Unexpected batch format: tuple of length {len(batch)}. Skipping example.")
                        continue
                else:
                    logger.warning(f"Unexpected batch format: {type(batch)}. Skipping example.")
                    continue
                
                # Handle the case where there might be multiple examples in a batch
                batch_size = 1
                if isinstance(features, tf.Tensor):
                    batch_size = features.shape[0]
                elif isinstance(features, dict) and list(features.values())[0].shape:
                    batch_size = list(features.values())[0].shape[0]
                  # Make predictions with verbosity suppressed
                # If batch already has shape [batch_size, ...], don't expand
                predict_features = features
                
                # Suppress output during prediction
                import contextlib
                import io
                with contextlib.redirect_stdout(io.StringIO()):
                    predictions = model.predict(predict_features, verbose=0)
                
                # If predictions are probabilities, convert to label indices
                if len(predictions.shape) == 3:  # (batch_size, seq_length, num_classes)
                    predictions = np.argmax(predictions, axis=-1)
                
                # Process each example in the batch
                for i in range(batch_size):
                    if example_count >= num_examples:
                        break
                        
                    try:
                        # Get example data
                        if batch_size == 1:
                            example_features = features
                            example_labels = labels
                            example_preds = predictions[0] if len(predictions.shape) > 1 else predictions
                        else:
                            example_features = features[i] if not isinstance(features, dict) else {k: v[i] for k, v in features.items()}
                            example_labels = labels[i]
                            example_preds = predictions[i]
                        
                        # Create a mask to filter out padding
                        mask = example_labels.numpy() != 0 if hasattr(example_labels, 'numpy') else example_labels != 0
                        
                        # Convert label indices to tag names
                        true_tags = [idx2tag.get(int(label), "O") for label in example_labels[mask]]
                        pred_tags = [idx2tag.get(int(pred), "O") for pred in example_preds[mask]]
                        
                        print(f"\nExample {example_count + 1}:")
                        print("-" * 50)
                        print("Truth | Prediction")
                        print("-" * 50)
                        
                        for t_tag, p_tag in zip(true_tags, pred_tags):
                            match = "✓" if t_tag == p_tag else "✗"
                            print(f"{t_tag.ljust(10)} | {p_tag.ljust(10)} {match}")
                        
                        # Add overall accuracy for this example
                        correct = sum(1 for t, p in zip(true_tags, pred_tags) if t == p)
                        total = len(true_tags)
                        accuracy = correct / total if total > 0 else 0
                        print(f"Accuracy: {correct}/{total} ({accuracy:.1%})")
                        print("-" * 50)
                        
                        example_count += 1
                    except Exception as ex:
                        logger.warning(f"Error processing individual example: {ex}")
                        logger.debug(f"Exception details: {str(ex)}", exc_info=True)
                        continue
            except Exception as e:
                logger.warning(f"Error processing example batch: {e}")
                logger.debug(f"Exception details: {str(e)}", exc_info=True)
            
            attempts += 1
            if attempts >= max_attempts:
                logger.warning(f"Reached maximum number of attempts ({max_attempts}). Stopping example visualization.")
                break
        
        if example_count == 0:
            logger.warning("No examples could be successfully visualized.")
    except Exception as e:
        logger.error(f"Error in visualize_example_predictions: {e}")
        logger.exception("Traceback for visualization error:")


def main(model_path, data_path=None, output_dir=None, viz_type='all'):
    """
    Visualize and analyze an NER model.
    
    Args:
        model_path: Path to the trained model file
        data_path: Path to the test data (optional)
        output_dir: Output directory for visualizations
        viz_type: Type of visualization to generate ('all', 'confusion', 'distribution', 'metrics', 'examples')
    """
    # Load configuration
    config = Config()
    
    # Set default output directory if not provided
    if output_dir is None:
        output_dir = os.path.join(project_root, 'visualizations')
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Saving visualizations to {output_dir}")
    
    # Configure TensorFlow to reduce verbosity
    configure_tensorflow_verbosity(verbose=False)
    logger.info("TensorFlow configured for reduced verbosity")
    
    model = None
    try:
        if not os.path.exists(model_path):
            logger.error(f"Model path does not exist: {model_path}")
            sys.exit(1)
            
        logger.info(f"Attempting to load model from {model_path} using 'load_model_with_custom_objects'.")
        
        model = load_model_with_custom_objects(model_path) 
        
        if model is None:
            logger.error(f"Failed to load model from {model_path} using 'load_model_with_custom_objects'. All strategies failed.")
            sys.exit(1) # Exit if model_loader fails.
        
        logger.info("Model loaded successfully.")
        try:
            model.summary(print_fn=logger.info)
        except Exception as e:
            logger.warning(f"Could not print model summary: {e}")

    except Exception as e:
        logger.error(f"An unexpected error occurred during model loading: {str(e)}")
        # Log the full traceback for debugging
        logger.exception("Traceback for model loading error:")
        sys.exit(1)
    
    # Load and preprocess data if provided    
    if data_path:
        logger.info(f"Loading data from {data_path}")
        data = load_data(data_path)
        preprocessed_data = preprocess_data(data)
        logger.info("Data preprocessing complete")
        
        # Initialize tag vocabularies
        tag_vocab = {}
        idx2tag = {}
        
        # Set up TensorFlow for better performance
        try:
            tf.config.threading.set_inter_op_parallelism_threads(1)
            tf.config.threading.set_intra_op_parallelism_threads(1)
            tf.data.experimental.enable_debug_mode()  # Enable dataset debug mode
            logger.info("TensorFlow threading configuration applied")
        except Exception as e:
            logger.warning(f"Could not configure TensorFlow threading: {e}")
        try:
            # Attempt to get/create tag vocab and idx2tag
            # This part is crucial and depends on your project structure.
            # Option 1: Load from config if available
            if hasattr(config, 'tag_to_id') and hasattr(config, 'id_to_tag'):
                tag_vocab = config.tag_to_id
                idx2tag = config.id_to_tag
                if not tag_vocab or not idx2tag:
                    logger.warning("tag_to_id or id_to_tag from config is empty.")
            
            # Option 2: Create from data (if not available from config or needs to be dynamic)
            if not tag_vocab or not idx2tag:
                logger.info("Attempting to create tag lookup tables from data.")
                # Extract unique tags from preprocessed data
                unique_tags = pd.unique(preprocessed_data['Tag'].explode())
                tag_vocab = {tag: idx for idx, tag in enumerate(unique_tags)}
                created_idx2tag, created_tag_vocab = create_tag_lookup_tables(tag_vocab)
                tag_vocab = created_tag_vocab if not tag_vocab else tag_vocab
                idx2tag = created_idx2tag if not idx2tag else idx2tag

            if not tag_vocab or not idx2tag:
                logger.error("Failed to load or create tag_vocab/idx2tag. Cannot proceed with evaluation.")
                sys.exit(1)

            logger.info(f"Using tag_vocab with {len(tag_vocab)} tags.")                  
            # Create dataset for visualization/evaluation with optimizations
            logger.info("Creating evaluation dataset...")
            datasets = create_ner_dataset(
                preprocessed_data,
                batch_size=min(32, config.batch_size),  # Use smaller batch size for visualization
                max_seq_length=config.max_seq_length,
                val_split=0.01,  # Minimal validation split required by sklearn
                test_split=0.98,  # Use most data for testing
                use_optimization=True  # Enable dataset optimizations
            )
              # Extract test dataset and vocabularies
            if len(datasets) >= 8:  # Full return with char_vocab
                _, _, test_dataset, word_vocab, tag_vocab, idx2word, idx2tag, _ = datasets
            else:  # Basic return without char_vocab
                _, _, test_dataset, word_vocab, tag_vocab, idx2word, idx2tag, _ = datasets
              # Optimize dataset for evaluation
            if isinstance(test_dataset, tf.data.Dataset):
                test_dataset = (test_dataset
                    .prefetch(tf.data.AUTOTUNE)
                    .cache()  # Cache the dataset in memory
                )
                # We don't map here anymore since we'll handle the dataset structure directly in compute_ner_metrics
                logger.info("Dataset optimizations applied: prefetch, cache")
            
            # Override existing tag_vocab and idx2tag if we got them from dataset creation
            if tag_vocab and idx2tag:
                logger.info("Using tag vocabulary from dataset creation")
            
            # Count batches to estimate processing time
            try:
                # Don't enumerate the entire dataset here as it might be large
                # Just check a few batches to confirm it's working
                for _, batch in zip(range(3), test_dataset):
                    pass
                logger.info("Dataset validation: Successfully iterated through initial batches")
            except Exception as e:
                logger.warning(f"Dataset validation failed: {e}")
        except Exception as e:
            logger.error(f"Failed to create test dataset or tag lookup tables: {e}")
            logger.exception("Traceback for dataset creation error:")
            sys.exit(1)
        
        logger.info("Computing metrics with improved progress tracking...")
        # First apply the deeper TensorFlow verbosity configuration
        configure_tensorflow_verbosity(verbose=False)
        
        # Then pass improved parameters to metrics computation
        metrics = compute_ner_metrics(
            model, 
            test_dataset, 
            tag_vocab, 
            idx2tag, 
            detailed=True,
            batch_timeout=60,  # 60 second timeout per batch
            max_batches=50     # Reduce max batch count for faster results and less output
        )
        
        print("\\n===== Model Performance =====")
        print(f"Token-level F1 Score: {metrics.get('token_f1_score', 0.0):.4f}")
        print(f"Token-level Precision: {metrics.get('token_precision', 0.0):.4f}")
        print(f"Token-level Recall: {metrics.get('token_recall', 0.0):.4f}")
        
        if 'entity_f1_score' in metrics:
            print(f"Entity-level F1 Score (Macro Avg): {metrics.get('entity_f1_score', 0.0):.4f}")
            print(f"Entity-level Precision (Macro Avg): {metrics.get('entity_precision', 0.0):.4f}")
            print(f"Entity-level Recall (Macro Avg): {metrics.get('entity_recall', 0.0):.4f}")

        viz_dir = output_dir
          # Visualize in sequence to avoid plot overlapping issues
        
        if 'confusion_matrix' in metrics and (viz_type == 'all' or viz_type == 'confusion'):
            logger.info("Generating confusion matrix visualization...")
            # Use labels provided by compute_ner_metrics if available, otherwise fallback
            cm_labels = metrics.get('confusion_matrix_labels')
            if cm_labels:
                visualize_confusion_matrix(metrics['confusion_matrix'], labels=cm_labels, 
                                           title='Confusion Matrix', save_path=os.path.join(viz_dir, 'confusion_matrix.png'))
            elif idx2tag: # Fallback to generating labels from idx2tag if specific cm_labels are not available
                logger.warning("Using idx2tag for confusion matrix labels as 'confusion_matrix_labels' not found in metrics. This might not perfectly match the matrix structure if it includes an 'OTHER' category.")
                labels = [idx2tag.get(i, f"Tag_{i}") for i in range(len(idx2tag))] # Ensure this matches how cm was built if no specific labels
                visualize_confusion_matrix(metrics['confusion_matrix'], labels=labels, 
                                           title='Confusion Matrix (idx2tag labels)', save_path=os.path.join(viz_dir, 'confusion_matrix.png'))
            else:
                logger.warning("Neither 'confusion_matrix_labels' in metrics nor idx2tag is available. Cannot generate confusion matrix labels.")

        # Force matplotlib to clear the current figure
        plt.clf()
        plt.close('all')

        if 'class_distribution' in metrics and (viz_type == 'all' or viz_type == 'distribution'):
            logger.info("Generating class distribution visualization...")
            visualize_class_distribution(metrics['class_distribution'], 
                                         title='Class Distribution of True Tags', 
                                         save_path=os.path.join(viz_dir, 'class_distribution_true.png'))
        
        # Force matplotlib to clear the current figure
        plt.clf()
        plt.close('all')
        
        if 'entity_metrics' in metrics and (viz_type == 'all' or viz_type == 'metrics'):
            logger.info("Generating entity metrics visualization...")
            visualize_entity_metrics(metrics['entity_metrics'], 
                                     title='Entity-level Metrics (Precision, Recall, F1)', 
                                     save_path=os.path.join(viz_dir, 'entity_metrics.png'))
        
        if viz_type == 'all' or viz_type == 'examples':
            logger.info("Visualizing example predictions...")
            visualize_example_predictions(model, test_dataset, idx2tag, num_examples=5)
        
        metrics_path = os.path.join(viz_dir, 'model_metrics.json')
        serializable_metrics = {}
        for k, v in metrics.items():
            if isinstance(v, np.ndarray):
                serializable_metrics[k] = v.tolist()
            elif isinstance(v, (np.int32, np.int64, np.float32, np.float64)): # Added np.int32
                serializable_metrics[k] = v.item()
            elif isinstance(v, dict):
                serializable_inner_dict = {}
                for ik, iv in v.items():
                    if isinstance(iv, dict): # Handle nested dicts like in entity_metrics
                        serializable_inner_dict[ik] = {iik: (iiv.tolist() if isinstance(iiv, np.ndarray) else (iiv.item() if isinstance(iiv, (np.int32, np.int64, np.float32, np.float64)) else iiv)) for iik, iiv in iv.items()}
                    elif isinstance(iv, np.ndarray):
                        serializable_inner_dict[ik] = iv.tolist()
                    elif isinstance(iv, (np.int32, np.int64, np.float32, np.float64)):
                        serializable_inner_dict[ik] = iv.item()
                    else:
                        serializable_inner_dict[ik] = iv
                serializable_metrics[k] = serializable_inner_dict
            else:
                serializable_metrics[k] = v
        
        with open(metrics_path, 'w') as f:
            json.dump(serializable_metrics, f, indent=4)
        logger.info(f"Saved metrics to {metrics_path}")
        
        summary_path = os.path.join(viz_dir, 'model_summary.txt')
        with open(summary_path, 'w') as f:
            f.write("Model Performance Summary:\\n")
            f.write(f"Token-level F1 Score: {metrics.get('token_f1_score', 'N/A'):.4f}\\n")
            f.write(f"Token-level Precision: {metrics.get('token_precision', 'N/A'):.4f}\\n")
            f.write(f"Token-level Recall: {metrics.get('token_recall', 'N/A'):.4f}\\n")
            if 'entity_f1_score' in metrics:
                f.write(f"Entity-level F1 Score (Macro Avg): {metrics.get('entity_f1_score', 'N/A'):.4f}\\n")
                f.write(f"Entity-level Precision (Macro Avg): {metrics.get('entity_precision', 'N/A'):.4f}\\n")
                f.write(f"Entity-level Recall (Macro Avg): {metrics.get('entity_recall', 'N/A'):.4f}\\n")
            # Add more details if needed
            if 'entity_metrics' in metrics:
                f.write("\\nDetailed Entity Metrics:\\n")
                for entity, emetrics in metrics['entity_metrics'].items():
                    # Check if emetrics is a dictionary or a direct float value
                    if isinstance(emetrics, dict):
                        f.write(f"  {entity}: P={emetrics.get('precision',0):.4f}, R={emetrics.get('recall',0):.4f}, F1={emetrics.get('f1-score',0):.4f}, Support={emetrics.get('support',0)}\\n")
                    else:
                        # Handle the case where a metric might be directly a float value
                        f.write(f"  {entity}: Value={emetrics:.4f}\\n")
        logger.info(f"Saved summary to {summary_path}")
    
    else: # if not data_path
        logger.info("No data path provided. Model loaded. To evaluate and visualize metrics, provide a data path.")
        try:
            plot_path = os.path.join(output_dir, "model_architecture.png")
            # Ensure tf.keras.utils.plot_model is available
            if hasattr(tf.keras.utils, 'plot_model'):
                tf.keras.utils.plot_model(model, to_file=plot_path, show_shapes=True, show_layer_names=True, expand_nested=True)
                logger.info(f"Model architecture plot saved to {plot_path}")
            else:
                logger.warning("tf.keras.utils.plot_model not available in this TensorFlow version.")
        except Exception as e:
            logger.warning(f"Could not plot model architecture: {e}")
            logger.exception("Traceback for plotting error:")

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description='Visualize NER model performance')
    parser.add_argument('--model-path', type=str, required=True, 
                        help='Path to the trained model file')
    parser.add_argument('--data-path', type=str, default=None,
                       help='Path to the test data (optional)')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory for visualizations (default is ./visualizations)')
    parser.add_argument('--type', type=str, default='all',
                       choices=['all', 'confusion', 'distribution', 'metrics', 'examples'],
                       help='Type of visualization to generate')
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging()
      # Run main function
    main(args.model_path, args.data_path, args.output_dir, args.type)
