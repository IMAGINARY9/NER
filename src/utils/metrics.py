"""
Evaluation metrics for Named Entity Recognition.
"""

import numpy as np
import tensorflow as tf
import logging
import time  # Import time module for timeout functionality
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
import pandas as pd

logger = logging.getLogger(__name__)


def compute_ner_metrics(model, test_dataset, tag_vocab, idx2tag=None, detailed=True, batch_timeout=30, max_batches=None):
    """
    Compute metrics for NER model with improved progress tracking and error handling.
    
    Args:
        model: Trained model to evaluate
        test_dataset: Test dataset
        tag_vocab: Tag vocabulary
        idx2tag: Optional pre-built index to tag mapping
        detailed: Whether to compute detailed metrics including per-entity stats
        batch_timeout: Timeout in seconds for each batch prediction (default: 30)
        max_batches: Maximum number of batches to process (None for all batches)
        
    Returns:
        dict: Dictionary of metrics
    """
    logger.info("Computing NER metrics...")
    
    # Count total batches for progress tracking
    try:
        # Temporarily convert to list first to get count without consuming the dataset
        if max_batches is None:
            try:
                # Try to count without converting to list if cardinality is known
                cardinality = tf.data.experimental.cardinality(test_dataset).numpy()
                if cardinality > 0:
                    total_batches = cardinality
                else:
                    # Fall back to manual counting with a limit to avoid excessive iteration
                    max_count = 1000  # Limit to avoid excessive counting
                    total_batches = sum(1 for _ in test_dataset.take(max_count))
                    if total_batches == max_count:
                        logger.warning(f"Dataset has at least {max_count} batches. Not counting further to avoid performance issues.")
                        total_batches = max_count
            except Exception:
                # Fall back to using a reasonable default
                total_batches = 100
                logger.warning(f"Could not determine total batches. Using default: {total_batches}")
        else:
            total_batches = max_batches
            
        logger.info(f"Starting evaluation of {total_batches} batches...")
    except Exception as e:
        logger.warning(f"Could not determine total number of batches: {e}")
        total_batches = 100  # Use a reasonable default
        
    # Initialize metrics dictionary early to handle early returns
    metrics = {
        'token_f1_score': 0.0,
        'token_precision': 0.0,
        'token_recall': 0.0,
        'token_report': {},
        'entity_metrics': {},
        'errors': []  # Track any errors that occur during processing
    }
    
    # Get the reverse mapping from index to tag
    if idx2tag is None:
        idx2tag = {idx: tag for tag, idx in tag_vocab.items()}
        idx2tag[0] = 'PAD'  # Add padding tag
    
    # Lists to store true and predicted labels
    y_true_flat = []
    y_pred_flat = []
    true_sequences = []
    pred_sequences = []
    
    # Counters for tracking progress
    processed_sequences = 0
    valid_sequences = 0
    
    # Add overall timeout for the entire evaluation process
    start_time = time.time()
    overall_timeout = batch_timeout * (total_batches or 100)  # Use an estimate if total_batches is unknown
      # Create a progress bar for overall processing, but print it less frequently
    progress_interval = max(1, total_batches // 10)
    last_progress_time = start_time
    progress_update_seconds = 10  # Update progress display only every 10 seconds (reduced frequency)
    
    # Track when we last printed a progress message to completely avoid floods
    last_print_time = start_time
    min_print_interval = 2.0  # Minimum seconds between any progress prints
    
    # Print initial progress message
    print(f"Starting metrics computation for {total_batches} batches...")
    
    # Process each batch
    for batch_idx, batch in enumerate(test_dataset):
        # Check if we've reached the max_batches limit
        if max_batches is not None and batch_idx >= max_batches:
            logger.info(f"Reached maximum batch limit ({max_batches}). Stopping evaluation.")
            break
            
        # Check for overall timeout
        if time.time() - start_time > overall_timeout:
            logger.warning(f"Evaluation exceeded overall timeout of {overall_timeout} seconds. Stopping early.")
            metrics['errors'].append("Evaluation stopped early due to overall timeout")
            break
          # Update progress at reasonable intervals to avoid flooding the console
        current_time = time.time()
        should_update = False
        
        # Determine if we should update progress based on several conditions
        if batch_idx == 0 or batch_idx == total_batches - 1:
            # Always update for first and last batch
            should_update = True
        elif batch_idx % progress_interval == 0:
            # Update on major interval points
            should_update = True
        elif current_time - last_progress_time >= progress_update_seconds:
            # Update after specified time interval has passed
            should_update = True
        
        # Only actually print if we haven't printed too recently
        if should_update and (current_time - last_print_time >= min_print_interval):
            progress_percent = min(100, (batch_idx + 1) / total_batches * 100)
            
            # Use \r to overwrite the line instead of adding new lines
            print(f"Computing metrics: {batch_idx+1}/{total_batches} ({progress_percent:.1f}%) - "
                  f"{valid_sequences}/{processed_sequences} valid sequences", 
                  end='\r')
                  
            # Update both timestamps
            last_progress_time = current_time
            last_print_time = current_time
        
        try:
            # Split batch into features and labels
            if isinstance(batch, (tuple, list)):
                if len(batch) >= 2:
                    X_batch, y_batch = batch[0], batch[1]
                else:
                    raise ValueError(f"Batch {batch_idx} has incorrect format")
            else:
                raise ValueError(f"Batch {batch_idx} is not a tuple/list")

            # Predict with timeout
            try:
                from concurrent.futures import ThreadPoolExecutor, TimeoutError
                
                # Suppress progress bar output by temporarily patching predict
                original_predict = model.predict
                  # Define a wrapper function that doesn't show progress bar for each batch
                def predict_without_verbose(x):
                    # Use the model's predict method with verbose=0 to suppress progress bar
                    if 'verbose' in original_predict.__code__.co_varnames:
                        return original_predict(x, verbose=0)
                    else:
                        # More robust approach to suppress all output during prediction
                        import contextlib
                        import io
                        import os
                        import tensorflow as tf
                        
                        # Store original env variable
                        original_tf_cpp_min_log_level = os.environ.get('TF_CPP_MIN_LOG_LEVEL', None)
                        
                        # Set maximum TF log level to suppress most output
                        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
                        
                        # Store original keras verbosity setting
                        try:
                            import keras
                            original_keras_verbosity = keras.utils.get_verbosity()
                            keras.utils.set_verbosity(0)
                        except (ImportError, AttributeError):
                            original_keras_verbosity = None
                            
                        # Redirect stdout during prediction
                        with contextlib.redirect_stdout(io.StringIO()):
                            try:
                                result = original_predict(x)
                            finally:
                                # Restore original settings
                                if original_tf_cpp_min_log_level is not None:
                                    os.environ['TF_CPP_MIN_LOG_LEVEL'] = original_tf_cpp_min_log_level
                                if original_keras_verbosity is not None:
                                    try:
                                        keras.utils.set_verbosity(original_keras_verbosity)
                                    except:
                                        pass
                                        
                        return result
                  # Use our modified predict function with timeout
                with ThreadPoolExecutor(max_workers=1) as executor:
                    # Set up silent prediction with timeout
                    future = executor.submit(predict_without_verbose, X_batch)
                    
                    # Extra verbosity control during prediction wait
                    current_time = time.time()
                    wait_message_shown = False
                    
                    # Wait for result with periodic checking (instead of just blocking)
                    while not future.done():
                        # Show a waiting message only once if prediction is taking a while
                        if not wait_message_shown and time.time() - current_time > 5:
                            print(f"Waiting for batch {batch_idx+1} prediction...", end="\r")
                            wait_message_shown = True
                        
                        # Sleep briefly to avoid CPU spinning
                        time.sleep(0.1)
                        
                        # Check for timeout
                        if time.time() - current_time > batch_timeout:
                            raise TimeoutError(f"Prediction timed out after {batch_timeout} seconds")
                    
                    # Get the result (should be immediate since we've confirmed it's done)
                    y_pred = future.result(timeout=1)
                    
                    # Clear the waiting message if shown
                    if wait_message_shown:
                        print(" " * 50, end="\r")  # Clear the waiting message line
                    
            except TimeoutError:
                error_msg = f"Batch {batch_idx+1} prediction timed out after {batch_timeout} seconds"
                logger.error(error_msg)
                metrics['errors'].append(error_msg)
                continue
            except Exception as predict_error:
                error_msg = f"Error predicting batch {batch_idx}: {predict_error}"
                logger.error(error_msg)
                metrics['errors'].append(error_msg)
                continue

            # Convert predictions to class indices if necessary
            if len(y_pred.shape) == 3:
                y_pred = np.argmax(y_pred, axis=-1)
            
            # Convert tensors to numpy if needed
            y_batch_np = y_batch.numpy() if hasattr(y_batch, 'numpy') else y_batch
            
            # Process each sequence in the batch
            for i in range(len(X_batch)):
                try:
                    # Get sequence mask
                    mask = None
                    if isinstance(X_batch, dict):
                        attention_mask = X_batch.get('attention_mask')
                        if attention_mask is not None:
                            mask = attention_mask[i].numpy() if hasattr(attention_mask[i], 'numpy') else attention_mask[i]
                            mask = mask.astype(bool)
                    if mask is None:
                        # Try to determine mask from non-zero values in the labels
                        mask = y_batch_np[i] != 0  # Assuming 0 is padding
                    
                    # Apply mask to get actual sequence
                    true_seq = y_batch_np[i][mask]
                    pred_seq = y_pred[i][mask]
                    
                    # Convert to tag names
                    true_tags = [idx2tag.get(idx, 'O') for idx in true_seq]
                    pred_tags = [idx2tag.get(idx, 'O') for idx in pred_seq]
                    
                    # Extend flat lists
                    y_true_flat.extend(true_seq)
                    y_pred_flat.extend(pred_seq)
                    
                    # Add sequences for entity-level metrics
                    true_sequences.append(true_tags)
                    pred_sequences.append(pred_tags)
                    
                    valid_sequences += 1
                except Exception as seq_error:
                    error_msg = f"Error processing sequence {i} in batch {batch_idx}: {seq_error}"
                    logger.error(error_msg)
                    metrics['errors'].append(error_msg)
                    continue
                
                processed_sequences += 1
            
            # Log progress
            if total_batches is not None:
                # We're using the custom progress tracking above instead of logging each batch
                # Only log significant issues
                if len(metrics['errors']) > 0 and len(metrics['errors']) % 10 == 0:
                    logger.warning(f"Accumulated {len(metrics['errors'])} errors during processing")
        
        except Exception as batch_error:
            error_msg = f"Error processing batch {batch_idx}: {batch_error}"
            logger.error(error_msg)
            metrics['errors'].append(error_msg)
            continue
      # Print final progress with newline to clear the progress line
    print(f"\nMetrics computation complete: {batch_idx+1}/{total_batches} batches processed")
    print(f"Results: {valid_sequences}/{processed_sequences} valid sequences ({(valid_sequences/processed_sequences*100 if processed_sequences else 0):.1f}%)")
    
    logger.info(f"Processing complete. Processed {processed_sequences} sequences, found {valid_sequences} valid sequences")
    
    # Store original errors so we can preserve them
    original_errors = metrics.get('errors', [])
    
    # Early return if no valid sequences found
    if not y_true_flat or not y_pred_flat:
        logger.error("No valid sequences found for evaluation")
        return metrics
    
    # Convert indices to tag names and continue with existing metrics computation
    y_true_tags = [idx2tag.get(idx, 'O') for idx in y_true_flat]
    y_pred_tags = [idx2tag.get(idx, 'O') for idx in y_pred_flat]
    
    # Use seqeval for entity-level metrics if available
    has_seqeval = False
    try:
        from seqeval.metrics import classification_report as seq_classification_report
        from seqeval.metrics import f1_score as seq_f1_score
        from seqeval.metrics import precision_score as seq_precision_score
        from seqeval.metrics import recall_score as seq_recall_score
        
        # Validate sequences before computing metrics
        if not true_sequences or not pred_sequences:
            logger.error("No valid sequences found for evaluation")
            return metrics
            
        # Validate that we have proper list of lists format
        if not all(isinstance(seq, list) for seq in true_sequences + pred_sequences):
            logger.error("Invalid sequence format detected")
            return metrics
            
        # Check sequence alignment
        if len(true_sequences) != len(pred_sequences):
            logger.error(f"Sequence length mismatch: true={len(true_sequences)}, pred={len(pred_sequences)}")
            return metrics
            
        # Compute entity-level metrics
        entity_f1 = seq_f1_score(true_sequences, pred_sequences)
        entity_precision = seq_precision_score(true_sequences, pred_sequences)
        entity_recall = seq_recall_score(true_sequences, pred_sequences)
        entity_report = seq_classification_report(true_sequences, pred_sequences, output_dict=True)
        
        logger.info(f"Entity-level F1 Score: {entity_f1:.4f}")
        logger.info(f"Entity-level Precision: {entity_precision:.4f}")
        logger.info(f"Entity-level Recall: {entity_recall:.4f}")
        
        has_seqeval = True
    except ImportError:
        logger.warning("seqeval not available. Install with 'pip install seqeval' for entity-level metrics.")
        has_seqeval = False
    
    # Compute token-level metrics
    token_report = classification_report(
        y_true_tags, 
        y_pred_tags,
        output_dict=True
    )
    
    # Extract entity-specific metrics (ignore O and PAD tags)
    entity_metrics = {
        tag: token_report.get(tag, {}) 
        for tag in token_report 
        if tag not in ['O', 'PAD', 'micro avg', 'macro avg', 'weighted avg']
    }
    
    # Calculate token-level F1 score
    token_f1 = f1_score(y_true_tags, y_pred_tags, average='weighted')
    token_precision = precision_score(y_true_tags, y_pred_tags, average='weighted')
    token_recall = recall_score(y_true_tags, y_pred_tags, average='weighted')
    
    # Log token-level results
    logger.info(f"Token-level F1 Score: {token_f1:.4f}")
    logger.info(f"Token-level Precision: {token_precision:.4f}")
    logger.info(f"Token-level Recall: {token_recall:.4f}")
    
    # Create metrics dictionary with preserved errors
    new_metrics = {
        'token_f1_score': token_f1,
        'token_precision': token_precision,
        'token_recall': token_recall,
        'token_report': token_report,
        'entity_metrics': entity_metrics,
        'errors': original_errors  # Preserve the original errors
    }
    
    # Add entity-level metrics if available
    if has_seqeval:
        new_metrics.update({
            'entity_f1_score': entity_f1,
            'entity_precision': entity_precision,
            'entity_recall': entity_recall,
            'entity_report': entity_report
        })
    
    # Calculate confusion matrix for the most common tags
    from sklearn.metrics import confusion_matrix
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Calculate class distribution and confusion matrix if detailed=True
    if detailed:
        try:
            # Get the most common tags (excluding PAD)
            tag_counts = {}
            for tag in set(y_true_tags + y_pred_tags):
                if tag != 'PAD':
                    count = len([t for t in y_true_tags if t == tag])
                    tag_counts[tag] = count

            # Sort tags by frequency and get top 10
            top_tags = [tag for tag, _ in sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:10]]
            
            # Create confusion matrix for top tags
            y_true_top = [tag if tag in top_tags else 'OTHER' for tag in y_true_tags]
            y_pred_top = [tag if tag in top_tags else 'OTHER' for tag in y_pred_tags]
            
            cm = confusion_matrix(
                y_true_top,
                y_pred_top,
                labels=top_tags + (['OTHER'] if len(tag_counts) > 10 else [])
            )
            
            # Store confusion matrix and class distribution
            new_metrics['confusion_matrix'] = cm
            new_metrics['confusion_matrix_labels'] = top_tags + (['OTHER'] if len(tag_counts) > 10 else [])
            new_metrics['class_distribution'] = tag_counts
            
            # Generate a histogram of sequence lengths
            seq_lengths = [len(seq) for seq in true_sequences]
            new_metrics['sequence_length_stats'] = {
                'mean': np.mean(seq_lengths),
                'median': np.median(seq_lengths),
                'min': np.min(seq_lengths),
                'max': np.max(seq_lengths)
            }
            
            logger.info("Sequence length statistics:")
            logger.info(f"Mean: {new_metrics['sequence_length_stats']['mean']:.1f}")
            logger.info(f"Median: {new_metrics['sequence_length_stats']['median']}")
            logger.info(f"Range: {new_metrics['sequence_length_stats']['min']} - {new_metrics['sequence_length_stats']['max']}")
            
        except Exception as e:
            error_msg = f"Error computing detailed metrics: {e}"
            logger.warning(error_msg)
            new_metrics['errors'].append(error_msg)

    # Add summary of any errors encountered
    if new_metrics['errors']:
        logger.warning(f"Encountered {len(new_metrics['errors'])} errors during metrics computation")
        logger.debug("Error summary: " + "\n".join(new_metrics['errors']))

    logger.info("Metrics computation complete.")
    return new_metrics


def visualize_prediction(sentence, predicted_tags, true_tags=None):
    """
    Create a visualization of the predicted entities in a sentence.
    
    Args:
        sentence: List of tokens in the sentence
        predicted_tags: List of predicted tags for each token
        true_tags: List of true tags (optional)
        
    Returns:
        pd.DataFrame: DataFrame with visualization
    """
    # Create DataFrame for visualization
    data = {
        'Token': sentence,
        'Predicted': predicted_tags
    }
    
    if true_tags is not None:
        data['True'] = true_tags
        data['Match'] = [pred == true for pred, true in zip(predicted_tags, true_tags)]
    
    df = pd.DataFrame(data)
    return df
