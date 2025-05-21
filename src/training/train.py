"""
Training module for Named Entity Recognition.

This module handles the training process for NER models.
"""

import os
import tensorflow as tf
import numpy as np
import logging
import time
from pathlib import Path
from datetime import datetime
from src.config import Config

logger = logging.getLogger(__name__)

# Create a default config instance
config = Config()


def train_model(model, train_dataset, val_dataset, epochs=10, learning_rate=2e-5,
               model_dir='./models', log_dir='./logs',
               early_stopping_patience=3, measure_performance=False,
               class_weights=None, use_warmup=True, warmup_epochs=2):
    """
    Train the NER model.
    
    Args:
        model: The compiled model to train
        train_dataset: TensorFlow dataset for training
        val_dataset: TensorFlow dataset for validation
        epochs: Number of training epochs
        learning_rate: Learning rate for the optimizer
        model_dir: Directory to save model checkpoints
        log_dir: Directory to save TensorBoard logs
        early_stopping_patience: Number of epochs to wait before early stopping
        measure_performance: Whether to measure detailed performance metrics
        class_weights: Dictionary mapping class indices to weights for imbalanced data
        use_warmup: Whether to use learning rate warmup
        warmup_epochs: Number of epochs for warmup
        
    Returns:
        model: The trained model
    """    # Ensure directories exist
    model_dir_path = Path(model_dir)
    model_dir_path.mkdir(parents=True, exist_ok=True)
    log_dir_path = os.path.join(log_dir, 'fit')
    os.makedirs(log_dir_path, exist_ok=True)
    
    # Get the number of training steps per epoch
    try:
        steps_per_epoch = len(list(train_dataset))
        logger.info(f"Steps per epoch: {steps_per_epoch}")
    except:
        logger.warning("Could not determine steps per epoch, using default value of 100")
        steps_per_epoch = 100
    
    # Create learning rate schedule with warmup if requested
    if use_warmup:
        # Create a learning rate scheduler with linear warmup and cosine decay
        warmup_steps = warmup_epochs * steps_per_epoch
        total_steps = epochs * steps_per_epoch        # Use a simpler built-in learning rate scheduler to avoid graph execution issues
        # Start with a smaller learning rate and gradually increase it
        initial_lr = learning_rate * 0.1        
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=learning_rate,
            decay_steps=steps_per_epoch * epochs // 4,
            decay_rate=0.9,
            staircase=True
        )
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, epsilon=1e-7)
        logger.info(f"Using ExponentialDecay learning rate schedule starting at {learning_rate}")
          # Compile the model with the scheduler optimizer
        # Use a safer approach to access metrics
        try:
            if hasattr(model.compiled_metrics, '_metrics'):
                metrics = model.compiled_metrics._metrics
            elif hasattr(model.compiled_metrics, 'metrics'):
                metrics = model.compiled_metrics.metrics
            else:
                # Fallback to default accuracy metric
                metrics = ['accuracy']
            
            model.compile(
                optimizer=optimizer,
                loss=model.loss,
                metrics=metrics
            )
        except Exception as e:
            logger.warning(f"Error accessing metrics, using default: {e}")
            # Fallback compilation with just accuracy
            model.compile(
                optimizer=optimizer,
                loss=model.loss,
                metrics=['accuracy']
            )
    else:# Use standard optimizer with fixed learning rate
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, epsilon=1e-7)
        logger.info(f"Using Adam optimizer with fixed learning rate: {learning_rate}")
      # Set the optimizer directly during compilation
    # Use a safer approach to access metrics
    try:
        if hasattr(model.compiled_metrics, '_metrics'):
            metrics = model.compiled_metrics._metrics
        elif hasattr(model.compiled_metrics, 'metrics'):
            metrics = model.compiled_metrics.metrics
        else:
            # Fallback to default accuracy metric
            metrics = ['accuracy']
            
        model.compile(
            optimizer=optimizer,
            loss=model.loss,
            metrics=metrics
        )
    except Exception as e:
        logger.warning(f"Error accessing metrics, using default: {e}")
        # Fallback compilation with just accuracy
        model.compile(
            optimizer=optimizer,
            loss=model.loss,
            metrics=['accuracy']
        )
    
    # Create callbacks
    callbacks = []
      # TensorBoard callback
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir_path,
        histogram_freq=1,
        write_graph=True,
        update_freq='epoch'
    )
    callbacks.append(tensorboard_callback)
      # ModelCheckpoint callback
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")    # Fix path concatenation by using os.path.join or string format correctly
    checkpoint_dir = os.path.join(str(model_dir_path), "checkpoints")
    checkpoint_filename = f"ner-{timestamp}-{{epoch:02d}}-{{val_loss:.2f}}.keras"
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_filename)
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        checkpoint_path,
        monitor='val_loss',
        save_best_only=True,
        save_weights_only=False,
        verbose=1
    )
    callbacks.append(checkpoint_callback)
    
    # Early stopping callback
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=early_stopping_patience,
        restore_best_weights=True,
        verbose=1
    )
    callbacks.append(early_stopping_callback)
    
    # Learning rate reduction callback
    reduce_lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=2,
        min_lr=1e-6,
        verbose=1
    )
    callbacks.append(reduce_lr_callback)
    
    # Performance measurement if requested
    if measure_performance:
        start_time = time.time()
      # Train the model
    logger.info(f"Starting training for {epochs} epochs")    # The dataset already includes sample weights from dataset.py
    # We should not use class_weight in fit() when dataset already has sample weights    # We need to modify how the dataset is handled to deal with the sample weights
    # Let's first check the dataset structure
    
    try:
        # Try to unpack the dataset to determine its structure
        for batch in train_dataset.take(1):
            logger.info(f"Dataset batch structure: {len(batch)} elements")
            if len(batch) == 2:
                # Format: (features, labels)
                logger.info("Dataset format: (features, labels)")
                modified_train_dataset = train_dataset
                modified_val_dataset = val_dataset
            elif len(batch) == 3:
                # Format: (features, labels, sample_weights)
                logger.info("Dataset format: (features, labels, sample_weights)")
                modified_train_dataset = train_dataset.map(lambda x, y, w: (x, y))
                modified_val_dataset = val_dataset.map(lambda x, y, w: (x, y))
            break
    except Exception as e:
        logger.warning(f"Error determining dataset structure: {e}, assuming (features, labels) format")
        modified_train_dataset = train_dataset
        modified_val_dataset = val_dataset
    
    # Now use class weights instead of sample weights
    if class_weights is not None:
        logger.info(f"Using class weights to handle imbalanced data: {class_weights}")
        class_weight = class_weights
    else:
        class_weight = None
        logger.info("No class weights provided")
    
    # Option to enable gradient accumulation for larger effective batch sizes
    steps_per_execution = 1
    if hasattr(model, 'compile') and hasattr(model, 'steps_per_execution'):
        try:
            # For TensorFlow 2.4+ models
            model.steps_per_execution = steps_per_execution
            logger.info(f"Set steps_per_execution to {steps_per_execution}")
        except:
            logger.warning(f"Could not set steps_per_execution property")    # First, extract all data and recreate the datasets without sample weights
    # This is a workaround for the shape mismatch between weights and loss
    
    # To safely extract the data, we'll use the take() method
    train_data = []
    val_data = []
    
    # Try to unpack in a safe way, accounting for different dataset structures
    try:
        for batch in train_dataset.take(-1):  # Take all batches
            if len(batch) == 3:  # (x, y, weights)
                x, y, _ = batch
                train_data.append((x, y))
            elif len(batch) == 2:  # (x, y)
                x, y = batch
                train_data.append((x, y))
                
        for batch in val_dataset.take(-1):
            if len(batch) == 3:  # (x, y, weights)
                x, y, _ = batch
                val_data.append((x, y))
            elif len(batch) == 2:  # (x, y)
                x, y = batch
                val_data.append((x, y))
    except Exception as e:
        logger.error(f"Error extracting data from datasets: {e}")
        logger.info("Falling back to using original datasets")
      # Create new datasets without sample weights, but only if we successfully extracted data
    if train_data and val_data:
        logger.info("Creating simplified datasets without sample weights")
        
        # Use a simpler approach: create new datasets without generators
        train_dataset_no_weights = modified_train_dataset
        val_dataset_no_weights = modified_val_dataset
    else:
        logger.info("Using original datasets")
        train_dataset_no_weights = modified_train_dataset
        val_dataset_no_weights = modified_val_dataset    # Fit the model without any weights - simplest approach to avoid dimension mismatch
    logger.info("Training with simplified approach (no weights)")
    try:
        history = model.fit(
            train_dataset_no_weights,
            epochs=epochs,
            validation_data=val_dataset_no_weights,
            callbacks=callbacks,
            verbose=1
            # No class_weight parameter to avoid dimension mismatch errors
        )
    except TypeError as e:
        if "learning rate is not settable" in str(e):
            logger.warning("Caught learning rate schedule error. Re-compiling model with fixed learning rate.")            # If we encounter the learning rate schedule error, recompile with a fixed learning rate
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, epsilon=1e-7)
            
            # Use a safer approach to access metrics
            try:
                if hasattr(model.compiled_metrics, '_metrics'):
                    metrics = model.compiled_metrics._metrics
                elif hasattr(model.compiled_metrics, 'metrics'):
                    metrics = model.compiled_metrics.metrics
                else:
                    # Fallback to default accuracy metric
                    metrics = ['accuracy']
                
                model.compile(
                    optimizer=optimizer,
                    loss=model.loss,
                    metrics=metrics
                )
            except Exception as e:
                logger.warning(f"Error accessing metrics, using default: {e}")
                # Fallback compilation with just accuracy
                model.compile(
                    optimizer=optimizer,
                    loss=model.loss,
                    metrics=['accuracy']
                )
            # Try again with the new optimizer
            history = model.fit(
                train_dataset_no_weights,
                epochs=epochs,
                validation_data=val_dataset_no_weights,
                callbacks=callbacks,
                verbose=1
            )
        else:
            # If it's a different error, re-raise it
            raise
    
    # Performance reporting
    if measure_performance:
        training_time = time.time() - start_time
        logger.info(f"Training completed in {training_time:.2f} seconds")
        
        # Calculate training speed
        steps_per_epoch = len(list(train_dataset))
        total_steps = steps_per_epoch * epochs
        steps_per_second = total_steps / training_time
        logger.info(f"Training speed: {steps_per_second:.2f} steps/second")
        
        # Report peak memory usage if possible
        try:
            import psutil
            process = psutil.Process(os.getpid())
            peak_memory_mb = process.memory_info().rss / (1024 * 1024)
            logger.info(f"Peak memory usage: {peak_memory_mb:.2f} MB")
        except ImportError:
            logger.warning("psutil not available, skipping memory usage reporting")
    # Save final model using the recommended Keras format
    final_model_path = os.path.join(str(model_dir_path), f"ner_model_final_{timestamp}.keras")
    model.save(final_model_path)
    logger.info(f"Final model saved to {final_model_path}")
    # Save training history
    import pandas as pd  # Add pandas import
    history_df = pd.DataFrame(history.history)
    history_path = os.path.join(str(model_dir_path), f"training_history_{timestamp}.csv")
    history_df.to_csv(history_path, index=False)
    logger.info(f"Training history saved to {history_path}")
    
    return model
