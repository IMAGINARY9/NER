"""
Utility functions for loading saved models with custom objects.
"""

import os
import logging
import sys
import tensorflow as tf
from keras.models import load_model

logger = logging.getLogger(__name__)

def get_custom_objects():
    """
    Create a dictionary of custom TensorFlow objects needed for model loading.
    
    Returns:
        dict: Dictionary mapping object names to their implementations
    """
    custom_objects = {}
    
    # Add CRF implementation - fallback to custom implementation if TFA is not available
    try:
        # Try importing TensorFlow Addons in a way that suppresses warnings
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TF logging
            
            import tensorflow_addons as tfa
            custom_objects.update({
                'CRF': tfa.layers.CRF,
                'crf_loss': tfa.text.crf_loss,
                'crf_decode': tfa.text.crf_decode
            })
            logger.info("Added TensorFlow Addons objects to custom objects dictionary")
    except (ImportError, AttributeError) as e:
        logger.warning(f"TensorFlow Addons not available: {e}")
        logger.info("Using stub implementations for CRF functionality")
          # Create stub implementations for CRF
        class CRFStub(tf.keras.layers.Layer):
            def __init__(self, units=None, chain_initializer='orthogonal', use_boundary=True, use_kernel=True,
                         kernel_regularizer=None, chain_regularizer=None, boundary_regularizer=None,
                         **kwargs):
                super().__init__(**kwargs)
                self.supports_masking = True
                self.units = units
                self.chain_initializer = chain_initializer
                self.use_boundary = use_boundary
                self.use_kernel = use_kernel
                self.kernel_regularizer = kernel_regularizer
                self.chain_regularizer = chain_regularizer
                self.boundary_regularizer = boundary_regularizer
                self.transitions = None

            def build(self, input_shape):
                if self.units is None and len(input_shape) > 2:
                    self.units = input_shape[-1]
                self.transitions = self.add_weight(
                    shape=(self.units, self.units),
                    name="transitions",
                    initializer="zeros"
                )
                super().build(input_shape)

            def call(self, inputs, mask=None, training=None):
                return inputs

            def get_config(self):
                config = super().get_config()
                config.update({
                    'units': self.units,
                    'chain_initializer': self.chain_initializer,
                    'use_boundary': self.use_boundary,
                    'use_kernel': self.use_kernel,
                    'kernel_regularizer': self.kernel_regularizer,
                    'chain_regularizer': self.chain_regularizer,
                    'boundary_regularizer': self.boundary_regularizer
                })
                return config

            def compute_mask(self, inputs, mask=None):
                return mask

            def decode(self, potentials, sequence_length=None, mask=None):
                return potentials, None

        class DummyLoss(tf.keras.losses.Loss):
            def call(self, y_true, y_pred):
                return tf.reduce_mean(y_pred)

        class DummyAccuracy(tf.keras.metrics.Metric):
            def __init__(self, name='dummy_accuracy', **kwargs):
                super().__init__(name=name, **kwargs)
                self.accuracy = self.add_weight(name='accuracy', initializer='zeros')

            def update_state(self, y_true, y_pred, sample_weight=None):
                value = tf.reduce_mean(tf.keras.metrics.categorical_accuracy(y_true, y_pred))
                self.accuracy.assign(value)

            def result(self):
                return self.accuracy

            def reset_states(self):
                self.accuracy.assign(0.0)
        
        custom_objects.update({
            'CRF': CRFStub,
            'crf_loss': DummyLoss(),
            'get_loss': DummyLoss(),
            'get_accuracy': DummyAccuracy(),
        })
    
    return custom_objects

def load_model_with_custom_objects(model_path):
    """
    Load a Keras model with custom objects.
    
    Args:
        model_path: Path to the saved model
        
    Returns:
        model: Loaded Keras model, or None if loading failed
    """
    if not os.path.exists(model_path):
        logger.error(f"Model path does not exist: {model_path}")
        return None
    
    custom_objects = get_custom_objects()
    logger.info(f"Loading model with {len(custom_objects)} custom objects")
    
    # Suppress TensorFlow warnings during model loading
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    
    # Check TensorFlow version to handle compatibility issues
    tf_version = tf.__version__
    logger.info(f"TensorFlow version: {tf_version}")
    
    # For TensorFlow 2.13+ we need to set this environment variable to avoid
    # errors with TensorFlow Addons compatibility
    if tf_version.startswith('2.') and int(tf_version.split('.')[1]) >= 13:
        os.environ['TF_USE_LEGACY_KERAS'] = '1'
        logger.info("Set TF_USE_LEGACY_KERAS=1 for TensorFlow 2.13+ compatibility")
    
    model = None
    errors = []
    # Try all loading strategies (remove unsupported 'options' argument)
    loading_strategies = [
        # First attempt - standard loading with compile=False
        lambda: tf.keras.models.load_model(model_path, custom_objects=custom_objects, compile=False),

        # Second attempt - SavedModel directory
        lambda: tf.saved_model.load(model_path) if os.path.isdir(model_path) else None,

        # Third attempt - H5 file direct loading (redundant but explicit)
        lambda: tf.keras.models.load_model(
            model_path,
            custom_objects=custom_objects,
            compile=False
        ) if model_path.endswith('.h5') else None,

        # Fourth attempt - keras.saving.load_model (for newer Keras, if available)
        lambda: tf.keras.saving.load_model(
            model_path,
            custom_objects=custom_objects,
            compile=False
        ) if hasattr(tf.keras, 'saving') and hasattr(tf.keras.saving, 'load_model') else None,
    ]

    # Try all strategies
    for i, strategy in enumerate(loading_strategies):
        try:
            logger.info(f"Trying model loading strategy {i+1}...")
            result = strategy()
            if result is not None:
                model = result
                logger.info(f"Successfully loaded model with strategy {i+1}")
                break
        except Exception as e:
            error_msg = str(e)
            errors.append(f"Strategy {i+1} failed: {error_msg}")
            logger.warning(f"Loading strategy {i+1} failed: {error_msg}")

    # If all strategies failed, try a final approach for CRF models
    if model is None:
        try:
            logger.info("Trying final CRF-specific loading approach...")
            # For .h5 files, try to load with a completely custom approach
            if model_path.endswith('.h5'):
                with tf.keras.utils.custom_object_scope(custom_objects):
                    model = tf.keras.models.load_model(
                        model_path,
                        compile=False,
                    )
                logger.info("Successfully loaded model with CRF-specific approach")
            else:
                # For SavedModel format, try without 'options' (for compatibility)
                with tf.keras.utils.custom_object_scope(custom_objects):
                    model = tf.keras.models.load_model(
                        model_path,
                        compile=False
                    )
                logger.info("Successfully loaded SavedModel with CRF-specific approach")
        except Exception as e:
            error_msg = str(e)
            errors.append(f"Final CRF approach failed: {error_msg}")
            logger.warning(f"Final loading approach failed: {error_msg}")
    
    if model is None:
        logger.error("All loading attempts failed:")
        for error in errors:
            logger.error(f"  - {error}")
    else:
        # Verify the model was loaded correctly
        try:
            logger.info(f"Model has {model.count_params()} parameters")
            logger.info(f"Model input shape: {model.input_shape}")
            logger.info(f"Model output shape: {model.output_shape}")
        except Exception as e:
            logger.warning(f"Could not get complete model information: {e}")
    
    return model