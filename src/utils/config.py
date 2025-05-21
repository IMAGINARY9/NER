"""
Configuration module for the Named Entity Recognition project.

Defines configuration parameters and environment setup.
"""

import os
import json
import logging
import tensorflow as tf

logger = logging.getLogger(__name__)

# Default configuration values
DEFAULT_CONFIG = {
    "train": {
        "batch_size": 32,
        "learning_rate": 2e-5,
        "epochs": 10,
        "early_stopping_patience": 3,
        "max_sequence_length": 128,
        "validation_split": 0.1,
        "test_split": 0.1
    },
    "model": {
        "model_type": "bilstm-crf",
        "embedding_dim": 100,
        "hidden_dim": 128,
        "dropout_rate": 0.1,
        "recurrent_dropout_rate": 0.1
    },
    "data": {
        "dataset_path": "./data/ner.csv",
        "cache_dir": "./cache"
    },
    "optimization": {
        "use_xla": True,
        "use_mixed_precision": True,
        "use_dataset_optimization": True,
        "use_gradient_accumulation": False,
        "gradient_accumulation_steps": 2,
        "prefetch_buffer_size": tf.data.experimental.AUTOTUNE
    },
    "logging": {
        "log_dir": "./logs",
        "tensorboard": True,
        "save_model_summary": True
    }
}


def parse_config(config_path=None):
    """
    Parse configuration from a JSON file or use defaults.
    
    Args:
        config_path: Path to the JSON configuration file
        
    Returns:
        dict: Configuration dictionary
    """
    config = DEFAULT_CONFIG.copy()
    
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                user_config = json.load(f)
            
            # Update default config with user config (deep merge)
            for section, section_config in user_config.items():
                if section in config:
                    config[section].update(section_config)
                else:
                    config[section] = section_config
                    
            logger.info(f"Loaded configuration from {config_path}")
        except Exception as e:
            logger.warning(f"Failed to load config from {config_path}: {e}")
            logger.info("Using default configuration")
    else:
        logger.info("No config file found, using default configuration")
    
    return config


def setup_environment(args, config):
    """
    Setup environment variables and TensorFlow configurations.
    
    Args:
        args: Command line arguments
        config: Configuration dictionary
    """
    # Check if we should disable all optimizations
    if args.no_optimizations:
        config["optimization"]["use_xla"] = False
        config["optimization"]["use_mixed_precision"] = False
        config["optimization"]["use_dataset_optimization"] = False
        config["optimization"]["use_gradient_accumulation"] = False
        logger.info("All optimizations have been disabled")
    
    # Override specific optimizations if specified in args
    if args.no_xla:
        config["optimization"]["use_xla"] = False
    if args.no_mixed_precision:
        config["optimization"]["use_mixed_precision"] = False
    
    # Set up TensorFlow optimizations
    if config["optimization"]["use_xla"]:
        tf.config.optimizer.set_jit(True)
        logger.info("XLA optimization enabled")
    
    if config["optimization"]["use_mixed_precision"]:
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
        logger.info(f"Mixed precision training enabled with policy: {policy.name}")
    
    # Setup environment variables for TensorFlow
    os.environ["TF_ENABLE_AUTO_MIXED_PRECISION"] = "1" if config["optimization"]["use_mixed_precision"] else "0"
    os.environ["TF_GPU_THREAD_MODE"] = "gpu_private"
    os.environ["TF_GPU_THREAD_COUNT"] = "2"
    os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices" if config["optimization"]["use_xla"] else ""
    
    # Log TensorFlow and GPU information
    gpus = tf.config.list_physical_devices('GPU')
    logger.info(f"TensorFlow version: {tf.__version__}")
    logger.info(f"GPUs available: {len(gpus)}")
    for i, gpu in enumerate(gpus):
        logger.info(f"  GPU {i}: {gpu.name}")
    
    # Memory growth for GPUs
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
            logger.info(f"Memory growth enabled for {gpu.name}")
        except RuntimeError as e:
            logger.warning(f"Error setting memory growth: {e}")
