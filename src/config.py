import os
import sys
import platform
import argparse
from typing import Dict, Any, Optional

class Config:
    def __init__(self, args: Optional[Dict[str, Any]] = None):
        # Get project root directory
        self.PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # Data paths - use os.path.join for cross-platform compatibility
        self.data_path = os.path.join(self.PROJECT_ROOT, 'data', 'ner.csv')
        self.cache_dir = os.path.join(self.PROJECT_ROOT, 'cache')
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Model paths
        self.model_dir = os.path.join(self.PROJECT_ROOT, 'models')
        self.model_path = os.path.join(self.model_dir, 'best_model.h5')
        self.checkpoint_path = os.path.join(self.model_dir, 'checkpoints')
        
        # Ensure model directory exists
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.checkpoint_path, exist_ok=True)
        
        # Training hyperparameters
        self.batch_size = 32
        self.epochs = 10
        self.learning_rate = 2e-5
        self.dropout_rate = 0.1
        self.recurrent_dropout_rate = 0.1
        self.max_seq_length = 128
        self.validation_split = 0.1
        self.test_split = 0.1
        self.early_stopping_patience = 3
          # Model configuration
        self.model_type = 'bilstm-crf'
        self.embedding_dim = 100
        self.hidden_dim = 128
        self.attention_heads = 8
        self.use_char_features = True
        self.char_embedding_dim = 25
        self.char_hidden_dim = 25
        self.max_char_length = 20
        self.use_pretrained_embeddings = False
        self.pretrained_embeddings_path = None
        
        # Optimization settings
        self.use_xla = True
        self.use_mixed_precision = True
        self.use_dataset_optimization = True
        self.use_gradient_accumulation = False
        self.gradient_accumulation_steps = 2
        self.prefetch_buffer_size = -1  # Use AUTOTUNE in TensorFlow
        self.optimized_training = True
        
        # Logging and output
        self.log_dir = os.path.join(self.PROJECT_ROOT, 'logs', 'fit')
        os.makedirs(self.log_dir, exist_ok=True)
        self.tensorboard_logs = True
        self.save_model_summary = True
        self.verbose = 1  # 0 = silent, 1 = progress bar, 2 = one line per epoch
        
        # GPU settings
        self.use_gpu = True
        self.memory_growth = True
        self.gpu_memory_limit = None  # None means no limit, otherwise in MB
        
        # Windows-specific settings
        self.is_windows = platform.system() == 'Windows'
        self.use_directml = False  # For TensorFlow-DirectML on Windows
        
        # Override with command line arguments if provided
        if args:
            for key, value in args.items():
                if value is not None:
                    setattr(self, key, value)

    @classmethod
    def from_args(cls):
        """Create a Config instance from command line arguments"""
        parser = argparse.ArgumentParser(description='Named Entity Recognition Training')
        
        # Data and model arguments
        parser.add_argument('--data-path', type=str, help='Path to the NER dataset')
        parser.add_argument('--model-path', type=str, help='Path to save the model')
        parser.add_argument('--model-type', type=str, default='bilstm-crf',
                          choices=['bilstm', 'bilstm-crf', 'transformer'],
                          help='Type of NER model to use')
        
        # Training hyperparameters
        parser.add_argument('--max-seq-length', type=int, help='Maximum sequence length')
        parser.add_argument('--batch-size', type=int, help='Batch size for training')
        parser.add_argument('--epochs', type=int, help='Number of training epochs')
        parser.add_argument('--learning-rate', type=float, help='Learning rate')        
        parser.add_argument('--embedding-dim', type=int, help='Dimension of word embeddings')
        parser.add_argument('--hidden-dim', type=int, help='Dimension of hidden layers')
        parser.add_argument('--dropout-rate', type=float, help='Dropout rate')
        
        # Character features
        parser.add_argument('--use-char-features', action='store_true', 
                          help='Use character-level features')
        parser.add_argument('--no-char-features', action='store_true', 
                          help='Disable character-level features')
        parser.add_argument('--char-embedding-dim', type=int, 
                          help='Dimension of character embeddings')
        parser.add_argument('--char-hidden-dim', type=int, 
                          help='Dimension of character hidden layer')
        parser.add_argument('--max-char-length', type=int,
                          help='Maximum number of characters per word')
        
        # Embeddings features
        parser.add_argument('--use-pretrained-embeddings', action='store_true',
                          help='Use pre-trained word embeddings (GloVe)')
        parser.add_argument('--no-pretrained-embeddings', action='store_true',
                          help='Disable pre-trained word embeddings')
        
        # Optimization flags
        parser.add_argument('--no-optimizations', action='store_true',
                          help='Disable all optimizations')
        parser.add_argument('--no-xla', action='store_true',
                          help='Disable XLA optimization')
        parser.add_argument('--no-mixed-precision', action='store_true',
                          help='Disable mixed precision training')
        
        # GPU options
        parser.add_argument('--no-gpu', action='store_true', help='Disable GPU usage')
        parser.add_argument('--use-directml', action='store_true', help='Use DirectML for Windows GPU support')
        
        # Modes
        parser.add_argument('--mode', type=str, default='train',
                          choices=['train', 'evaluate', 'predict'],
                          help='Mode to run in')
        
        # Parse arguments and convert to dictionary
        args = parser.parse_args()
        args_dict = {k: v for k, v in vars(args).items() if v is not None}
          # Special handling for flags
        if args.no_gpu:
            args_dict['use_gpu'] = False
        if args.no_optimizations:
            args_dict['use_xla'] = False
            args_dict['use_mixed_precision'] = False
            args_dict['use_dataset_optimization'] = False
            args_dict['optimized_training'] = False
        if args.no_xla:
            args_dict['use_xla'] = False
        if args.no_mixed_precision:
            args_dict['use_mixed_precision'] = False
            
        # Handle character feature flags
        if args.use_char_features:
            args_dict['use_char_features'] = True
        if args.no_char_features:
            args_dict['use_char_features'] = False
            
        # Handle pretrained embeddings flags
        if args.use_pretrained_embeddings:
            args_dict['use_pretrained_embeddings'] = True
        if args.no_pretrained_embeddings:
            args_dict['use_pretrained_embeddings'] = False
        
        # Create config with the parsed arguments
        return cls(args_dict)
        
    def display(self):
        """Display configuration settings"""
        print("\n============ NER Configuration Settings ============")
        print(f"Platform: {platform.system()} {platform.release()}")
        print(f"Data Path: {self.data_path}")
        print(f"Model Path: {self.model_path}")
        print(f"Model Type: {self.model_type}")
        print(f"Max Sequence Length: {self.max_seq_length}")
        print(f"Batch Size: {self.batch_size}")
        print(f"Epochs: {self.epochs}")
        print(f"Learning Rate: {self.learning_rate}")
        print(f"Embedding Dimension: {self.embedding_dim}")
        print(f"Hidden Dimension: {self.hidden_dim}")
        print(f"Dropout Rate: {self.dropout_rate}")
        print(f"Log Directory: {self.log_dir}")
        print(f"TensorBoard Logs: {self.tensorboard_logs}")
        print(f"Use GPU: {self.use_gpu}")
        
        # Optimization settings
        print(f"Use XLA: {self.use_xla}")
        print(f"Use Mixed Precision: {self.use_mixed_precision}")
        print(f"Use Dataset Optimization: {self.use_dataset_optimization}")
        
        if self.is_windows and self.use_directml:
            print(f"Using TensorFlow-DirectML for Windows GPU support")
        
        print("===================================================")
