# NER Project Technical Guide

## Table of Contents
1. [Project Architecture](#project-architecture)
2. [Core Components](#core-components)
3. [Development Workflow](#development-workflow)
4. [Technical Implementation](#technical-implementation)
5. [Data Flow](#data-flow)
6. [Testing and Evaluation](#testing-and-evaluation)

## Project Architecture

The project follows a modular architecture designed for scalability and maintainability:

### Source Code Organization (`src/`)

- **data_processing/**: Data pipeline implementations
  - Data loading and preprocessing
  - Dataset creation and batching
  - Tokenization and vocabulary management
  
- **models/**: Model architectures and components
  - Base model interfaces
  - BiLSTM and Transformer implementations
  - CRF layer integration
  
- **training/**: Training infrastructure
  - Training loops and optimization
  - Metrics calculation
  - Checkpoint management
  
- **utils/**: Shared utilities
  - Configuration management
  - Logging setup
  - Embedding utilities
  
- **services/**: API and deployment services
  - REST API implementation
  - Model serving utilities
  
### Core Components

### Data Flow Architecture

The project implements a streamlined data flow:

1. **Data Ingestion**
   ```
   Raw Text -> Tokenization -> Feature Extraction -> TensorFlow Dataset
   ```

2. **Training Pipeline**
   ```
   Dataset -> Batching -> Model Training -> Checkpoints -> Evaluation
   ```

3. **Inference Pipeline**
   ```
   Input Text -> Preprocessing -> Model Inference -> Post-processing -> Entity Output
   ```

### Development Workflow

The development workflow is structured around these key components:

1. **Configuration Management**
   - All parameters in `config.json`
   - Environment-specific settings
   - Model hyperparameters

2. **Data Processing Pipeline**
   - Standardized data format
   - Consistent preprocessing steps
   - Efficient batching strategies

## Technical Implementation

### Model Architecture

The NER implementation supports multiple architectures:

1. **BiLSTM Base Model**
   ```
   Input -> Embedding -> BiLSTM -> Dense -> Output
   ```

2. **BiLSTM-CRF Model**
   ```
   Input -> Embedding -> BiLSTM -> Dense -> CRF -> Output
   ```

3. **Transformer Model**
   ```
   Input -> Embedding -> Transformer Block -> Dense -> Output
   ```

### Data Processing Pipeline

The data processing pipeline includes:

1. **Text Preprocessing**
   - Tokenization
   - Sequence padding
   - Label encoding

2. **Feature Engineering**
   - Character-level features
   - Word embeddings
   - Position encoding

3. **Dataset Management**
   - TensorFlow data pipeline
   - Batching and prefetching
   - Cache management

### Enhancement Profiles

The project offers pre-configured enhancement profiles:

```powershell
.\run.ps1 --basic           # Basic configuration (no enhancements)
.\run.ps1 --embeddings      # With pre-trained word embeddings
.\run.ps1 --char-features   # With character-level features
.\run.ps1 --full            # All enhancements enabled
```

## Model Improvements

### Summary of Improvements

1. **Enhanced Model Architecture**
   - Added deeper BiLSTM layers with residual connections
   - Implemented self-attention mechanisms
   - Improved CRF layer implementation
   - Added character-level feature extraction

2. **Data Preparation Improvements**
   - Fixed tokenization process
   - Improved padding and masking for variable-length sequences
   - Added class weights to handle tag imbalance
   - Added support for pre-trained word embeddings

3. **Training Optimizations**
   - Implemented learning rate warmup and decay
   - Added mixed precision training
   - Improved sample weighting for padded sequences
   - Enhanced evaluation metrics with the seqeval library

### Implementation Details

The enhanced model introduces several configuration parameters:

```python
# Model Architecture
use_char_features = True     # Whether to use character-level features
char_embedding_dim = 25      # Dimension of character embeddings
char_hidden_dim = 25         # Hidden layer size for character BiLSTM
attention_heads = 8          # Number of attention heads in transformer or self-attention
recurrent_dropout_rate = 0.1 # Dropout rate for recurrent connections

# Training Optimizations
use_warmup = True            # Whether to use learning rate warmup
warmup_epochs = 2            # Number of warmup epochs
use_mixed_precision = True   # Whether to use mixed precision training
```

## Testing and Evaluation

### Evaluation Pipeline

The evaluation process consists of several stages:

1. **Model Validation**
   - Cross-validation on training data
   - Performance metrics calculation
   - Model comparison and selection

2. **Testing Framework**
   - Unit tests for components
   - Integration tests for pipelines
   - End-to-end testing scripts

3. **Performance Metrics**
   - Entity-level precision/recall/F1
   - Token-level accuracy
   - Confusion matrix analysis

### Monitoring and Logging

The project implements comprehensive monitoring:

1. **Training Monitoring**
   - TensorBoard integration
   - Custom metric logging
   - Resource utilization tracking

2. **Model Performance Tracking**
   - Model checkpointing
   - Performance history
   - Experiment tracking

3. **Debug Information**
   - Detailed logging levels
   - Error tracebacks
   - GPU utilization stats

### Deployment Considerations

Key aspects for production deployment:

1. **Model Serving**
   - REST API implementation
   - Batch prediction support
   - Model versioning

2. **Performance Optimization**
   - Model quantization
   - Batch processing
   - Caching strategies

3. **Monitoring**
   - Health checks
   - Performance metrics
   - Error handling

For implementation details of specific components, refer to the corresponding source files in the `src/` directory.
