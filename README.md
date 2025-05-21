# Enhanced Entity Extraction Project

A project for Named Entity Recognition (NER) using neural networks, built with TensorFlow and optimized for performance.

## Overview

This project implements a Named Entity Recognition system that can identify entities such as:
- Persons
- Organizations
- Locations
- Miscellaneous entities

The implementation includes multiple model architectures:
- BiLSTM
- BiLSTM-CRF (Conditional Random Fields)
- Transformer-based models

## Project Updates

- All functionality is now available through unified scripts
- Comprehensive documentation is available in [docs/guide.md](docs/guide.md)
- Original script names still work for backward compatibility

## Model Improvements

This implementation includes significant improvements:

- **Enhanced Model Architecture**:
  - Deeper BiLSTM layers with residual connections
  - Self-attention mechanisms for better context understanding
  - Character-level features to capture morphological patterns
  - Improved CRF layer implementation

- **Data Preparation Improvements**:
  - Better tokenization and padding strategies
  - Class imbalance handling with weighted loss
  - Support for pre-trained word embeddings (GloVe)

- **Training Optimizations**:
  - Learning rate scheduling with warmup and decay
  - Enhanced mixed precision training
  - Improved evaluation metrics with entity-level scoring

## Project Structure

```
ner/
├── data/                 # Dataset directory
├── logs/                 # Training logs
│   └── fit/              # TensorBoard logs
├── models/               # Saved models
│   └── checkpoints/      # Model checkpoints 
├── notebooks/            # Jupyter notebooks
├── docs/                 # Documentation
│   ├── model_improvements.md      # Details of model improvements
│   └── implementation_guide.md    # Implementation guide
├── visualizations/       # Visualization outputs
└── src/                  # Source code
    ├── data_processing/  # Data loading and processing
    ├── models/           # Model architectures
    ├── training/         # Training utilities
    ├── utils/            # Helper functions
    │   └── embeddings.py # Pre-trained embedding utilities
    └── services/         # API services
    └── visualize.py      # Visualization tools
```

## Features

- Multiple model architectures (BiLSTM, BiLSTM-CRF, Transformer)
- Performance optimizations (XLA compilation, mixed precision)
- Dataset optimizations (caching, prefetching)
- Detailed evaluation metrics
- Entity visualization
- Pre-trained word embeddings
- Character-level features

## Getting Started

### Prerequisites

- Python 3.8+
- TensorFlow 2.8+
- Required packages listed in `requirements.txt`

### Installation

1. Clone the repository
2. Set up the environment:

```bash
# Windows
setup.bat

# Linux/Mac
./setup.sh
```

3. Activate the environment:

```bash
# Windows
activate.bat

# Linux/Mac
source activate.sh
```

### Running the Improved Model

Use the dedicated scripts to run the improved model:

```bash
# Basic training
./run.bat --basic

# With pre-trained embeddings
./run.bat --embeddings

# With character features
./run.bat --char-features

# With all enhancements
./run.bat --full

# Customize parameters
./run.bat --full --batch-size 64 --epochs 20 --model-type transformer --learning-rate 0.0005 --max-seq-length 128 --use-gpu --use-xla --use-mixed-precision
```

### Visualizing Results

After training, visualize the model performance:

```bash
# Windows
visualize.bat --model-path models/your_model.h5 --data-path data/ner.csv

# Or let it find the latest model automatically
visualize.bat
```

## Optimizations

The project includes several optimizations for better performance:

- **XLA compilation**: Just-in-time compilation for faster training
- **Mixed precision training**: Uses both float16 and float32 for better performance
- **Dataset optimizations**: Caching, optimized shuffling, and prefetching
- **Learning rate scheduling**: Warmup and cosine decay for better convergence
- **Gradient accumulation**: For training with larger effective batch sizes

## Performance

The enhanced model achieves significantly better performance than the baseline:

- Baseline model: ~13% accuracy
- Enhanced model: Expected 70%+ accuracy (varies by configuration)

For detailed performance analysis, see the visualization outputs after training.

## References

This implementation is based on techniques from:

1. Lample et al. "Neural Architectures for Named Entity Recognition" (2016)
2. Ma and Hovy "End-to-end Sequence Labeling via Bi-directional LSTM-CNNs-CRF" (2016)
3. Vaswani et al. "Attention is All You Need" (2017)
4. Devlin et al. "BERT: Pre-training of Deep Bidirectional Transformers" (2018)

- **Gradient accumulation**: Train with larger effective batch sizes even on memory-limited hardware
- **Environment optimizations**: Optimized TensorFlow environment variables for better GPU utilization

## Getting Started

### Prerequisites

- Python 3.8+
- TensorFlow 2.x
- TensorFlow Addons (optional, for CRF layer)
- NumPy, Pandas, Scikit-learn
- Matplotlib, Seaborn (for visualizations)

### Installation

1. Clone the repository

2. Use the provided setup scripts (recommended):
   ```bash
   # Windows PowerShell
   .\setup.ps1

   # Windows CMD
   setup.bat

   # Linux/MacOS
   chmod +x setup.sh
   ./setup.sh
   ```

3. Or manually set up a virtual environment:
   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate
   pip install -r requirements.txt

   # Linux/MacOS
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

4. Special setup options:
   ```bash
   # Windows - Force CPU-only mode
   .\setup.ps1 -UseCPUOnly

   # Linux/MacOS - Force CPU-only mode
   ./setup.sh --cpu-only
   ```

### Training a Model

Use the unified run script (recommended):
```bash
# Windows
.\run.bat --data-path data\ner.csv --model-type bilstm-crf --epochs 10 --batch-size 32

# PowerShell
.\run.ps1 -data-path data\ner.csv -model-type bilstm-crf -epochs 10 -batch-size 32

# Linux/MacOS
chmod +x run.sh
./run.sh --data-path data/ner.csv --model-type bilstm-crf --epochs 10 --batch-size 32
```

Alternatively, use the enhancement profiles for improved performance:
```bash 
# Windows - run with all enhancements
.\run.bat --full --epochs 15

# PowerShell - run with pre-trained embeddings only
.\run.ps1 -embeddings -epochs 15
```

Or manually activate the virtual environment first:
```bash
# Windows
.\activate.bat
# or
venv\Scripts\activate

# Linux/MacOS
source ./activate.sh
# or
source venv/bin/activate
```

Then run the training:
```bash
python src/main.py --data-path data/ner.csv \
                 --model-dir ./models \
                 --log-dir ./logs \
                 --epochs 10 \
                 --batch-size 32 \
                 --model-type bilstm-crf
```

#### CPU/GPU Options

To force CPU-only mode:
```bash
# Windows
.\run.bat --no-gpu --data-path data\ner.csv --model-type bilstm-crf

# PowerShell
.\run.ps1 -no-gpu -data-path data\ner.csv -model-type bilstm-crf

# Linux/MacOS
./run.sh --no-gpu --data-path data/ner.csv --model-type bilstm-crf
```

To test GPU availability:
```bash
# Windows
.\run.bat --test-gpu

# PowerShell
.\run.ps1 -test-gpu

# Linux/MacOS
./run.sh --test-gpu
```

### Making Predictions

Use the unified run script:
```bash
# Windows
.\run.bat --mode predict --predict-text "Apple is looking to buy U.K. startup for $1 billion" --model-dir models/ner_model_final.h5

# PowerShell
.\run.ps1 -mode predict -predict-text "Apple is looking to buy U.K. startup for $1 billion" -model-dir models/ner_model_final.h5

# Linux/MacOS
./run.sh --mode predict --predict-text "Apple is looking to buy U.K. startup for $1 billion" --model-dir models/ner_model_final.h5
```

Or with virtual environment activated manually:
```bash
python src/main.py --mode predict \
                 --predict-text "Apple is looking to buy U.K. startup for $1 billion" \
                 --model-dir ./models/ner_model_final.h5
```

#### Jupyter Notebook for Interactive NER

For an interactive experience, use the Jupyter notebook:
```bash
# Windows
.\run.bat --jupyter

# PowerShell
.\run.ps1 -jupyter

# Linux/MacOS
./run.sh --jupyter

# Or manually with virtual environment activated:
jupyter notebook notebooks/ner_training_example.ipynb
```

## Evaluation and Visualization

The model is evaluated using standard NER metrics:
- F1 Score
- Precision
- Recall
- Entity-level evaluation

Use the unified visualization tools to analyze model performance:

```bash
# Windows - basic usage with latest model
visualize.bat

# Windows - specify model and data
visualize.bat --model-path models/your_model.keras --data-path data/ner.csv

# Windows - specific visualization type
visualize.bat --type confusion  # Options: confusion, distribution, metrics, examples

# Windows PowerShell - basic usage
.\visualize.ps1

# Windows PowerShell - with interactive mode
.\visualize.ps1 -interactive

# Linux/MacOS
./visualize.sh --model-path models/your_model.keras --type all

# Custom output directory
visualize.bat --output-dir my_visualizations

# Multiple visualization types
visualize.bat --type "confusion,metrics,examples" --interactive
```

This will generate various visualizations including:
- Confusion matrices for entity classification
- Entity distribution plots
- Precision/Recall/F1 metrics by entity type
- Sample predictions on test data

The visualizations will be saved in the specified output directory (default: `visualizations/`).

For detailed visualization options, see the [unified documentation guide](docs/unified_guide.md).

## License

MIT

## References

- [Named Entity Recognition with LSTMs](https://www.depends-on-the-definition.com/named-entity-recognition-with-lstms-and-elmo/)
- [Neural Architectures for Named Entity Recognition](https://arxiv.org/abs/1603.01360)
- [A Survey on Deep Learning for Named Entity Recognition](https://arxiv.org/abs/1812.09449)

## Usage

### Training Mode
To train the model, use the following command:
```bash
python src/main.py --mode train --batch-size 32 --epochs 15 --learning-rate 0.001 --model-type bilstm-crf --max-seq-length 128 --use-char-features --no-pretrained-embeddings
```

### Evaluation Mode
To evaluate the model, use the following command:
```bash
python src/main.py --mode evaluate --model-dir models/my_model --batch-size 32
```

### Prediction Mode
To predict entities for a given text, use the following command:
```bash
python src/main.py --mode predict --predict-text "John works at OpenAI in San Francisco." --model-dir models/my_model
```

For more details on the arguments, refer to the script documentation or use the `--help` flag:
```bash
python src/main.py --help
```
