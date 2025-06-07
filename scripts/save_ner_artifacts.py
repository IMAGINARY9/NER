#!/usr/bin/env python3
"""
Script to save NER model artifacts for integration with Comment-ABSA.

This script loads the trained NER model and extracts/saves:
1. Word vocabulary (TextVectorization-compatible format)
2. Tag vocabulary (mapping tags to indices)
3. Model metadata (max_seq_length, etc.)
"""

import os
import sys
import json
import logging
import tensorflow as tf
from pathlib import Path

# Add the NER src directory to the path
ner_root = Path(__file__).resolve().parent.parent
src_dir = ner_root / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from data_processing.data_loader import load_data, preprocess_data
from data_processing.dataset import create_ner_dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def save_ner_artifacts(model_path, data_path, output_dir, max_seq_length=128):
    """
    Save NER model artifacts for integration with Comment-ABSA.
    
    Args:
        model_path: Path to the trained NER model
        data_path: Path to the NER training data
        output_dir: Directory to save artifacts
        max_seq_length: Maximum sequence length used by NER model
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data to get vocabularies
    logger.info(f"Loading data from {data_path}")
    data = load_data(data_path)
    preprocessed_data = preprocess_data(data)
    
    # Create dataset to get vocabularies (using same parameters as training)
    logger.info("Creating dataset to extract vocabularies")
    _, _, _, word_vocab, tag_vocab, idx2word, idx2tag, _ = create_ner_dataset(
        preprocessed_data, 
        batch_size=32,  # Batch size doesn't matter for vocab extraction
        max_seq_length=max_seq_length,
        val_split=0.1,
        test_split=0.1,
        use_optimization=False
    )
    
    # Save word vocabulary in a format compatible with TextVectorization
    word_vocab_file = os.path.join(output_dir, "word_vocab.json")
    logger.info(f"Saving word vocabulary to {word_vocab_file}")
    
    # Convert to format expected by CommentABSANE RExtractor
    # We need to reverse the word_vocab to get word->index mapping
    word_index_mapping = {word: idx for word, idx in word_vocab.items()}
    
    # Add special tokens if not present
    if '<PAD>' not in word_index_mapping:
        word_index_mapping['<PAD>'] = 0
    if '<UNK>' not in word_index_mapping:
        word_index_mapping['<UNK>'] = 1
        
    # Create a Keras tokenizer-like structure
    tokenizer_config = {
        "class_name": "Tokenizer",
        "config": {
            "num_words": len(word_vocab),
            "filters": '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
            "lower": True,
            "split": " ",
            "char_level": False,
            "oov_token": "<UNK>",
            "document_count": 0
        },
        "word_index": word_index_mapping,
        "word_counts": {},  # Not needed for prediction
        "index_docs": {},   # Not needed for prediction
        "index_word": {str(idx): word for word, idx in word_index_mapping.items()}
    }
    
    with open(word_vocab_file, 'w', encoding='utf-8') as f:
        json.dump(tokenizer_config, f, indent=2, ensure_ascii=False)
    
    # Save tag vocabulary
    tag_vocab_file = os.path.join(output_dir, "tag_vocab.json")
    logger.info(f"Saving tag vocabulary to {tag_vocab_file}")
    with open(tag_vocab_file, 'w', encoding='utf-8') as f:
        json.dump(tag_vocab, f, indent=2, ensure_ascii=False)
    
    # Save model metadata
    metadata_file = os.path.join(output_dir, "model_metadata.json")
    logger.info(f"Saving model metadata to {metadata_file}")
    metadata = {
        "max_seq_length": max_seq_length,
        "vocab_size": len(word_vocab),
        "num_tags": len(tag_vocab),
        "tag_names": list(tag_vocab.keys()),
        "model_path": model_path
    }
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    # Save idx2tag for easy lookup
    idx2tag_file = os.path.join(output_dir, "idx2tag.json")
    logger.info(f"Saving idx2tag mapping to {idx2tag_file}")
    # Convert keys to strings for JSON serialization
    idx2tag_str = {str(k): v for k, v in idx2tag.items()}
    with open(idx2tag_file, 'w', encoding='utf-8') as f:
        json.dump(idx2tag_str, f, indent=2, ensure_ascii=False)
    
    logger.info(f"All NER artifacts saved to {output_dir}")
    logger.info(f"Vocabulary size: {len(word_vocab)}")
    logger.info(f"Number of tags: {len(tag_vocab)}")
    logger.info(f"Tag names: {list(tag_vocab.keys())}")

def main():
    # Default paths - adjust as needed
    model_path = "models/ner_model_final_20250520-204254.keras"
    data_path = "data/ner.csv"
    output_dir = "models/ner_artifacts"
    max_seq_length = 128
    
    # Check if model exists
    if not os.path.exists(model_path):
        logger.error(f"Model file not found: {model_path}")
        logger.info("Available model files:")
        models_dir = Path("models")
        if models_dir.exists():
            for f in models_dir.glob("*.keras"):
                logger.info(f"  {f}")
            for f in models_dir.glob("*.h5"):
                logger.info(f"  {f}")
        return 1
    
    # Check if data exists
    if not os.path.exists(data_path):
        logger.error(f"Data file not found: {data_path}")
        return 1
    
    try:
        save_ner_artifacts(model_path, data_path, output_dir, max_seq_length)
        logger.info("✅ NER artifacts saved successfully!")
        return 0
    except Exception as e:
        logger.error(f"❌ Error saving NER artifacts: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
