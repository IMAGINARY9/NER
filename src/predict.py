"""
Prediction module for Named Entity Recognition.

This module handles predictions using a trained NER model.
"""

import numpy as np
import tensorflow as tf
import logging
import os
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
import pandas as pd

logger = logging.getLogger(__name__)


class NERPredictor:
    """Class for making predictions with a trained NER model."""
    
    def __init__(self, model_path, word_tokenizer, tag_vocab):
        """
        Initialize the predictor.
        
        Args:
            model_path: Path to the trained model file
            word_tokenizer: Tokenizer for words or word vocabulary dict
            tag_vocab: Tag vocabulary mapping
        """
        self.model = self._load_model(model_path)
        self.word_tokenizer = word_tokenizer
        self.is_dict_vocab = isinstance(word_tokenizer, dict)
        
        # Create index to tag mapping
        self.idx2tag = {idx: tag for tag, idx in tag_vocab.items()}
        self.idx2tag[0] = 'PAD'  # Add padding tag
        
        logger.info(f"NER predictor initialized with model from {model_path}")
    
    def _load_model(self, model_path):
        """Load the trained model."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        try:
            model = load_model(model_path)
            logger.info(f"Model loaded successfully from {model_path}")
            return model
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def tokenize_sentence(self, sentence):
        """
        Tokenize a sentence.
        
        Args:
            sentence: Text string to tokenize
            
        Returns:
            list: List of tokens
        """
        # Simple tokenization by splitting on spaces
        # In a more advanced implementation, use a proper tokenizer
        tokens = sentence.split()
        return tokens
    
    def predict(self, text, max_seq_length=128):
        """
        Predict entities in text.
        
        Args:
            text: Text to find entities in
            max_seq_length: Maximum sequence length
            
        Returns:
            tuple: (tokens, predicted_tags, entities)
        """
        # Tokenize the input text
        tokens = self.tokenize_sentence(text)
        
        # Convert tokens to sequences based on vocabulary
        if self.is_dict_vocab:
            # Using dictionary vocabulary directly
            X = []
            for token in tokens:
                # Get token index or unknown token index
                token_idx = self.word_tokenizer.get(token, self.word_tokenizer.get('<UNK>', 1))
                X.append(token_idx)
            X = [X]
        else:
            # Using TextVectorization or Tokenizer
            X = self.word_tokenizer.texts_to_sequences([tokens])
        
        # Pad sequences
        X = pad_sequences(X, maxlen=max_seq_length, padding='post', truncating='post')
        
        # Make prediction
        y_pred = self.model.predict(X)
        
        # Get the tag with highest probability for each token
        if len(y_pred.shape) == 3:  # For models with softmax output
            y_pred = np.argmax(y_pred, axis=-1)
        
        # Convert predicted indices to tags
        predicted_tags = []
        for idx in y_pred[0][:len(tokens)]:
            tag = self.idx2tag.get(idx, 'O')
            predicted_tags.append(tag)
        
        # Extract entities from predicted tags
        entities = self._extract_entities_from_tags(tokens, predicted_tags)
        
        return tokens, predicted_tags, entities
    
    def _extract_entities_from_tags(self, tokens, tags):
        """Extract named entities from tokens and corresponding tags."""
        entities = []
        current_entity = None
        
        for i, (token, tag) in enumerate(zip(tokens, tags)):
            # Skip if it's a padding token
            if tag == 'PAD':
                continue
                
            # Check if it's a beginning of a new entity
            if tag.startswith('B-'):
                # If there was an active entity, add it to the list
                if current_entity:
                    entities.append(current_entity)
                
                # Start a new entity
                entity_type = tag[2:]  # Remove the 'B-' prefix
                current_entity = {
                    'text': token,
                    'type': entity_type,
                    'start': i,
                    'end': i
                }
            
            # Check if it's inside an entity
            elif tag.startswith('I-'):
                entity_type = tag[2:]  # Remove the 'I-' prefix
                
                # If the current entity exists and the type matches
                if current_entity and current_entity['type'] == entity_type:
                    current_entity['text'] += ' ' + token
                    current_entity['end'] = i
                # If types don't match or no active entity, treat as O
                else:
                    if current_entity:
                        entities.append(current_entity)
                        current_entity = None
            
            # Outside any entity
            else:  # 'O' or any other tag
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None
        
        # Add the last entity if there is one
        if current_entity:
            entities.append(current_entity)
        
        return entities
    
    def visualize_entities(self, tokens, tags, entities):
        """
        Create a visualization of the predicted entities.
        
        Args:
            tokens: List of tokens
            tags: List of predicted tags
            entities: List of extracted entities
            
        Returns:
            pd.DataFrame: DataFrame for visualization
        """
        # Create DataFrame for visualization
        df = pd.DataFrame({
            'Token': tokens,
            'Tag': tags
        })
        
        # Create entity info
        entity_info = []
        for i, token in enumerate(tokens):
            entity_info.append('')
        
        for entity in entities:
            entity_info[entity['start']] = f"START {entity['type']}"
            entity_info[entity['end']] = f"END {entity['type']}"
        
        df['Entity'] = entity_info
        
        return df
