"""
Data loading and preprocessing utilities for Named Entity Recognition.
"""

import pandas as pd
import numpy as np
import logging
import ast
from pathlib import Path

logger = logging.getLogger(__name__)

def load_data(data_path):
    """
    Load NER dataset from CSV file.
    
    Args:
        data_path: Path to the CSV file containing NER data
        
    Returns:
        pd.DataFrame: Loaded dataframe
    """
    logger.info(f"Loading data from {data_path}")
    
    try:
        data = pd.read_csv(data_path)
        logger.info(f"Data loaded successfully. Shape: {data.shape}")
        
        # Check for NaN values
        nan_count = data.isna().sum().sum()
        if nan_count > 0:
            logger.warning(f"Found {nan_count} NaN values in the dataset")
            data.dropna(inplace=True)
            logger.info(f"Dropped NaN values. New shape: {data.shape}")
        
        return data
    
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        raise


def preprocess_data(data):
    """
    Preprocess the NER data.
    
    Args:
        data: Dataframe containing NER data
        
    Returns:
        pd.DataFrame: Preprocessed dataframe
    """
    logger.info("Preprocessing data")
    
    # Make a copy to avoid modifying the original
    df = data.copy()
    
    try:
        # Process the POS and Tag columns
        for i in range(len(df)):
            # Parse string representations of lists
            pos = ast.literal_eval(df.at[i, 'POS']) if isinstance(df.at[i, 'POS'], str) else df.at[i, 'POS']
            tags = ast.literal_eval(df.at[i, 'Tag']) if isinstance(df.at[i, 'Tag'], str) else df.at[i, 'Tag']
            
            # Convert to string and normalize tags
            df.at[i, 'POS'] = [str(word) for word in pos]
            df.at[i, 'Tag'] = [str(word.upper()) for word in tags]
        
        # Split sentences into words for consistency
        df['Words'] = df['Sentence'].apply(lambda x: x.split())
        
        # Validate lengths
        invalid_rows = []
        for i in range(len(df)):
            if len(df.at[i, 'Words']) != len(df.at[i, 'Tag']):
                logger.warning(f"Row {i}: Mismatch in length between words ({len(df.at[i, 'Words'])}) and tags ({len(df.at[i, 'Tag'])})")
                invalid_rows.append(i)
        
        if invalid_rows:
            logger.warning(f"Removing {len(invalid_rows)} rows with mismatched word/tag lengths")
            df = df.drop(invalid_rows).reset_index(drop=True)
        
        logger.info("Data preprocessing completed successfully")
        return df
    
    except Exception as e:
        logger.error(f"Error during data preprocessing: {e}")
        raise


def extract_entities(words, tags):
    """
    Extract named entities from a list of words and corresponding tags.
    
    Args:
        words: List of words in a sentence
        tags: List of NER tags in IOB or BILOU format
        
    Returns:
        list: List of entity dicts with entity type and text
    """
    entities = []
    current_entity = None
    
    for word, tag in zip(words, tags):
        if tag == 'O' or tag == '0':  # Outside any entity
            if current_entity:
                entities.append(current_entity)
                current_entity = None
        elif tag.startswith('B-') or tag.startswith('I-') or tag.startswith('U-') or tag.startswith('L-'):
            # Extract entity type (PER, LOC, ORG, etc.)
            tag_type = tag[2:]
            
            # Start a new entity on B- or U- tag
            if tag.startswith('B-') or tag.startswith('U-'):
                if current_entity:
                    entities.append(current_entity)
                current_entity = {'type': tag_type, 'text': word}
            # Continue current entity on I- or L- tag
            elif current_entity and tag_type == current_entity['type']:
                current_entity['text'] += ' ' + word
            
            # End the entity on L- or U- tag
            if tag.startswith('L-') or tag.startswith('U-'):
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None
    
    # Add the last entity if we have one
    if current_entity:
        entities.append(current_entity)
    
    return entities
