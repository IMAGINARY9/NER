"""
Utilities for working with pre-trained word embeddings.
"""

import os
import numpy as np
import logging
import gzip
import shutil
from pathlib import Path
import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)

def download_file(url, destination):
    """
    Download a file from a URL to a destination with progress bar.
    
    Args:
        url: URL of the file to download
        destination: Local path to save the file
    """
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    
    with open(destination, 'wb') as file, tqdm(
        desc=f"Downloading {os.path.basename(destination)}",
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            file.write(data)
            bar.update(len(data))

def extract_gz(gz_path, extract_path):
    """
    Extract a gzip file.
    
    Args:
        gz_path: Path to the gzip file
        extract_path: Path to extract the file to
    """
    with gzip.open(gz_path, 'rb') as f_in:
        with open(extract_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    
    logger.info(f"Extracted {gz_path} to {extract_path}")

def download_glove_embeddings(embedding_dim=100, cache_dir='./cache'):
    """
    Download GloVe embeddings if not already cached.
    
    Args:
        embedding_dim: Dimension of embeddings to download (50, 100, 200, or 300)
        cache_dir: Directory to cache embeddings
    
    Returns:
        str: Path to the embeddings file
    """
    valid_dims = [50, 100, 200, 300]
    if embedding_dim not in valid_dims:
        logger.warning(f"Invalid embedding dimension {embedding_dim}. Using 100 instead.")
        embedding_dim = 100
    
    # Create cache directory if it doesn't exist
    os.makedirs(cache_dir, exist_ok=True)
    
    # Define file paths
    gz_file_path = os.path.join(cache_dir, f'glove.6B.{embedding_dim}d.txt.gz')
    embeddings_file_path = os.path.join(cache_dir, f'glove.6B.{embedding_dim}d.txt')
    
    # Download embeddings if not already downloaded
    if not os.path.exists(embeddings_file_path):
        if not os.path.exists(gz_file_path):
            logger.info(f"Downloading GloVe embeddings with dimension {embedding_dim}...")
            url = f"https://nlp.stanford.edu/data/glove.6B.zip"
            
            # Since the file is a zip containing all dimensions, we need to handle differently
            zip_path = os.path.join(cache_dir, 'glove.6B.zip')
            download_file(url, zip_path)
            
            # Extract the zip
            import zipfile
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extract(f'glove.6B.{embedding_dim}d.txt', cache_dir)
            
            logger.info(f"Extracted embeddings from zip file")
            
            # Remove the zip file to save space
            os.remove(zip_path)
        else:
            logger.info(f"Found compressed embeddings file, extracting...")
            extract_gz(gz_file_path, embeddings_file_path)
            
            # Remove the gz file to save space
            os.remove(gz_file_path)
    else:
        logger.info(f"Using cached GloVe embeddings from {embeddings_file_path}")
    
    return embeddings_file_path

def load_embeddings(embeddings_file, word_vocab, embedding_dim=100):
    """
    Load pre-trained embeddings for vocabulary words.
    
    Args:
        embeddings_file: Path to the embeddings file
        word_vocab: Dictionary mapping words to indices
        embedding_dim: Dimension of the embeddings
        
    Returns:
        numpy.ndarray: Embedding matrix
    """
    logger.info(f"Loading embeddings from {embeddings_file}")
    
    # Initialize embedding matrix - add 1 for padding token
    vocab_size = len(word_vocab) + 1
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    
    # Load word vectors
    word_vectors = {}
    with open(embeddings_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(tqdm(f, desc="Loading word vectors")):
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            word_vectors[word] = vector
    
    logger.info(f"Loaded {len(word_vectors)} word vectors")
    
    # Create embedding matrix
    found_words = 0
    for word, idx in tqdm(word_vocab.items(), desc="Building embedding matrix"):
        embedding_vector = word_vectors.get(word.lower())
        if embedding_vector is not None:
            embedding_matrix[idx] = embedding_vector
            found_words += 1
    
    logger.info(f"Found embeddings for {found_words}/{len(word_vocab)} words ({found_words / len(word_vocab):.2%})")
    
    # Save the embedding matrix
    matrix_path = os.path.join(os.path.dirname(embeddings_file), f'embedding_matrix_{embedding_dim}d.npy')
    np.save(matrix_path, embedding_matrix)
    logger.info(f"Saved embedding matrix to {matrix_path}")
    
    return embedding_matrix, matrix_path
