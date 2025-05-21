"""
Dataset creation and processing for Named Entity Recognition.
"""

import tensorflow as tf
import numpy as np
import logging
from sklearn.model_selection import train_test_split
from keras.layers import TextVectorization

logger = logging.getLogger(__name__)


def create_ner_dataset(data, batch_size, max_seq_length, val_split=0.1, test_split=0.1, 
                       use_optimization=True, use_char_features=False, max_char_length=None):
    """
    Create TensorFlow datasets for NER from preprocessed data.
    ...
    """
    logger.info(f"Creating NER datasets with batch_size={batch_size}, max_seq_length={max_seq_length}, use_char_features={use_char_features}")

    sentences = data['Words'].values
    tags = data['Tag'].values    # Word tokenizer using TextVectorization
    word_tokenizer = TextVectorization(
        standardize=None,
        split='whitespace',
        output_mode='int',
        output_sequence_length=max_seq_length,
    )
    word_tokenizer.adapt([word for sentence in sentences for word in sentence])
    # Get vocabulary from TextVectorization
    word_vocab_list = word_tokenizer.get_vocabulary()
    # Create word_index dictionary for backward compatibility
    word_vocab = {word: idx for idx, word in enumerate(word_vocab_list)}
    vocabulary_size = len(word_vocab)
    logger.info(f"Word vocabulary size: {vocabulary_size}")

    # Tag tokenizer using TextVectorization
    tag_tokenizer = TextVectorization(
        standardize=None,
        split='whitespace',
        output_mode='int',
        output_sequence_length=max_seq_length,
    )
    tag_tokenizer.adapt([tag for sentence_tags in tags for tag in sentence_tags])
    # Get vocabulary from TextVectorization
    tag_vocab_list = tag_tokenizer.get_vocabulary()
    # Create tag_index dictionary for backward compatibility
    tag_vocab = {tag: idx for idx, tag in enumerate(tag_vocab_list)}

    max_tag_index = max(tag_vocab.values())
    num_tags = max_tag_index + 1

    logger.info(f"Number of unique tags in vocabulary: {len(tag_vocab)}")
    logger.info(f"Maximum tag index in vocabulary: {max_tag_index}")
    logger.info(f"Number of tag units needed for model: {num_tags}")

    # Character features using TextVectorization instead of the legacy Tokenizer
    char_vocab = None
    if use_char_features:
        all_chars = set()
        for sentence in sentences:
            for word in sentence:
                all_chars.update(word)

        # Create a character vectorizer that splits a string into a list of characters.
        char_vectorizer = TextVectorization(
            standardize=None,
            split=lambda x: list(x),
            output_mode='int'
        )
        # Include special tokens and adapt on all characters
        special_tokens = ['<PAD>', '<UNK>']
        char_vectorizer.adapt(special_tokens + list(all_chars))
        char_vocab_list = char_vectorizer.get_vocabulary()
        char_vocab = {char: idx for idx, char in enumerate(char_vocab_list)}
        logger.info(f"Character vocabulary size: {len(char_vocab)}")
        if max_char_length is None:
            max_char_length = 20

    X = []
    y = []
    char_X = [] if use_char_features else None
    max_observed_tag_value = 0

    for sentence_idx, (sentence, sentence_tags) in enumerate(zip(sentences, tags)):
        if len(sentence) != len(sentence_tags):
            min_len = min(len(sentence), len(sentence_tags))
            sentence = sentence[:min_len]
            sentence_tags = sentence_tags[:min_len]
            logger.warning(f"Sample {sentence_idx}: Adjusted sequence lengths to match: {min_len}")
        if len(sentence) == 0:
            logger.warning(f"Sample {sentence_idx}: Empty sentence. Skipping.")
            continue
        if len(sentence) > max_seq_length:
            logger.debug(f"Sample {sentence_idx}: Truncating sentence of length {len(sentence)} to {max_seq_length}")
            sentence = sentence[:max_seq_length]
            sentence_tags = sentence_tags[:max_seq_length]

        sentence_indices = []
        tag_indices = []
        char_indices = [] if use_char_features else None
        
        for word, tag in zip(sentence, sentence_tags):
            # For individual word processing, we need a different approach
            # TextVectorization is designed for batches of sequences, not individual words
            # Look up the word in our vocabulary directly
            word_index = word_vocab.get(word, word_vocab.get('<UNK>', 1))
            if word_index == 0:  # If it's padding token
                word_index = word_vocab.get('<UNK>', 1)
                logger.debug(f"Sample {sentence_idx}: Unknown word '{word}', using OOV token")
            
            # Look up the tag in our vocabulary directly
            tag_index = tag_vocab.get(tag, 0)
            if tag_index == 0:  # If it's padding token or unknown
                logger.warning(f"Sample {sentence_idx}: Invalid tag '{tag}' for word '{word}', skipping this token")
                continue
            max_observed_tag_value = max(max_observed_tag_value, tag_index)

            sentence_indices.append(word_index)
            tag_indices.append(tag_index)

            if use_char_features:
                char_word_indices = []
                for char in word[:max_char_length]:
                    # Lookup using the new char_vocab dictionary
                    char_word_indices.append(char_vocab.get(char, char_vocab.get('<UNK>')))
                if len(char_word_indices) < max_char_length:
                    char_word_indices.extend([0] * (max_char_length - len(char_word_indices)))
                elif len(char_word_indices) > max_char_length:
                    char_word_indices = char_word_indices[:max_char_length]
                char_indices.append(char_word_indices)

        if len(sentence_indices) > 0:
            X.append(sentence_indices)
            y.append(tag_indices)
            if use_char_features:
                char_X.append(char_indices)
        else:
            logger.warning(f"Sample {sentence_idx}: No valid tokens in sentence. Skipping.")

    logger.info(f"Processed {len(X)} valid sequences out of {len(sentences)} original samples")
    logger.info(f"Maximum observed tag value: {max_observed_tag_value}")

    if max_observed_tag_value >= num_tags:
        logger.warning(f"Found tag index {max_observed_tag_value} >= num_tags {num_tags}")
        logger.info(f"Adjusting num_tags from {num_tags} to {max_observed_tag_value + 1}")
        num_tags = max_observed_tag_value + 1

    X_padded = tf.keras.preprocessing.sequence.pad_sequences(X, maxlen=max_seq_length, padding='post', truncating='post', value=0)
    y_padded = tf.keras.preprocessing.sequence.pad_sequences(y, maxlen=max_seq_length, padding='post', truncating='post', value=0)

    char_X_padded = None
    if use_char_features:
        max_sent_len = min(max(len(x) for x in char_X), max_seq_length)
        char_X_padded = np.zeros((len(char_X), max_seq_length, max_char_length), dtype='int32')
        for i, char_sentence in enumerate(char_X):
            for j, char_word in enumerate(char_sentence[:max_seq_length]):
                char_X_padded[i, j, :len(char_word)] = char_word[:max_char_length]

    sample_weights = np.ones(len(X_padded), dtype=np.float32)

    tag_counts = {}
    for tags_seq in y:
        for tag in tags_seq:
            tag_counts[tag] = tag_counts.get(tag, 0) + 1

    total_samples = sum(tag_counts.values())
    num_classes = len(tag_counts)
    class_weights = {
        tag_idx: total_samples / (num_classes * count)
        for tag_idx, count in tag_counts.items()
    }
    logger.info(f"Class weights calculated to address imbalance: {class_weights}")

    if use_char_features:
        train_indices, temp_indices = train_test_split(
            np.arange(len(X_padded)), test_size=(val_split + test_split), random_state=42
        )
        val_indices, test_indices = train_test_split(
            temp_indices, test_size=test_split/(val_split + test_split), random_state=42
        )

        X_train, y_train = X_padded[train_indices], y_padded[train_indices]
        X_val, y_val = X_padded[val_indices], y_padded[val_indices]
        X_test, y_test = X_padded[test_indices], y_padded[test_indices]

        char_X_train = char_X_padded[train_indices]
        char_X_val = char_X_padded[val_indices]
        char_X_test = char_X_padded[test_indices]

        weights_train = sample_weights[train_indices]
        weights_val = sample_weights[val_indices]
        weights_test = sample_weights[test_indices]
    else:
        X_train, X_temp, y_train, y_temp, weights_train, weights_temp = train_test_split(
            X_padded, y_padded, sample_weights, test_size=(val_split + test_split), random_state=42
        )
        test_ratio = test_split / (val_split + test_split)
        X_val, X_test, y_val, y_test, weights_val, weights_test = train_test_split(
            X_temp, y_temp, weights_temp, test_size=test_ratio, random_state=42
        )

    logger.info(f"Train set size: {len(X_train)}, Validation set size: {len(X_val)}, Test set size: {len(X_test)}")

    if use_char_features:
        train_dataset = tf.data.Dataset.from_tensor_slices(
            ({"word_input": X_train, "char_input": char_X_train}, y_train, weights_train)
        )
        val_dataset = tf.data.Dataset.from_tensor_slices(
            ({"word_input": X_val, "char_input": char_X_val}, y_val, weights_val)
        )
        test_dataset = tf.data.Dataset.from_tensor_slices(
            ({"word_input": X_test, "char_input": char_X_test}, y_test, weights_test)
        )
    else:
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train, weights_train))
        val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val, weights_val))
        test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test, weights_test))

    if use_optimization:
        train_dataset = train_dataset.cache()
        buffer_size = min(len(X_train), 10000)
        train_dataset = train_dataset.shuffle(buffer_size=buffer_size, reshuffle_each_iteration=True)
        train_dataset = train_dataset.batch(batch_size, drop_remainder=True)
        val_dataset = val_dataset.batch(batch_size)
        test_dataset = test_dataset.batch(batch_size)
        train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)
        val_dataset = val_dataset.prefetch(tf.data.experimental.AUTOTUNE)
        test_dataset = test_dataset.prefetch(tf.data.experimental.AUTOTUNE)
    else:
        train_dataset = train_dataset.batch(batch_size)
        val_dataset = val_dataset.batch(batch_size)
        test_dataset = test_dataset.batch(batch_size)

    idx2word = {v: k for k, v in word_vocab.items()}
    idx2tag = {v: k for k, v in tag_vocab.items()}

    if use_char_features:
        return (train_dataset, val_dataset, test_dataset, 
                word_vocab, tag_vocab, char_vocab, 
                idx2word, idx2tag, class_weights)
    else:
        return (train_dataset, val_dataset, test_dataset, 
                word_vocab, tag_vocab, 
                idx2word, idx2tag, class_weights)


def create_tag_lookup_tables(tag_input):
    """
    Create lookup tables for tags.
    
    Args:
        tag_input: Either a dictionary mapping tags to indices, or a sequence of tags
        
    Returns:
        tuple: (idx2tag, tag2idx) mappings
    """
    if isinstance(tag_input, dict):
        # If input is already a dictionary, use it as tag2idx
        tag2idx = dict(tag_input)  # Create a copy
    else:
        # If input is a sequence (list, Series, etc), create new mapping
        unique_tags = set(tag_input)
        tag2idx = {tag: idx + 1 for idx, tag in enumerate(sorted(unique_tags))}  # Start from 1
    
    # Ensure PAD token has index 0
    tag2idx['<PAD>'] = 0
    # Create reverse mapping
    idx2tag = {idx: tag for tag, idx in tag2idx.items()}
    
    return idx2tag, tag2idx
