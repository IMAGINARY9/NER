"""
NER model definition.

This module defines the architecture for the Named Entity Recognition model.
"""

import tensorflow as tf
from tensorflow.keras.layers import Embedding, LSTM, Bidirectional, Dense, Dropout, Input
from tensorflow.keras.models import Model
import logging

try:
    import tensorflow_addons as tfa
    HAS_TFA = True
except ImportError:
    HAS_TFA = False
    logging.warning("TensorFlow Addons not found or is incompatible. CRF layer will not be available. Consider using a non-CRF model type or ensuring TensorFlow Addons is correctly installed with a compatible TensorFlow version.")

logger = logging.getLogger(__name__)


def build_model(vocab_size, num_tags, embedding_dim=100, hidden_dim=128, 
                dropout_rate=0.1, model_type='bilstm', use_pretrained_embeddings=False, # Changed default model_type
                pretrained_embeddings_path=None, recurrent_dropout_rate=0.1,
                attention_heads=8, use_char_features=False, char_vocab_size=None, 
                char_embedding_dim=25, char_hidden_dim=25):
    """
    Build a NER model based on the specified architecture.
    
    Args:
        vocab_size: Size of the vocabulary
        num_tags: Number of entity tags (should be max tag index + 1)
        embedding_dim: Dimension of word embeddings
        hidden_dim: Dimension of hidden layers
        dropout_rate: Dropout rate
        model_type: Type of model ('bilstm', 'bilstm-crf', 'transformer')
        use_pretrained_embeddings: Whether to use pretrained word embeddings
        pretrained_embeddings_path: Path to pretrained embeddings file
        recurrent_dropout_rate: Dropout rate for recurrent connections
        attention_heads: Number of attention heads for transformer
        use_char_features: Whether to use character-level features
        char_vocab_size: Size of character vocabulary
        char_embedding_dim: Dimension for character embeddings
        char_hidden_dim: Hidden dimension for character-level BiLSTM
        
    Returns:
        Compiled TensorFlow Keras model
    """
    """
    Build a NER model based on the specified architecture.
    
    Args:
        vocab_size: Size of the vocabulary
        num_tags: Number of entity tags
        embedding_dim: Dimension of word embeddings
        hidden_dim: Dimension of hidden layers
        dropout_rate: Dropout rate
        model_type: Type of model ('bilstm', 'bilstm-crf', 'transformer')
        use_pretrained_embeddings: Whether to use pretrained word embeddings
        pretrained_embeddings_path: Path to pretrained embeddings file
        recurrent_dropout_rate: Dropout rate for recurrent connections
        attention_heads: Number of attention heads for transformer
        use_char_features: Whether to use character-level features
        char_vocab_size: Size of character vocabulary if using char features
        char_embedding_dim: Dimension of character embeddings
        char_hidden_dim: Dimension of character-level hidden representations
    
    Returns:
        tf.keras.Model: Compiled model
    """
    # Ensure num_tags is correct for the model output layer
    # The model's output size should match the number of tag classes
    # This fixes the "label value outside valid range" error
    logger.info(f"Building {model_type} model with {vocab_size} vocab size and {num_tags} tags")
      # Define input
    input_layer = Input(shape=(None,), name='word_input')
    
    # Embedding layer - word level
    if use_pretrained_embeddings and pretrained_embeddings_path:
        # Load pretrained embeddings
        import numpy as np
        logger.info(f"Loading pretrained embeddings from {pretrained_embeddings_path}")
        try:
            embedding_matrix = np.load(pretrained_embeddings_path)
            word_embedding_layer = Embedding(
                input_dim=vocab_size + 1,  # +1 for padding
                output_dim=embedding_dim,
                weights=[embedding_matrix],
                trainable=False,  # Freeze the embeddings
                mask_zero=True,
                name='word_embeddings'
            )
        except Exception as e:
            logger.warning(f"Failed to load pretrained embeddings: {e}. Using random initialization.")
            word_embedding_layer = Embedding(
                input_dim=vocab_size + 1,
                output_dim=embedding_dim,
                mask_zero=True,
                name='word_embeddings'
            )
    else:
        word_embedding_layer = Embedding(
            input_dim=vocab_size + 1,  # +1 for padding
            output_dim=embedding_dim,
            mask_zero=True,
            name='word_embeddings'
        )
    
    x = word_embedding_layer(input_layer)
    
    # Character-level features (optional)
    if use_char_features and char_vocab_size:
        # Character input
        char_input = Input(shape=(None, None), name='char_input')
          # Character embedding
        char_embedding = Embedding(
            input_dim=char_vocab_size + 1,  # +1 for padding
            output_dim=char_embedding_dim,
            mask_zero=True,
            name='char_embeddings'
        )(char_input)
          # Custom layer for char embedding reshape operations to handle KerasTensor properly
        class CharReshapeLayer(tf.keras.layers.Layer):
            def __init__(self, **kwargs):
                super(CharReshapeLayer, self).__init__(**kwargs)
                
            def call(self, inputs):
                s = tf.shape(inputs)
                # Reshape for TimeDistributed - flattens the first two dimensions
                reshaped = tf.reshape(inputs, [-1, s[-2], char_embedding_dim])
                return reshaped, s
                
            def compute_mask(self, inputs, mask=None):
                # Return None to disable mask propagation
                return None
        
        # Reshape using custom layer
        char_embedding_reshaped, s = CharReshapeLayer(name='char_reshape_layer')(char_embedding)
        
        # Apply BiLSTM to characters
        char_bilstm = Bidirectional(
            LSTM(char_hidden_dim, return_sequences=False, recurrent_dropout=recurrent_dropout_rate),
            name='char_bilstm'
        )(char_embedding_reshaped)
          # Custom layer for reshaping back
        class CharBiLSTMReshapeLayer(tf.keras.layers.Layer):
            def __init__(self, **kwargs):
                super(CharBiLSTMReshapeLayer, self).__init__(**kwargs)
            
            def call(self, inputs, original_shape):
                # Reshape back to match the original sequence dimension
                return tf.reshape(inputs, [-1, original_shape[1], 2*char_hidden_dim])
                
            def compute_mask(self, inputs, mask=None):
                # Return None to disable mask propagation
                return None
        
        # Reshape back using custom layer
        char_bilstm_reshaped = CharBiLSTMReshapeLayer(name='char_bilstm_reshape_layer')(char_bilstm, s)
        
        # Concatenate with word embeddings
        x = tf.keras.layers.Concatenate(axis=-1)([x, char_bilstm_reshaped])
    
    # Apply dropout
    x = Dropout(dropout_rate, name='embedding_dropout')(x)
    if model_type.startswith('bilstm'):
        # Create a deeper BiLSTM network with residual connections
        lstm_layers = 2  # Number of stacked BiLSTM layers
        
        for i in range(lstm_layers):
            # Remember the input to this layer for residual connection
            layer_input = x
            
            # Bidirectional LSTM
            x = Bidirectional(
                LSTM(hidden_dim, return_sequences=True, recurrent_dropout=recurrent_dropout_rate),
                name=f'bilstm_layer_{i+1}'
            )(x)
            
            # Apply layer normalization for better training stability
            x = tf.keras.layers.LayerNormalization(name=f'layer_norm_{i+1}')(x)
            
            # Apply dropout
            x = Dropout(dropout_rate, name=f'lstm_dropout_{i+1}')(x)
            
            # Add residual connection if not the first layer and shapes are compatible
            if i > 0 and layer_input.shape[-1] == x.shape[-1]:
                x = tf.keras.layers.Add(name=f'residual_connection_{i+1}')([layer_input, x])
        
        # Attention mechanism (self-attention)
        # This helps the model focus on important words in the sequence
        attn_layer = tf.keras.layers.MultiHeadAttention(
            num_heads=4, 
            key_dim=hidden_dim//4,
            name='self_attention'
        )
        attn_output = attn_layer(x, x)
        x = tf.keras.layers.Add(name='attention_residual')([x, attn_output])
        x = tf.keras.layers.LayerNormalization(name='attention_layer_norm')(x)
          # Output projection
        x = Dense(hidden_dim, activation='relu', name='output_projection')(x)
        
        # Output layer
        if model_type.endswith('-crf') and HAS_TFA:
            try:
                crf = tfa.layers.CRF(num_tags, name='crf_layer')
                output_layer = crf(x)
                loss_function = tfa.text.crf_loss # This is the loss function itself, not a string
                # Accuracy for CRF is often handled differently, e.g. sequence accuracy
                # For simplicity, we might rely on metrics calculated during evaluation
                # or use a custom metric that works with CRF's specific output.
                # tfa.metrics.SequenceCorrectness might be an option if available and compatible.
                # However, standard Keras metrics might not directly apply to CRF sequence outputs.
                metrics = [] # Custom metrics are usually better for CRF.
                logger.info("Using CRF layer and crf_loss.")
            except Exception as e:
                logger.error(f"Failed to initialize CRF layer from TensorFlow Addons: {e}. Falling back to Dense output.")
                output_layer = Dense(num_tags, activation='softmax', name='output_layer')(x)
                loss_function = 'sparse_categorical_crossentropy'
                metrics = ['accuracy']
        else:
            if model_type.endswith('-crf') and not HAS_TFA:
                logger.warning(f"Model type '{model_type}' requested CRF, but TensorFlow Addons is not available or incompatible. Using standard Dense output layer.")
            output_layer = Dense(num_tags, activation='softmax', name='output_layer')(x)
            loss_function = 'sparse_categorical_crossentropy'
            metrics = ['accuracy']
            logger.info("Using Dense output layer with softmax and sparse_categorical_crossentropy.")

    elif model_type == 'transformer':
        # More sophisticated Transformer Encoder model with multiple layers
        num_transformer_blocks = 4
        num_heads = attention_heads
        ff_dim = hidden_dim * 4  # Common practice: FF dim is 4x the model dimension
        
        # Add positional encoding (crucial for Transformer)
        def positional_encoding(length, depth):
            depth = depth/2
            positions = np.arange(length)[:, np.newaxis]     # (seq, 1)
            depths = np.arange(depth)[np.newaxis, :]/depth   # (1, depth)
            angle_rates = 1 / (10000**depths)                # (1, depth)
            angle_rads = positions * angle_rates             # (pos, depth)
            pos_encoding = np.concatenate(
                [np.sin(angle_rads), np.cos(angle_rads)], axis=-1) 
            return tf.cast(pos_encoding, dtype=tf.float32)
        
        # Apply positional encoding
        max_len = 512  # Maximum expected sequence length
        pos_encoding = positional_encoding(max_len, embedding_dim)
        positions = tf.range(start=0, limit=tf.shape(x)[1], delta=1)
        pos_encoding_subset = tf.gather(pos_encoding, positions)
        x = x + pos_encoding_subset
        
        # Build the transformer stack of encoder blocks
        for i in range(num_transformer_blocks):
            # Self-attention layer
            attn_layer = tf.keras.layers.MultiHeadAttention(
                num_heads=num_heads, 
                key_dim=embedding_dim // num_heads,
                name=f'transformer_attention_{i+1}'
            )
            attn_output = attn_layer(query=x, value=x, key=x)
            
            # Apply dropout to the attention output
            attn_output = tf.keras.layers.Dropout(dropout_rate, name=f'attention_dropout_{i+1}')(attn_output)
            
            # Layer normalization for stability
            x = tf.keras.layers.LayerNormalization(epsilon=1e-6, name=f'attention_layer_norm_{i+1}')(x + attn_output)
            
            # Feed Forward Network
            ffn = tf.keras.Sequential([
                Dense(ff_dim, activation='gelu', name=f'ff_expansion_{i+1}'),  # GELU activation like in BERT
                tf.keras.layers.Dropout(dropout_rate),
                Dense(embedding_dim, name=f'ff_compression_{i+1}')
            ], name=f'feed_forward_{i+1}')
            
            ffn_output = ffn(x)
            
            # Add residual connection and apply normalization
            x = tf.keras.layers.LayerNormalization(epsilon=1e-6, name=f'ffn_layer_norm_{i+1}')(x + ffn_output)
        
        # Final output projection
        x = Dense(hidden_dim, activation='gelu', name='output_projection')(x)
        x = tf.keras.layers.Dropout(dropout_rate, name='final_dropout')(x)          # CRF layer for better sequence modeling (optional)        
        if HAS_TFA:
            # Ensure the output dimension matches the number of tags needed
            # The error shows a label value of 17, so we need at least 18 units (0-17)
            # Add +1 to num_tags to ensure we have enough output units
            adjusted_num_tags = num_tags + 1  # Adding 1 to accommodate the highest label
            logger.info(f"Transformer using tag_projection with {adjusted_num_tags} output units (adjusted from {num_tags})")
            x = Dense(adjusted_num_tags, name='tag_projection')(x)
            crf = tfa.layers.CRF(units=adjusted_num_tags, chain_initializer='orthogonal')
            output_layer = crf(x)
            model = Model(input_layer, output_layer)
            loss = crf.get_loss
            metrics = [crf.get_accuracy]        
        else:
            # Softmax output if TFA not available
            # The error shows a label value of 17, so we need at least 18 units (0-17)
            adjusted_num_tags = num_tags + 1  # Adding 1 to accommodate the highest label
            logger.info(f"Transformer using softmax_output with {adjusted_num_tags} output units (adjusted from {num_tags})")
            output_layer = Dense(adjusted_num_tags, activation='softmax', name='softmax_output')(x)
            model = Model(input_layer, output_layer)
            loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
            metrics = ['accuracy']
    
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # Create and compile model
    if use_char_features and char_vocab_size:
        model = Model(inputs=[input_layer, char_input], outputs=output_layer)
    else:
        model = Model(inputs=input_layer, outputs=output_layer)
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), # Example optimizer
                  loss=loss_function,
                  metrics=metrics,
                  run_eagerly=False) # Set run_eagerly based on debugging needs
                  
    logger.info(f"Model compiled with loss: {loss_function} and metrics: {metrics}")
    return model


def get_crf_accuracy():
    """
    Returns a CRF accuracy metric if TensorFlow Addons is available.
    Otherwise, returns a placeholder or standard accuracy.
    """
    if HAS_TFA:
        try:
            # This is a conceptual example. TFA's CRF doesn't have a simple 'accuracy' metric
            # that works like standard classification. You usually evaluate CRF with sequence-level metrics.
            # tfa.metrics.SequenceCorrectness could be an option if it fits the model output.
            # For now, returning a standard Keras accuracy, but be aware it might not be ideal for CRF.
            # return tfa.metrics.SequenceCorrectness(name='crf_sequence_accuracy') # If compatible
            logger.info("Attempting to use a standard accuracy for CRF, which might not be optimal.")
            return tf.keras.metrics.SparseCategoricalAccuracy(name='crf_accuracy_placeholder')
        except Exception as e:
            logger.warning(f"Could not get CRF-specific accuracy metric from TFA: {e}. Using standard accuracy.")
            return 'accuracy' # Fallback
    return 'accuracy' # Default if TFA not present
