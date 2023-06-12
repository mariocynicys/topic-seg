"""Code for the Two Level Transformer creation."""

import config
import encoder
import numpy as np
import tensorflow as tf
from tensorflow import keras


if 0:
    # For some reason, VScode's intellisense thinks that keras lives inside keras.keras.
    # This code is never gonna execute, but will help VScode's intellisense recognize stuff.
    keras = keras.keras



def TwoLevelTransformerModel(weights=None, word_repr=None):
    if word_repr is None:
        assert weights is not None, "You must pass either `word_repr` or `weights` for the model to be initialized with the right dimensions."
        word_emb_layer = weights[2].shape
        # This is just a place holder, the weights are gonna be overwritten later.
        word_repr = np.array([np.zeros(word_emb_layer[1])] * word_emb_layer[0])

    inputs = keras.Input((config.snippet_length, config.segment_length), config.batch_size)

    # Word embeddings. This layer is not trainable, since we use pre-trained word embeddings.
    word_embeddings = keras.layers.Embedding(
        word_repr.shape[0], word_repr.shape[1],
        embeddings_initializer=keras.initializers.Constant(word_repr),
        trainable=False
    )(inputs)

    # If we provide the nominal positions right away, we will end up with a disconnected network.
    # That's why we have to pass the input through a lambda layer just to wire up the graph correctly.
    token_nominals = keras.layers.Lambda(
        lambda _: tf.constant(
            np.tile(np.arange(config.segment_length), [config.batch_size, config.snippet_length, 1])
        )
    )(inputs)
    sentence_nominals = keras.layers.Lambda(
        lambda _: tf.constant(
            np.tile(
                np.tile(np.arange(config.snippet_length), [config.segment_length, 1]).T,
                [config.batch_size, 1, 1]
            )
        )
    )(inputs)

    # Positional embeddings on sentence/segment level (the word's position in a sentence).
    segment_pos_embeddings = keras.layers.Embedding(
        config.segment_length, config.positional_embeddings_length // 2
    )(token_nominals)

    # Positional embeddings on a snippet/paragraph level (the word's position in a snippet).
    snippet_pos_embeddings = keras.layers.Embedding(
        config.snippet_length, config.positional_embeddings_length // 2
    )(sentence_nominals)

    # Combine all of these to get an embeddings vector for each token
    token_embeddings = tf.concat([snippet_pos_embeddings, segment_pos_embeddings, word_embeddings], axis=3)

    # Reshape the token embeddings to suppress `config.snippet_length` dimension, this is because
    # the token transformer processes the input in segment level and doesn't know about snippets.
    # NOTE: We could avoid reshaping if we were to specify `attention_axes=2` for the encoder stack,
    # but this makes the pipeline 3x slower for some reason.
    token_embeddings = tf.reshape(token_embeddings, [config.batch_size * config.snippet_length, config.segment_length, -1])
    transformed_token_embeddings = encoder.EncoderStack(**config.token_transformer_params)(token_embeddings)

    # Pick the first transformed embeddings as the sentence embeddings.
    sentence_representations = transformed_token_embeddings[:, 0, :]

    # And then reshape back to account for `config.snippet_length`.
    sentence_representations = tf.reshape(sentence_representations, [config.batch_size, config.snippet_length, -1])

    # Transform the sentence embeddings.
    transformed_sentence_representations = encoder.EncoderStack(**config.sentence_transformer_params)(sentence_representations)

    # Predict whether each of these sentence representations are a sentence boundary or not.
    classifications = keras.layers.Dense(1, "sigmoid")(transformed_sentence_representations)
    classifications = tf.squeeze(classifications)

    model = keras.Model(inputs, classifications)
    model.summary()
    model.compile(
        optimizer=keras.optimizers.Adam(config.learning_rate),
        loss="binary_crossentropy",
        metrics=[keras.metrics.BinaryAccuracy()]
    )

    # Overwrite with these weights if provided.
    if weights is not None:
        model.set_weights(weights)
    return model
