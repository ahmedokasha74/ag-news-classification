"""Small Keras neural network for AG News classification."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report


def prepare_sequences(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    text_column: str = "clean_text",
    num_words: int = 10000,
    max_len: int = 100,
):
    """Tokenize text and return padded arrays for TensorFlow training."""
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.preprocessing.text import Tokenizer

    tokenizer = Tokenizer(num_words=num_words, oov_token="<OOV>")
    tokenizer.fit_on_texts(train_df[text_column])
    x_train = pad_sequences(tokenizer.texts_to_sequences(train_df[text_column]), maxlen=max_len)
    x_test = pad_sequences(tokenizer.texts_to_sequences(test_df[text_column]), maxlen=max_len)
    return tokenizer, x_train, x_test


def build_dl_model(num_words: int = 10000, embedding_dim: int = 128, num_classes: int = 4):
    """Create a compact neural text classifier."""
    import tensorflow as tf

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Embedding(num_words, embedding_dim),
            tf.keras.layers.GlobalAveragePooling1D(),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_dl_model(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    epochs: int = 5,
    batch_size: int = 128,
    num_words: int = 10000,
    max_len: int = 100,
):
    """Train the Keras model and return model, tokenizer, and test arrays."""
    tokenizer, x_train, x_test = prepare_sequences(train_df, test_df, num_words=num_words, max_len=max_len)
    y_train = train_df["label"].to_numpy() - 1
    model = build_dl_model(num_words=num_words)
    history = model.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
    )
    return model, tokenizer, x_test, history


def evaluate_dl_model(model, x_test, test_df: pd.DataFrame) -> dict:
    """Evaluate the Keras classifier."""
    y_test = test_df["label"].to_numpy() - 1
    predictions = np.argmax(model.predict(x_test), axis=1)
    return {
        "accuracy": accuracy_score(y_test, predictions),
        "classification_report": classification_report(y_test, predictions),
    }

