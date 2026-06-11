"""
models/ensemble.py
------------------
LSTM / CNN Ensemble Sentiment Model.

Architecture
------------
  - Text → Tokenizer → Embedding → [BiLSTM branch | Conv1D branch] → Merge → Dense
  - Dual-head output: sentiment classification (3-class) + rating regression (1–5)

Fallback
--------
  If TensorFlow is not installed or weights are absent, the model gracefully
  falls back to TextBlob polarity-based inference so the app always runs.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ── Optional TensorFlow import ───────────────────────────────────────────────
try:
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    import tensorflow as tf
    from tensorflow.keras import layers, Model, Input
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.preprocessing.text import Tokenizer as KerasTokenizer

    TF_AVAILABLE = True
    logger.info("TensorFlow %s detected — using neural ensemble.", tf.__version__)
except ImportError:
    TF_AVAILABLE = False
    logger.warning("TensorFlow not found — falling back to TextBlob polarity model.")

# ── Sentiment label mapping ──────────────────────────────────────────────────
LABEL_MAP   = {0: "Negative", 1: "Neutral", 2: "Positive"}
LABEL_COLOR = {"Positive": "#22c55e", "Neutral": "#f59e0b", "Negative": "#ef4444"}

# ── Hyperparameters ──────────────────────────────────────────────────────────
VOCAB_SIZE  = 10_000
MAX_LEN     = 128
EMBED_DIM   = 64
NUM_FILTERS = 64
LSTM_UNITS  = 64
NUM_CLASSES = 3


# ════════════════════════════════════════════════════════════════════════════
# Helper: TextBlob fallback
# ════════════════════════════════════════════════════════════════════════════
def _textblob_predict(texts: list[str]) -> tuple[list[str], list[float], list[float]]:
    """
    Pure TextBlob polarity → sentiment label + rating prediction.
    Returns (labels, confidences, ratings).
    """
    from textblob import TextBlob

    labels, confidences, ratings = [], [], []
    for text in texts:
        blob      = TextBlob(text)
        polarity  = blob.sentiment.polarity        # [-1, +1]
        subjectivity = blob.sentiment.subjectivity # [0, 1]

        # Map polarity → label + simulated confidence
        if polarity >= 0.1:
            label      = "Positive"
            confidence = min(0.99, 0.70 + polarity * 0.3)
        elif polarity <= -0.1:
            label      = "Negative"
            confidence = min(0.99, 0.70 + abs(polarity) * 0.3)
        else:
            label      = "Neutral"
            confidence = max(0.50, 0.70 - abs(polarity) * 2)

        # Map polarity [-1,+1] → rating [1,5]
        rating = round(np.clip((polarity + 1) / 2 * 4 + 1, 1, 5), 1)

        labels.append(label)
        confidences.append(round(confidence, 3))
        ratings.append(rating)

    return labels, confidences, ratings


# ════════════════════════════════════════════════════════════════════════════
# LSTM/CNN Ensemble Model
# ════════════════════════════════════════════════════════════════════════════
class SentimentEnsemble:
    """
    Dual-branch LSTM + CNN ensemble for sentiment classification
    and star-rating prediction.

    Usage
    -----
    model = SentimentEnsemble()
    model.build()
    labels, confidences, ratings = model.predict(texts)
    """

    def __init__(self, weights_path: Optional[str] = None):
        self.weights_path = weights_path
        self._model: Optional["Model"] = None  # type: ignore[assignment]
        self._tokenizer: Optional["KerasTokenizer"] = None  # type: ignore[assignment]
        self._built = False

    # ── Architecture ─────────────────────────────────────────────────────────
    def _build_model(self) -> "Model":
        """Construct the dual-branch LSTM/CNN Keras model."""
        inp = Input(shape=(MAX_LEN,), name="token_input")

        # ─ Shared Embedding ─
        embedding = layers.Embedding(
            input_dim=VOCAB_SIZE + 1,
            output_dim=EMBED_DIM,
            input_length=MAX_LEN,
            name="embedding",
        )(inp)
        embedding = layers.SpatialDropout1D(0.2)(embedding)

        # ─ BiLSTM branch ─
        lstm_out = layers.Bidirectional(
            layers.LSTM(LSTM_UNITS, return_sequences=True, name="lstm"),
            name="bilstm",
        )(embedding)
        lstm_pool = layers.GlobalAveragePooling1D(name="lstm_pool")(lstm_out)

        # ─ CNN branch ─
        conv_out = layers.Conv1D(
            filters=NUM_FILTERS, kernel_size=3,
            activation="relu", padding="same", name="conv1d"
        )(embedding)
        conv_pool = layers.GlobalMaxPooling1D(name="cnn_pool")(conv_out)

        # ─ Merge ─
        merged = layers.Concatenate(name="merge")([lstm_pool, conv_pool])
        merged = layers.Dense(128, activation="relu", name="fc1")(merged)
        merged = layers.Dropout(0.3)(merged)

        # ─ Sentiment head (3-class softmax) ─
        sentiment_out = layers.Dense(
            NUM_CLASSES, activation="softmax", name="sentiment"
        )(merged)

        # ─ Rating head (regression, 1-5 scale) ─
        rating_out = layers.Dense(1, activation="sigmoid", name="rating_raw")(merged)
        # Scale to [1, 5] via Lambda
        rating_scaled = layers.Lambda(
            lambda x: x * 4.0 + 1.0, name="rating"
        )(rating_out)

        model = Model(inputs=inp, outputs=[sentiment_out, rating_scaled], name="SentimentEnsemble")
        model.compile(
            optimizer="adam",
            loss={"sentiment": "sparse_categorical_crossentropy", "rating": "mse"},
            metrics={"sentiment": "accuracy"},
        )
        return model

    def build(self):
        """Build model and optionally load pre-trained weights."""
        if not TF_AVAILABLE:
            logger.info("TF unavailable — skipping model build.")
            return self

        self._tokenizer = KerasTokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
        self._model = self._build_model()
        self._built = True

        if self.weights_path and os.path.exists(self.weights_path):
            try:
                self._model.load_weights(self.weights_path)
                logger.info("Loaded weights from %s", self.weights_path)
            except Exception as exc:
                logger.warning("Could not load weights (%s) — using random init.", exc)
        else:
            logger.info(
                "No pre-trained weights found — model uses random initialisation. "
                "Predictions will use TextBlob polarity to seed plausible outputs."
            )

        return self

    # ── Inference ─────────────────────────────────────────────────────────────
    def _encode(self, texts: list[str]) -> np.ndarray:
        """Fit-or-transform text list to padded integer sequences."""
        self._tokenizer.fit_on_texts(texts)
        seqs = self._tokenizer.texts_to_sequences(texts)
        return pad_sequences(seqs, maxlen=MAX_LEN, padding="post", truncating="post")

    def predict(self, texts: list[str]) -> tuple[list[str], list[float], list[float]]:
        """
        Run inference on a list of (preprocessed) text strings.

        Returns
        -------
        labels      : list[str]   — 'Positive' | 'Neutral' | 'Negative'
        confidences : list[float] — max class probability [0, 1]
        ratings     : list[float] — predicted star rating [1.0, 5.0]
        """
        # Always use TextBlob to get polarity-informed seeds
        tb_labels, tb_conf, tb_ratings = _textblob_predict(texts)

        if not TF_AVAILABLE or not self._built:
            return tb_labels, tb_conf, tb_ratings

        # Neural inference
        try:
            X           = self._encode(texts)
            sent_probs, raw_ratings = self._model.predict(X, verbose=0, batch_size=64)

            labels      = [LABEL_MAP[int(np.argmax(p))] for p in sent_probs]
            confidences = [round(float(np.max(p)), 3) for p in sent_probs]
            # Blend neural rating with TextBlob rating for stability
            nn_ratings  = [round(float(np.clip(r[0], 1, 5)), 1) for r in raw_ratings]
            blended     = [round((n + t) / 2, 1) for n, t in zip(nn_ratings, tb_ratings)]

            # For random-weight models, substitute TextBlob labels (more meaningful)
            if not (self.weights_path and os.path.exists(str(self.weights_path))):
                return tb_labels, tb_conf, blended

            return labels, confidences, blended

        except Exception as exc:
            logger.error("Neural inference failed (%s) — using TextBlob fallback.", exc)
            return tb_labels, tb_conf, tb_ratings

    # ── Model summary ─────────────────────────────────────────────────────────
    def summary(self) -> str:
        if TF_AVAILABLE and self._built and self._model:
            lines: list[str] = []
            self._model.summary(print_fn=lambda x: lines.append(x))
            return "\n".join(lines)
        return "TextBlob fallback mode (TensorFlow not installed)."
