"""
pipeline.py
-----------
SentimentPipeline — the single entry point for all ETL and inference operations.

Usage
-----
    from pipeline import SentimentPipeline

    pl = SentimentPipeline()
    result_df = pl.run(raw_df, text_column="review_text")
"""

from __future__ import annotations

import logging
import time
from typing import Callable, Optional

import pandas as pd

from utils.preprocessing import preprocess_series
from models.ensemble import SentimentEnsemble

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════════════
# Pipeline
# ════════════════════════════════════════════════════════════════════════════
class SentimentPipeline:
    """
    End-to-end batch sentiment analysis pipeline.

    Stages
    ------
    1. Validate   — check required columns, data types
    2. Clean      — surface text normalisation (HTML, URLs, punctuation …)
    3. NLP        — tokenise → POS tag → lemmatise → remove stopwords
    4. Infer      — LSTM/CNN ensemble → sentiment label + confidence + rating
    5. Enrich     — attach all derived columns back to the original DataFrame

    Parameters
    ----------
    weights_path : str | None
        Path to pre-trained Keras model weights (.h5). If None or not found,
        the model falls back to TextBlob polarity scoring.
    batch_size : int
        Number of rows processed per inference call.
    progress_callback : Callable[[float, str], None] | None
        Optional hook called at each stage with (fraction_done, stage_name).
        Useful for updating Streamlit progress bars.
    """

    # Output column names
    COL_CLEANED    = "cleaned_text"
    COL_PROCESSED  = "processed_text"
    COL_TOKENS     = "tokens"
    COL_LABEL      = "sentiment_label"
    COL_CONFIDENCE = "confidence"
    COL_RATING     = "predicted_rating"

    def __init__(
        self,
        weights_path      : Optional[str] = None,
        batch_size        : int = 256,
        progress_callback : Optional[Callable[[float, str], None]] = None,
    ):
        self.weights_path       = weights_path
        self.batch_size         = batch_size
        self.progress_callback  = progress_callback or (lambda *_: None)

        self._model: Optional[SentimentEnsemble] = None
        self._stats: dict = {}

    # ── Internal helpers ─────────────────────────────────────────────────────
    def _report(self, fraction: float, msg: str):
        logger.info("[%.0f%%] %s", fraction * 100, msg)
        self.progress_callback(fraction, msg)

    def _load_model(self):
        if self._model is None:
            self._report(0.05, "Loading inference model …")
            self._model = SentimentEnsemble(weights_path=self.weights_path).build()

    # ── Stage 1: Validation ──────────────────────────────────────────────────
    def _validate(self, df: pd.DataFrame, text_col: str) -> pd.DataFrame:
        if text_col not in df.columns:
            raise ValueError(
                f"Column '{text_col}' not found. Available columns: {list(df.columns)}"
            )
        df = df.copy()
        df[text_col] = df[text_col].fillna("").astype(str)
        return df

    # ── Stage 2 + 3: Preprocessing ──────────────────────────────────────────
    def _preprocess(self, df: pd.DataFrame, text_col: str) -> pd.DataFrame:
        self._report(0.15, "Running NLP preprocessing pipeline …")
        t0   = time.perf_counter()
        prep = preprocess_series(df[text_col])       # DataFrame: cleaned_text, tokens, processed_text
        elapsed = time.perf_counter() - t0
        logger.info("Preprocessing: %.1f rows/sec", len(df) / max(elapsed, 1e-9))

        df[self.COL_CLEANED]   = prep["cleaned_text"].values
        df[self.COL_TOKENS]    = prep["tokens"].values
        df[self.COL_PROCESSED] = prep["processed_text"].values
        return df

    # ── Stage 4: Batch Inference ─────────────────────────────────────────────
    def _infer(self, df: pd.DataFrame) -> pd.DataFrame:
        self._report(0.40, "Running batch inference (LSTM/CNN ensemble) …")
        texts = df[self.COL_PROCESSED].tolist()

        labels_all, conf_all, rating_all = [], [], []
        n_batches = max(1, (len(texts) + self.batch_size - 1) // self.batch_size)

        for i in range(0, len(texts), self.batch_size):
            batch_num  = i // self.batch_size + 1
            chunk      = texts[i: i + self.batch_size]
            fraction   = 0.40 + 0.50 * (batch_num / n_batches)
            self._report(
                fraction,
                f"Inference — batch {batch_num}/{n_batches} ({len(chunk)} rows) …",
            )

            labels, confs, ratings = self._model.predict(chunk)   # type: ignore[union-attr]
            labels_all .extend(labels)
            conf_all   .extend(confs)
            rating_all .extend(ratings)

        df[self.COL_LABEL]      = labels_all
        df[self.COL_CONFIDENCE] = conf_all
        df[self.COL_RATING]     = rating_all
        return df

    # ── Stage 5: Enrichment ──────────────────────────────────────────────────
    def _enrich(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add derived columns: sentiment_score (numeric) and rating_rounded."""
        score_map = {"Positive": 1, "Neutral": 0, "Negative": -1}
        df["sentiment_score"]  = df[self.COL_LABEL].map(score_map)
        df["rating_rounded"]   = df[self.COL_RATING].round().clip(1, 5).astype(int)
        return df

    # ── Public API ───────────────────────────────────────────────────────────
    def run(self, df: pd.DataFrame, text_column: str = "review_text") -> pd.DataFrame:
        """
        Execute the full pipeline on a DataFrame.

        Parameters
        ----------
        df          : Raw input DataFrame (must contain `text_column`).
        text_column : Name of the column containing raw review text.

        Returns
        -------
        Enriched DataFrame with all derived columns appended.
        """
        t_start = time.perf_counter()
        self._report(0.01, "Validating dataset …")
        df = self._validate(df, text_column)

        self._load_model()

        df = self._preprocess(df, text_column)
        df = self._infer(df)
        df = self._enrich(df)

        elapsed = time.perf_counter() - t_start
        self._stats = {
            "total_rows"     : len(df),
            "elapsed_seconds": round(elapsed, 2),
            "rows_per_second": round(len(df) / max(elapsed, 1e-9), 1),
            "sentiment_counts": df[self.COL_LABEL].value_counts().to_dict(),
            "avg_confidence"  : round(df[self.COL_CONFIDENCE].mean(), 3),
            "avg_rating"      : round(df[self.COL_RATING].mean(), 2),
        }
        self._report(1.0, f"Pipeline complete — {len(df)} rows in {elapsed:.1f}s ✓")
        return df

    @property
    def stats(self) -> dict:
        """Return pipeline run statistics (populated after `run()`)."""
        return self._stats

    def output_columns(self) -> list[str]:
        """List of all columns this pipeline adds to the DataFrame."""
        return [
            self.COL_CLEANED,
            self.COL_TOKENS,
            self.COL_PROCESSED,
            self.COL_LABEL,
            self.COL_CONFIDENCE,
            self.COL_RATING,
            "sentiment_score",
            "rating_rounded",
        ]
