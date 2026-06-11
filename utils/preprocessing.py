"""
utils/preprocessing.py
----------------------
Full NLP preprocessing pipeline for e-commerce review text.
Handles: HTML stripping, URL/emoji removal, tokenization,
POS tagging, lemmatization, and stopword removal.
"""

import re
import string
import unicodedata
import logging

import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk import pos_tag

logger = logging.getLogger(__name__)

# ── Download required NLTK assets on first import ──────────────────────────
def _ensure_nltk_data():
    assets = [
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("corpora/stopwords", "stopwords"),
        ("corpora/wordnet", "wordnet"),
        ("taggers/averaged_perceptron_tagger", "averaged_perceptron_tagger"),
        ("taggers/averaged_perceptron_tagger_eng", "averaged_perceptron_tagger_eng"),
    ]
    for path, pkg in assets:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(pkg, quiet=True)

_ensure_nltk_data()

# ── Constants ───────────────────────────────────────────────────────────────
STOP_WORDS = set(stopwords.words("english"))
# Keep negations — important for sentiment
NEGATIONS = {"no", "not", "nor", "never", "neither", "hardly", "barely", "scarcely"}
STOP_WORDS -= NEGATIONS

LEMMATIZER = WordNetLemmatizer()

# Regex patterns
_URL_RE    = re.compile(r"https?://\S+|www\.\S+")
_HTML_RE   = re.compile(r"<[^>]+>")
_EMOJI_RE  = re.compile(
    "["
    u"\U0001F600-\U0001F64F"
    u"\U0001F300-\U0001F5FF"
    u"\U0001F680-\U0001F9FF"
    u"\U00002700-\U000027BF"
    u"\U0001FA00-\U0001FA6F"
    "]+",
    flags=re.UNICODE,
)
_PUNCT_RE   = re.compile(r"[%s]" % re.escape(string.punctuation))
_WHITESPACE_RE = re.compile(r"\s+")


# ── POS tag → WordNet tag mapping ───────────────────────────────────────────
def _wordnet_pos(treebank_tag: str) -> str:
    """Map Penn Treebank POS tag to WordNet POS constant."""
    if treebank_tag.startswith("J"):
        return wordnet.ADJ
    elif treebank_tag.startswith("V"):
        return wordnet.VERB
    elif treebank_tag.startswith("N"):
        return wordnet.NOUN
    elif treebank_tag.startswith("R"):
        return wordnet.ADV
    else:
        return wordnet.NOUN  # default


# ── Individual cleaning helpers ─────────────────────────────────────────────
def strip_html(text: str) -> str:
    return _HTML_RE.sub(" ", text)


def strip_urls(text: str) -> str:
    return _URL_RE.sub(" ", text)


def strip_emojis(text: str) -> str:
    return _EMOJI_RE.sub(" ", text)


def normalize_unicode(text: str) -> str:
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")


def strip_punctuation(text: str) -> str:
    return _PUNCT_RE.sub(" ", text)


def collapse_whitespace(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", text).strip()


# ── Core pipeline functions ─────────────────────────────────────────────────
def clean_text(raw_text: str) -> str:
    """
    Stage 1 — Surface cleaning:
    HTML → URLs → emojis → unicode → lowercase → punctuation → whitespace
    """
    if not isinstance(raw_text, str) or not raw_text.strip():
        return ""
    text = strip_html(raw_text)
    text = strip_urls(text)
    text = strip_emojis(text)
    text = normalize_unicode(text)
    text = text.lower()
    text = strip_punctuation(text)
    text = collapse_whitespace(text)
    return text


def tokenize_and_lemmatize(cleaned_text: str) -> list[str]:
    """
    Stage 2 — Linguistic normalization:
    tokenize → POS tag → lemmatize → remove stopwords & short tokens
    Returns list of meaningful lemmas.
    """
    if not cleaned_text:
        return []

    tokens = word_tokenize(cleaned_text)
    pos_tags = pos_tag(tokens)                         # [('great', 'JJ'), ...]

    lemmas = []
    for token, tag in pos_tags:
        wn_pos = _wordnet_pos(tag)
        lemma  = LEMMATIZER.lemmatize(token, pos=wn_pos)
        if lemma not in STOP_WORDS and len(lemma) > 2:
            lemmas.append(lemma)

    return lemmas


def preprocess(raw_text: str) -> dict:
    """
    Full preprocessing for a single review.

    Returns
    -------
    dict with keys:
        cleaned_text  : str  — surface-cleaned text (Stage 1 output)
        tokens        : list — lemmatized tokens (Stage 2 output)
        processed_text: str  — tokens joined as a string (model-ready)
    """
    cleaned = clean_text(raw_text)
    tokens  = tokenize_and_lemmatize(cleaned)
    return {
        "cleaned_text"  : cleaned,
        "tokens"        : tokens,
        "processed_text": " ".join(tokens),
    }


def preprocess_series(series) -> "pd.DataFrame":
    """
    Vectorized preprocessing for a pandas Series.
    Returns a DataFrame with columns: cleaned_text, tokens, processed_text.
    """
    import pandas as pd
    results = series.fillna("").apply(preprocess)
    return pd.DataFrame(list(results))
