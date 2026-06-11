<div align="center">

# 🧠 RIO-125 — Sentiment Batch Analyser

### Automated Bulk NLP Pipeline for E-Commerce Feedback

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776ab?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![NLTK](https://img.shields.io/badge/NLTK-3.8%2B-4ea94b?style=for-the-badge)](https://www.nltk.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-Optional-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

> **Upload. Process. Analyse. Export.**  
> One CSV in — thousands of reviews classified in seconds.

</div>

---

## 📌 Overview

**RIO-125** is a production-ready Streamlit web application that transforms raw e-commerce customer feedback into structured, actionable sentiment intelligence — **automatically and at scale**.

Originally developed as a Jupyter Notebook prototype, the project has been rebuilt as a fully modular, batch-first application capable of processing hundreds or thousands of reviews in a single run. The pipeline handles everything from raw text to enriched, downloadable results without any manual intervention.

### Why Batch Automation?

Manual, row-by-row analysis doesn't scale. RIO-125 is built around the idea that the **entire workflow should be triggered by a single file upload**:

| Manual Approach | RIO-125 Approach |
|----------------|-----------------|
| Analyse one review at a time | Process entire dataset in one run |
| Manually clean text | Automated ETL pipeline |
| No export workflow | One-click CSV download |
| Static charts | Dynamic, auto-generated insights |
| Requires ML expertise | Zero-configuration interface |

---

## ✨ Key Features

### 🔁 Automated Batch ETL Pipeline
Upload a CSV → the system immediately triggers a 5-stage automated pipeline:
1. **Validation** — schema check and null handling
2. **Text Cleaning** — HTML, URLs, emojis, punctuation stripped
3. **NLP Preprocessing** — tokenisation → POS tagging → lemmatisation → stopword removal
4. **Batch Inference** — LSTM/CNN ensemble classifies every row
5. **Enrichment** — derived columns (score, rounded rating) appended to DataFrame

### 🤖 LSTM / CNN Ensemble Model
- **BiLSTM branch** captures long-range sequential context
- **CNN branch** extracts local n-gram features
- **Dual-head output**: 3-class sentiment + star rating regression
- **Graceful fallback** to TextBlob polarity when TensorFlow is unavailable

### 📊 Automated Visual Insights
Charts are generated automatically on every batch run:
- 🍩 Sentiment distribution donut chart
- ⭐ Predicted rating histogram
- 📦 Sentiment vs. Rating boxplot
- 📈 Model confidence distribution
- 🔢 KPI gauges (avg confidence, avg rating, % positive)
- ☁️ Per-sentiment word clouds

### 📥 One-Click Export
Download the fully enriched dataset as a structured CSV — ready for BI tools, further analysis, or reporting.

---

## 🏗️ Architecture

```
RIO-125/
│
├── app.py                    # Streamlit dashboard (5-stage UI)
├── pipeline.py               # SentimentPipeline orchestrator class
│
├── models/
│   ├── __init__.py
│   └── ensemble.py           # BiLSTM + CNN dual-head Keras model
│
├── utils/
│   ├── __init__.py
│   ├── preprocessing.py      # NLTK tokenise → POS → lemmatise pipeline
│   └── visualizations.py     # Plotly charts + word cloud generators
│
├── sample_data/
│   └── sample_reviews.csv    # 50-row demo dataset
│
├── requirements.txt
├── .gitignore
└── README.md
```

### Data Flow

```
CSV Upload
    │
    ▼
┌─────────────────────────────────────────────────────┐
│                  SentimentPipeline                   │
│                                                      │
│  Validate → Clean → Tokenise → POS Tag → Lemmatise  │
│                           │                          │
│               SentimentEnsemble.predict()            │
│            (BiLSTM ⊕ CNN → softmax + sigmoid)       │
│                           │                          │
│         Enrich DataFrame with derived columns        │
└─────────────────────────────────────────────────────┘
    │
    ▼
Enriched DataFrame + Auto Charts + Download Button
```

---

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/RIO-125.git
cd RIO-125
```

### 2. Create a Virtual Environment

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

> **Optional — Neural Model (GPU/CPU):**
> ```bash
> pip install tensorflow        # GPU
> pip install tensorflow-cpu    # CPU only
> ```
> Without TensorFlow, the app uses TextBlob polarity as a fallback — still fully functional.

### 4. Download NLTK Data (auto-runs on first launch)

NLTK assets are downloaded automatically on the first run. To pre-download manually:

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
```

### 5. Launch the App

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 📂 Input CSV Format

The app accepts any CSV with **at least one text column** containing raw review text.

| Column | Required | Description |
|--------|----------|-------------|
| `review_text` | ✅ Default | Raw customer review text |
| Any other columns | Optional | Preserved in the output |

> 💡 You can configure a different column name in the **sidebar** of the app.

**Sample input row:**
```csv
review_text,product_category
"Absolutely amazing product! Fast delivery and great quality.",Electronics
```

---

## 📤 Output CSV Format

The enriched output adds these columns to your original data:

| Column | Type | Description |
|--------|------|-------------|
| `cleaned_text` | str | Surface-cleaned text (no HTML/URLs/punctuation) |
| `processed_text` | str | Lemmatized tokens joined as a string |
| `tokens` | list | Individual lemma tokens |
| `sentiment_label` | str | `Positive` / `Neutral` / `Negative` |
| `confidence` | float | Model confidence score [0–1] |
| `predicted_rating` | float | Predicted star rating [1.0–5.0] |
| `sentiment_score` | int | Numeric mapping: +1 / 0 / -1 |
| `rating_rounded` | int | Rounded star rating [1–5] |

---

## 🧪 NLP Preprocessing Pipeline

Each review passes through the following stages:

```
Raw Text
    │
    ├─ Strip HTML tags
    ├─ Remove URLs (http/www)
    ├─ Strip emojis (Unicode)
    ├─ Normalise unicode (NFKD → ASCII)
    ├─ Lowercase
    ├─ Remove punctuation
    └─ Collapse whitespace
    │
    ▼
Cleaned Text
    │
    ├─ NLTK word_tokenize
    ├─ Penn Treebank POS tagging
    ├─ WordNet lemmatisation (POS-aware)
    └─ Stopword removal (preserving negations: not, no, never…)
    │
    ▼
Lemmatized Tokens → joined as processed_text
```

**Why POS-aware lemmatisation?**  
`running` tagged as a verb → `run` ✓  
`running` tagged as a noun → `running` ✓  
This avoids incorrect lemmatisation that would distort sentiment signals.

**Why preserve negations?**  
`"not great"` → retaining `not` changes the sentiment vs. stripping it.

---

## 🤖 Model Architecture

```
Input (padded token sequences, max_len=128)
         │
    Embedding (vocab=10K, dim=64)
         │
    SpatialDropout(0.2)
    ┌────┴────┐
    │         │
 BiLSTM     Conv1D
 (64 units)  (64 filters, k=3)
    │         │
 GlobalAvgPool  GlobalMaxPool
    │         │
    └────┬────┘
       Concat
         │
      Dense(128, relu)
      Dropout(0.3)
    ┌────┴────┐
    │         │
 Softmax    Sigmoid → scale [1,5]
 (3-class)  (rating)
```

---

## 📊 Output Examples

### Sentiment Distribution
| Label | Count | % |
|-------|-------|---|
| Positive | 28 | 56% |
| Neutral | 12 | 24% |
| Negative | 10 | 20% |

### Sample Enriched Rows
| review_text | sentiment_label | confidence | predicted_rating |
|-------------|----------------|------------|-----------------|
| "Amazing quality! Fast delivery." | Positive | 0.94 | 4.8 |
| "It's okay, nothing special." | Neutral | 0.71 | 3.1 |
| "Terrible, fell apart after one day." | Negative | 0.89 | 1.4 |

---

## 🛠️ Configuration

All settings are available in the **Streamlit sidebar** — no config files needed:

| Setting | Default | Description |
|---------|---------|-------------|
| Text column name | `review_text` | Column to run the pipeline on |
| Inference batch size | `128` | Rows per model call (trade RAM for speed) |

---

## 🧩 Extending the Pipeline

### Add Pre-trained Weights

```python
pipeline = SentimentPipeline(weights_path="models/weights/ensemble_v1.h5")
```

### Use Programmatically

```python
import pandas as pd
from pipeline import SentimentPipeline

df = pd.read_csv("my_reviews.csv")
pl = SentimentPipeline(batch_size=256)
result = pl.run(df, text_column="review_text")

print(result[["review_text", "sentiment_label", "predicted_rating"]].head())
print(pl.stats)
```

### Progress Callback (for custom UIs)

```python
def my_callback(fraction: float, message: str):
    print(f"[{fraction:.0%}] {message}")

pl = SentimentPipeline(progress_callback=my_callback)
```

---

## 📦 Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `streamlit` | ≥1.35 | Web application framework |
| `pandas` | ≥2.1 | DataFrame operations |
| `numpy` | ≥1.26 | Numerical operations |
| `nltk` | ≥3.8 | Tokenisation, POS tagging, lemmatisation |
| `textblob` | ≥0.18 | Sentiment fallback & polarity scoring |
| `plotly` | ≥5.22 | Interactive charts |
| `wordcloud` | ≥1.9 | Word cloud generation |
| `Pillow` | ≥10.3 | Image processing |
| `scikit-learn` | ≥1.4 | ML utilities |
| `tensorflow` | ≥2.16 | *(Optional)* Neural ensemble model |

---

## 📋 Roadmap

- [ ] Upload model weights via UI
- [ ] Fine-tuning interface with labelled CSV
- [ ] Multilingual preprocessing support
- [ ] REST API endpoint (`/predict`)
- [ ] Docker container for deployment
- [ ] Streamlit Cloud / Hugging Face Spaces deployment

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit changes (`git commit -m 'feat: add your feature'`)
4. Push to branch (`git push origin feature/your-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

---

<div align="center">

Built with ❤️ using **Streamlit** · **NLTK** · **TensorFlow** · **Plotly**

</div>
