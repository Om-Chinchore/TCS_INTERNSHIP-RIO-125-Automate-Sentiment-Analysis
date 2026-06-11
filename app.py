"""
app.py
------
RIO-125 | Sentiment Batch Analyser  — Streamlit Application
============================================================
A fully automated batch NLP dashboard:
  1. Upload a CSV of raw e-commerce reviews
  2. ETL + NLP preprocessing runs automatically
  3. LSTM/CNN ensemble generates sentiment labels + star ratings
  4. Visual insights render automatically
  5. Download the enriched CSV

Run:
    streamlit run app.py
"""

from __future__ import annotations

import io
import logging

import pandas as pd
import streamlit as st

# ── Page config (MUST be first Streamlit call) ──────────────────────────────
st.set_page_config(
    page_title     = "RIO-125 | Sentiment Batch Analyser",
    page_icon      = "🧠",
    layout         = "wide",
    initial_sidebar_state = "expanded",
)

# ── Now import project modules ───────────────────────────────────────────────
from pipeline import SentimentPipeline
from utils.visualizations import (
    sentiment_donut,
    rating_bar,
    sentiment_rating_box,
    confidence_histogram,
    word_cloud_image,
    kpi_gauge_row,
)

logging.basicConfig(level=logging.INFO)

# ════════════════════════════════════════════════════════════════════════════
# Custom CSS — Dark glassmorphism theme
# ════════════════════════════════════════════════════════════════════════════
CUSTOM_CSS = """
<style>
/* ── Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── App background ── */
.stApp {
    background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%);
    background-attachment: fixed;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: rgba(15, 23, 42, 0.85);
    backdrop-filter: blur(16px);
    border-right: 1px solid rgba(99, 102, 241, 0.2);
}

/* ── Cards / containers ── */
.glass-card {
    background: rgba(30, 41, 59, 0.6);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(99, 102, 241, 0.15);
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 16px;
    transition: border-color 0.3s ease;
}
.glass-card:hover {
    border-color: rgba(99, 102, 241, 0.4);
}

/* ── Hero header ── */
.hero-header {
    background: linear-gradient(135deg,
        rgba(99, 102, 241, 0.15) 0%,
        rgba(139, 92, 246, 0.10) 50%,
        rgba(59, 130, 246, 0.10) 100%);
    border: 1px solid rgba(99, 102, 241, 0.25);
    border-radius: 20px;
    padding: 36px 40px;
    margin-bottom: 28px;
    text-align: center;
}
.hero-title {
    font-size: 2.6rem;
    font-weight: 700;
    background: linear-gradient(135deg, #818cf8, #a78bfa, #60a5fa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 8px;
}
.hero-subtitle {
    font-size: 1.05rem;
    color: #94a3b8;
    max-width: 640px;
    margin: 0 auto;
}

/* ── Stage badges ── */
.stage-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: rgba(99, 102, 241, 0.15);
    border: 1px solid rgba(99, 102, 241, 0.35);
    border-radius: 999px;
    padding: 6px 16px;
    font-size: 0.82rem;
    font-weight: 600;
    color: #a5b4fc;
    margin-bottom: 12px;
}

/* ── Metric cards ── */
.metric-row {
    display: flex;
    gap: 12px;
    margin-bottom: 16px;
}
.metric-card {
    flex: 1;
    background: rgba(30, 41, 59, 0.7);
    border: 1px solid rgba(99, 102, 241, 0.2);
    border-radius: 12px;
    padding: 16px;
    text-align: center;
}
.metric-value {
    font-size: 1.8rem;
    font-weight: 700;
    color: #e2e8f0;
}
.metric-label {
    font-size: 0.75rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-top: 4px;
}

/* ── Upload zone ── */
[data-testid="stFileUploader"] {
    border: 2px dashed rgba(99, 102, 241, 0.4) !important;
    border-radius: 14px !important;
    background: rgba(99, 102, 241, 0.05) !important;
    transition: border-color 0.3s;
}
[data-testid="stFileUploader"]:hover {
    border-color: rgba(99, 102, 241, 0.75) !important;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    padding: 10px 24px !important;
    transition: opacity 0.2s, transform 0.15s !important;
}
.stButton > button:hover {
    opacity: 0.88 !important;
    transform: translateY(-1px) !important;
}

/* ── Download button ── */
.stDownloadButton > button {
    background: linear-gradient(135deg, #059669, #10b981) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
}

/* ── Selectbox ── */
.stSelectbox > div > div {
    background: rgba(30, 41, 59, 0.8) !important;
    border-color: rgba(99, 102, 241, 0.3) !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background: transparent;
}
.stTabs [data-baseweb="tab"] {
    background: rgba(30, 41, 59, 0.6);
    border: 1px solid rgba(99, 102, 241, 0.2);
    border-radius: 10px;
    color: #94a3b8;
    font-weight: 500;
    padding: 8px 20px;
}
.stTabs [aria-selected="true"] {
    background: rgba(99, 102, 241, 0.25) !important;
    border-color: rgba(99, 102, 241, 0.5) !important;
    color: #c7d2fe !important;
}

/* ── Dataframe ── */
.stDataFrame {
    border-radius: 12px;
    overflow: hidden;
}

/* ── Progress bar ── */
.stProgress > div > div > div {
    background: linear-gradient(90deg, #6366f1, #8b5cf6, #06b6d4) !important;
    border-radius: 999px !important;
}

/* ── Info / warning boxes ── */
.stInfo {
    background: rgba(99, 102, 241, 0.1) !important;
    border: 1px solid rgba(99, 102, 241, 0.3) !important;
    border-radius: 10px !important;
}

/* ── Section dividers ── */
hr {
    border-color: rgba(99, 102, 241, 0.15) !important;
}

/* ── Sentiment pill helpers ── */
.pill-positive { color: #22c55e; font-weight: 600; }
.pill-neutral  { color: #f59e0b; font-weight: 600; }
.pill-negative { color: #ef4444; font-weight: 600; }

/* ── Sidebar stats ── */
.sidebar-stat {
    display: flex;
    justify-content: space-between;
    padding: 8px 0;
    border-bottom: 1px solid rgba(99,102,241,0.1);
    font-size: 0.85rem;
    color: #94a3b8;
}
.sidebar-stat span { color: #e2e8f0; font-weight: 600; }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# Sidebar
# ════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🧠 RIO-125")
    st.markdown("**Sentiment Batch Analyser**")
    st.markdown("---")

    st.markdown("### ⚙️ Pipeline Configuration")

    text_col_input = st.text_input(
        "Text column name",
        value       = "review_text",
        help        = "The column in your CSV that contains raw review text.",
        key         = "text_col",
    )

    batch_size = st.select_slider(
        "Inference batch size",
        options     = [32, 64, 128, 256, 512],
        value       = 128,
        help        = "Rows processed per model call. Larger = faster but more RAM.",
    )

    st.markdown("---")
    st.markdown("### 📊 Model Info")
    st.markdown("""
    <div style='font-size:0.82rem; color:#94a3b8; line-height:1.7'>
    🔷 <b>Architecture:</b> BiLSTM + CNN Ensemble<br>
    🔷 <b>Heads:</b> 3-class sentiment + rating regression<br>
    🔷 <b>Fallback:</b> TextBlob polarity scoring<br>
    🔷 <b>Preprocessing:</b> NLTK tokenise → POS → lemmatise<br>
    🔷 <b>Labels:</b> Positive / Neutral / Negative
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Runtime stats (populated after pipeline run)
    if "pipeline_stats" in st.session_state:
        s = st.session_state["pipeline_stats"]
        st.markdown("### 📈 Last Run Stats")
        for label, val in [
            ("Rows processed", s.get("total_rows", "—")),
            ("Time (sec)",     s.get("elapsed_seconds", "—")),
            ("Rows / sec",     s.get("rows_per_second", "—")),
            ("Avg confidence", s.get("avg_confidence", "—")),
            ("Avg rating",     s.get("avg_rating", "—")),
        ]:
            st.markdown(
                f"<div class='sidebar-stat'>{label}<span>{val}</span></div>",
                unsafe_allow_html=True,
            )

    st.markdown("---")
    st.caption("RIO-125 · v1.0 · Built with Streamlit")


# ════════════════════════════════════════════════════════════════════════════
# Hero Header
# ════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero-header">
    <div class="hero-title">🧠 Sentiment Batch Analyser</div>
    <p class="hero-subtitle">
        Upload a CSV of e-commerce reviews and the pipeline will automatically
        clean, preprocess, and classify every row — delivering enriched data
        and visual insights in seconds.
    </p>
</div>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# Stage 1 — File Upload
# ════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="stage-badge">📂 Stage 1 — Upload Dataset</div>', unsafe_allow_html=True)

col_upload, col_sample = st.columns([3, 1])

with col_upload:
    uploaded_file = st.file_uploader(
        "Drop your CSV file here, or click to browse",
        type        = ["csv"],
        help        = "CSV must contain a text column with raw review/feedback text.",
        key         = "csv_upload",
    )

with col_sample:
    st.markdown("<br>", unsafe_allow_html=True)
    with open("sample_data/sample_reviews.csv", "rb") as f:
        st.download_button(
            label       = "⬇️ Download sample CSV",
            data        = f,
            file_name   = "sample_reviews.csv",
            mime        = "text/csv",
            key         = "dl_sample",
            help        = "No CSV? Download our 50-row demo file.",
        )


# ── Load or use uploaded file ────────────────────────────────────────────────
raw_df: pd.DataFrame | None = None

if uploaded_file is not None:
    try:
        raw_df = pd.read_csv(uploaded_file)
        st.success(f"✅ Loaded **{len(raw_df):,} rows** × **{len(raw_df.columns)} columns**")

        # Column preview
        with st.expander("👁️ Preview raw data", expanded=False):
            st.dataframe(raw_df.head(10), use_container_width=True, height=250)

        # Column mapping
        available_cols = list(raw_df.columns)
        if text_col_input not in available_cols:
            st.warning(
                f"Column **'{text_col_input}'** not found. Select the correct column below."
            )
            text_col_input = st.selectbox(
                "Select the text column",
                options = available_cols,
                key     = "col_select",
            )

    except Exception as e:
        st.error(f"❌ Failed to read CSV: {e}")
        raw_df = None


# ════════════════════════════════════════════════════════════════════════════
# Stage 2 — ETL + NLP Preprocessing  &  Stage 3 — Batch Inference
# ════════════════════════════════════════════════════════════════════════════
result_df: pd.DataFrame | None = st.session_state.get("result_df", None)

if raw_df is not None and result_df is None:
    st.markdown("---")
    st.markdown(
        '<div class="stage-badge">⚡ Stage 2 & 3 — Automated ETL · NLP · Inference</div>',
        unsafe_allow_html=True,
    )

    progress_bar  = st.progress(0, text="Initialising pipeline …")
    status_text   = st.empty()

    def _cb(fraction: float, msg: str):
        progress_bar.progress(min(fraction, 1.0), text=msg)
        status_text.markdown(
            f"<span style='color:#94a3b8;font-size:0.85rem'>⟳ {msg}</span>",
            unsafe_allow_html=True,
        )

    try:
        pipeline = SentimentPipeline(
            batch_size        = batch_size,
            progress_callback = _cb,
        )
        result_df = pipeline.run(raw_df.copy(), text_column=text_col_input)
        st.session_state["result_df"]      = result_df
        st.session_state["pipeline_stats"] = pipeline.stats

        progress_bar.progress(1.0, text="✅ Pipeline complete!")
        status_text.empty()
        st.balloons()

    except Exception as exc:
        progress_bar.empty()
        status_text.empty()
        st.error(f"❌ Pipeline error: {exc}")
        st.exception(exc)
        result_df = None

elif raw_df is None and result_df is not None:
    # File was removed — clear state
    st.session_state.pop("result_df", None)
    st.session_state.pop("pipeline_stats", None)
    result_df = None


# ════════════════════════════════════════════════════════════════════════════
# Stage 4 — Results & Insights
# ════════════════════════════════════════════════════════════════════════════
if result_df is not None:
    st.markdown("---")
    st.markdown(
        '<div class="stage-badge">📊 Stage 4 — Results & Visual Insights</div>',
        unsafe_allow_html=True,
    )

    # ── KPI Gauges ───────────────────────────────────────────────────────────
    st.plotly_chart(kpi_gauge_row(result_df), use_container_width=True)

    # ── Summary metrics ───────────────────────────────────────────────────────
    counts     = result_df["sentiment_label"].value_counts()
    pos_count  = counts.get("Positive", 0)
    neu_count  = counts.get("Neutral",  0)
    neg_count  = counts.get("Negative", 0)
    total      = len(result_df)

    c1, c2, c3, c4 = st.columns(4)
    for col, label, value, color in [
        (c1, "Total Reviews",  f"{total:,}",                  "#6366f1"),
        (c2, "✅ Positive",    f"{pos_count:,} ({pos_count/total*100:.0f}%)", "#22c55e"),
        (c3, "⚠️ Neutral",    f"{neu_count:,} ({neu_count/total*100:.0f}%)", "#f59e0b"),
        (c4, "❌ Negative",   f"{neg_count:,} ({neg_count/total*100:.0f}%)", "#ef4444"),
    ]:
        col.markdown(
            f"""<div class="glass-card" style="text-align:center;">
                    <div style="font-size:1.6rem;font-weight:700;color:{color}">{value}</div>
                    <div style="font-size:0.78rem;color:#64748b;margin-top:4px">{label}</div>
                </div>""",
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ── Tabs: Data | Charts | Word Clouds ────────────────────────────────────
    tab_data, tab_charts, tab_wc = st.tabs(
        ["📋 Processed Data", "📈 Visual Insights", "☁️ Word Clouds"]
    )

    # ── Tab 1: Data ──────────────────────────────────────────────────────────
    with tab_data:
        st.markdown("#### 🗃️ Enriched Dataset Sample")

        display_cols = [
            text_col_input,
            "cleaned_text",
            "sentiment_label",
            "confidence",
            "predicted_rating",
            "rating_rounded",
            "sentiment_score",
        ]
        display_cols = [c for c in display_cols if c in result_df.columns]

        filter_sentiment = st.multiselect(
            "Filter by sentiment",
            options  = ["Positive", "Neutral", "Negative"],
            default  = ["Positive", "Neutral", "Negative"],
            key      = "filter_sent",
        )
        filtered = result_df[result_df["sentiment_label"].isin(filter_sentiment)]

        st.dataframe(
            filtered[display_cols].head(200),
            use_container_width = True,
            height              = 420,
        )
        st.caption(f"Showing up to 200 of {len(filtered):,} filtered rows.")

    # ── Tab 2: Charts ─────────────────────────────────────────────────────────
    with tab_charts:
        r1c1, r1c2 = st.columns(2)
        with r1c1:
            st.plotly_chart(sentiment_donut(result_df),       use_container_width=True)
        with r1c2:
            st.plotly_chart(rating_bar(result_df),            use_container_width=True)

        st.plotly_chart(sentiment_rating_box(result_df),      use_container_width=True)
        st.plotly_chart(confidence_histogram(result_df),      use_container_width=True)

    # ── Tab 3: Word Clouds ────────────────────────────────────────────────────
    with tab_wc:
        wc_col = st.selectbox(
            "Generate word cloud for",
            options = ["All Reviews", "Positive", "Neutral", "Negative"],
            key     = "wc_select",
        )
        sentiment_filter = None if wc_col == "All Reviews" else wc_col

        with st.spinner("Generating word cloud …"):
            wc_bytes = word_cloud_image(result_df, sentiment=sentiment_filter)

        if wc_bytes:
            st.image(wc_bytes, caption=f"Word Cloud — {wc_col}", use_container_width=True)
        else:
            st.info(
                "Word cloud requires the `wordcloud` library. "
                "Run `pip install wordcloud` and restart."
            )

    # ════════════════════════════════════════════════════════════════════════
    # Stage 5 — Export
    # ════════════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.markdown(
        '<div class="stage-badge">⬇️ Stage 5 — Export Enriched Dataset</div>',
        unsafe_allow_html=True,
    )

    csv_buffer = io.StringIO()
    result_df.to_csv(csv_buffer, index=False)
    csv_bytes  = csv_buffer.getvalue().encode("utf-8")

    dl_col, info_col = st.columns([1, 3])
    with dl_col:
        st.download_button(
            label     = "⬇️ Download Enriched CSV",
            data      = csv_bytes,
            file_name = "sentiment_results.csv",
            mime      = "text/csv",
            key       = "dl_results",
        )
    with info_col:
        st.markdown(
            f"""<div style='color:#94a3b8;font-size:0.88rem;padding-top:10px'>
                The exported file includes all original columns plus:
                <code>cleaned_text</code>, <code>processed_text</code>,
                <code>tokens</code>, <code>sentiment_label</code>,
                <code>confidence</code>, <code>predicted_rating</code>,
                <code>sentiment_score</code>, <code>rating_rounded</code>.
            </div>""",
            unsafe_allow_html=True,
        )

# ── Empty state ──────────────────────────────────────────────────────────────
elif raw_df is None:
    st.markdown("---")
    st.markdown("""
    <div style='text-align:center; padding:60px 20px; color:#475569;'>
        <div style='font-size:3.5rem'>📁</div>
        <div style='font-size:1.2rem; font-weight:600; color:#64748b; margin-top:12px'>
            Awaiting your dataset
        </div>
        <div style='font-size:0.9rem; margin-top:8px; max-width:420px; margin-left:auto; margin-right:auto;'>
            Upload a CSV file above to trigger the automated sentiment analysis pipeline.
            Use the sample CSV to see a live demo.
        </div>
    </div>
    """, unsafe_allow_html=True)
