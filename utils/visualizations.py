"""
utils/visualizations.py
-----------------------
Chart generation utilities for the Streamlit dashboard.
All charts return Plotly figures or PIL Images — never render directly.
"""

from __future__ import annotations

import io
import logging
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)

# ── Shared colour palette ────────────────────────────────────────────────────
SENTIMENT_COLORS = {
    "Positive": "#22c55e",
    "Neutral" : "#f59e0b",
    "Negative": "#ef4444",
}

CHART_TEMPLATE = "plotly_dark"
FONT_FAMILY    = "Inter, Segoe UI, sans-serif"

_BASE_LAYOUT = dict(
    template    = CHART_TEMPLATE,
    font        = dict(family=FONT_FAMILY, size=13, color="#e2e8f0"),
    paper_bgcolor = "rgba(15,23,42,0)",
    plot_bgcolor  = "rgba(15,23,42,0)",
    margin        = dict(l=40, r=20, t=50, b=40),
)


# ════════════════════════════════════════════════════════════════════════════
# 1. Sentiment Distribution — Donut Chart
# ════════════════════════════════════════════════════════════════════════════
def sentiment_donut(df: pd.DataFrame, sentiment_col: str = "sentiment_label") -> go.Figure:
    """Animated donut chart of sentiment label distribution."""
    counts = df[sentiment_col].value_counts().reset_index()
    counts.columns = ["Sentiment", "Count"]

    fig = px.pie(
        counts,
        names   = "Sentiment",
        values  = "Count",
        hole    = 0.55,
        color   = "Sentiment",
        color_discrete_map = SENTIMENT_COLORS,
        title   = "Sentiment Distribution",
    )
    fig.update_traces(
        textinfo      = "percent+label",
        textfont_size = 13,
        pull          = [0.03] * len(counts),
        marker        = dict(line=dict(color="#0f172a", width=2)),
    )
    fig.update_layout(
        **_BASE_LAYOUT,
        showlegend   = True,
        legend       = dict(orientation="h", yanchor="bottom", y=-0.15, x=0.5, xanchor="center"),
        annotations  = [dict(
            text      = f"<b>{len(df)}</b><br>reviews",
            x=0.5, y=0.5,
            font_size = 16,
            font_color= "#e2e8f0",
            showarrow = False,
        )],
    )
    return fig


# ════════════════════════════════════════════════════════════════════════════
# 2. Rating Distribution — Horizontal Bar
# ════════════════════════════════════════════════════════════════════════════
def rating_bar(df: pd.DataFrame, rating_col: str = "predicted_rating") -> go.Figure:
    """Horizontal bar chart of predicted star rating distribution."""
    df_copy = df.copy()
    df_copy["star_bin"] = df_copy[rating_col].round().astype(int).clip(1, 5)
    counts = df_copy["star_bin"].value_counts().sort_index().reset_index()
    counts.columns = ["Stars", "Count"]

    star_labels = [f"{'★' * s} ({s})" for s in counts["Stars"]]
    colors      = px.colors.sequential.Viridis_r[:len(counts)]

    fig = go.Figure(go.Bar(
        x            = counts["Count"],
        y            = star_labels,
        orientation  = "h",
        marker_color = colors,
        text         = counts["Count"],
        textposition = "outside",
        textfont     = dict(color="#e2e8f0"),
    ))
    fig.update_layout(
        **_BASE_LAYOUT,
        title  = "Predicted Rating Distribution",
        xaxis  = dict(title="Number of Reviews", gridcolor="rgba(255,255,255,0.07)"),
        yaxis  = dict(title="Star Rating",       gridcolor="rgba(255,255,255,0.07)"),
    )
    return fig


# ════════════════════════════════════════════════════════════════════════════
# 3. Sentiment vs. Rating — Box Plot
# ════════════════════════════════════════════════════════════════════════════
def sentiment_rating_box(
    df: pd.DataFrame,
    sentiment_col: str = "sentiment_label",
    rating_col   : str = "predicted_rating",
) -> go.Figure:
    """Box plot showing rating spread per sentiment category."""
    fig = px.box(
        df,
        x      = sentiment_col,
        y      = rating_col,
        color  = sentiment_col,
        color_discrete_map = SENTIMENT_COLORS,
        points = "outliers",
        title  = "Rating Distribution by Sentiment",
        labels = {sentiment_col: "Sentiment", rating_col: "Predicted Rating"},
        category_orders = {sentiment_col: ["Negative", "Neutral", "Positive"]},
    )
    fig.update_traces(
        marker = dict(opacity=0.6, size=5),
        line   = dict(width=2),
    )
    fig.update_layout(
        **_BASE_LAYOUT,
        showlegend = False,
        yaxis      = dict(range=[0.5, 5.5], dtick=1, gridcolor="rgba(255,255,255,0.07)"),
        xaxis      = dict(gridcolor="rgba(255,255,255,0.07)"),
    )
    return fig


# ════════════════════════════════════════════════════════════════════════════
# 4. Confidence Score Distribution — Histogram
# ════════════════════════════════════════════════════════════════════════════
def confidence_histogram(
    df: pd.DataFrame,
    confidence_col: str = "confidence",
    sentiment_col : str = "sentiment_label",
) -> go.Figure:
    """Overlaid histograms of confidence scores per sentiment."""
    fig = px.histogram(
        df,
        x      = confidence_col,
        color  = sentiment_col,
        color_discrete_map = SENTIMENT_COLORS,
        nbins  = 20,
        barmode = "overlay",
        opacity = 0.75,
        title   = "Model Confidence Score Distribution",
        labels  = {confidence_col: "Confidence", sentiment_col: "Sentiment"},
    )
    fig.update_layout(
        **_BASE_LAYOUT,
        xaxis = dict(range=[0, 1], gridcolor="rgba(255,255,255,0.07)"),
        yaxis = dict(title="Count", gridcolor="rgba(255,255,255,0.07)"),
        legend = dict(orientation="h", yanchor="bottom", y=-0.25, x=0.5, xanchor="center"),
    )
    return fig


# ════════════════════════════════════════════════════════════════════════════
# 5. Word Cloud — Returns PIL Image bytes
# ════════════════════════════════════════════════════════════════════════════
def word_cloud_image(
    df          : pd.DataFrame,
    text_col    : str = "processed_text",
    sentiment   : Optional[str] = None,
    max_words   : int = 150,
    width       : int = 800,
    height      : int = 400,
) -> Optional[bytes]:
    """
    Generate a word cloud from processed text.
    Returns PNG bytes or None if wordcloud is not installed.
    """
    try:
        from wordcloud import WordCloud, STOPWORDS
    except ImportError:
        logger.warning("wordcloud not installed — skipping word cloud.")
        return None

    subset = df[df["sentiment_label"] == sentiment] if sentiment else df
    if subset.empty or text_col not in subset.columns:
        return None

    corpus = " ".join(subset[text_col].dropna().tolist())
    if not corpus.strip():
        return None

    colour_fn = {
        "Positive": lambda *a, **k: "hsl(142, 70%%, %d%%)" % np.random.randint(40, 70),
        "Negative": lambda *a, **k: "hsl(0, 80%%, %d%%)"   % np.random.randint(40, 65),
        "Neutral" : lambda *a, **k: "hsl(40, 85%%, %d%%)"  % np.random.randint(45, 70),
    }
    color_func = colour_fn.get(sentiment or "Neutral")  # type: ignore[arg-type]

    wc = WordCloud(
        width            = width,
        height           = height,
        background_color = "#0f172a",
        max_words        = max_words,
        color_func       = color_func,
        collocations     = False,
        prefer_horizontal= 0.8,
    ).generate(corpus)

    buf = io.BytesIO()
    wc.to_image().save(buf, format="PNG")
    return buf.getvalue()


# ════════════════════════════════════════════════════════════════════════════
# 6. Batch Overview KPI Gauges — Plotly
# ════════════════════════════════════════════════════════════════════════════
def kpi_gauge_row(df: pd.DataFrame) -> go.Figure:
    """Three KPI gauges: avg confidence, avg rating, % positive."""
    avg_conf    = df["confidence"].mean()         if "confidence"      in df.columns else 0.0
    avg_rating  = df["predicted_rating"].mean()   if "predicted_rating" in df.columns else 0.0
    pct_positive = (
        (df["sentiment_label"] == "Positive").sum() / len(df) * 100
        if "sentiment_label" in df.columns else 0.0
    )

    fig = make_subplots(
        rows=1, cols=3,
        specs=[[{"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"}]],
    )

    def _gauge(value, title, max_val, suffix, bar_color, row, col):
        fig.add_trace(go.Indicator(
            mode  = "gauge+number",
            value = value,
            title = {"text": title, "font": {"size": 13, "color": "#94a3b8"}},
            number = {"suffix": suffix, "font": {"size": 22, "color": "#e2e8f0"}},
            gauge = {
                "axis"     : {"range": [0, max_val], "tickcolor": "#475569"},
                "bar"      : {"color": bar_color},
                "bgcolor"  : "#1e293b",
                "borderwidth": 0,
                "steps"    : [{"range": [0, max_val], "color": "#0f172a"}],
            },
        ), row=row, col=col)

    _gauge(avg_conf * 100,  "Avg Confidence", 100, "%",  "#6366f1", 1, 1)
    _gauge(avg_rating,      "Avg Rating",     5,   "★",  "#f59e0b", 1, 2)
    _gauge(pct_positive,    "% Positive",     100, "%",  "#22c55e", 1, 3)

    gauge_layout = {
        **_BASE_LAYOUT,
        "height": 200,
        "margin": dict(l=20, r=20, t=30, b=10),
    }
    fig.update_layout(**gauge_layout)
    return fig
