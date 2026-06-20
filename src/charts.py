"""
Chart generation and column detection.

Generates Plotly charts from DataFrames with intelligent column detection.
"""

import logging
from typing import Optional, Tuple

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

logger = logging.getLogger("financial_analyst.charts")

# ══════════════════════════════════════════════════════
# CHART THEME — Consistent premium look
# ══════════════════════════════════════════════════════

CHART_TEMPLATE = "plotly_dark"
CHART_COLOR_SEQUENCE = [
    "#6366f1", "#8b5cf6", "#a78bfa", "#c4b5fd",  # Indigo → Violet
    "#34d399", "#2dd4bf", "#22d3ee", "#38bdf8",  # Emerald → Sky
    "#f472b6", "#fb923c", "#facc15", "#a3e635",  # Pink → Lime
]


def detect_chart_type(prompt: str) -> str:
    """
    Detect the chart type from the user's prompt.

    Args:
        prompt: The user's natural-language request.

    Returns:
        One of: 'bar', 'line', 'pie', 'scatter', 'histogram'.
    """
    q = prompt.lower()

    if any(w in q for w in ["line", "trend", "over time", "time series", "growth"]):
        return "line"
    elif any(w in q for w in ["pie", "distribution", "share", "proportion", "percentage breakdown"]):
        return "pie"
    elif any(w in q for w in ["scatter", "correlation", "relationship", "vs", "versus"]):
        return "scatter"
    elif any(w in q for w in ["histogram", "frequency", "bin"]):
        return "histogram"
    else:
        return "bar"


def detect_columns(prompt: str, df: pd.DataFrame) -> Tuple[str, str]:
    """
    Detect the best x and y columns from a natural-language prompt.

    Strategy:
    1. Look for column names explicitly mentioned in the prompt.
    2. Assign numeric columns to y-axis, text columns to x-axis.
    3. Fall back to first text/numeric column pair.

    Args:
        prompt: The user's question.
        df: The DataFrame to chart.

    Returns:
        Tuple of (x_column, y_column).
    """
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    text_cols = df.select_dtypes(include="object").columns.tolist()
    all_cols = df.columns.tolist()

    prompt_lower = prompt.lower()

    # Try to find columns explicitly mentioned in the question
    mentioned_cols = [col for col in all_cols if col.lower() in prompt_lower]

    x_col = None
    y_col = None

    # Assign mentioned columns to x or y based on type
    for col in mentioned_cols:
        if col in numeric_cols and y_col is None:
            y_col = col
        elif col not in numeric_cols and x_col is None:
            x_col = col

    # If we found a numeric mention but no x, and also a text mention but no y — swap isn't needed
    # Just fill in defaults for whatever is missing
    if x_col is None:
        x_col = text_cols[0] if text_cols else all_cols[0]
    if y_col is None:
        y_col = numeric_cols[0] if numeric_cols else (all_cols[1] if len(all_cols) > 1 else all_cols[0])

    logger.info("Column detection: x=%s, y=%s", x_col, y_col)
    return x_col, y_col


def generate_chart(
    chart_type: str,
    x_col: str,
    y_col: str,
    df: pd.DataFrame,
    title: Optional[str] = None,
) -> go.Figure:
    """
    Generate a styled Plotly chart.

    Args:
        chart_type: One of 'bar', 'line', 'pie', 'scatter', 'histogram'.
        x_col: Column name for x-axis.
        y_col: Column name for y-axis / values.
        df: Source DataFrame.
        title: Optional custom title.

    Returns:
        A Plotly Figure object.
    """
    logger.info("Generating %s chart: x=%s, y=%s", chart_type, x_col, y_col)

    # Group data for aggregate charts
    if chart_type in ("bar", "line", "pie"):
        grouped = df.groupby(x_col)[y_col].sum().reset_index()
    else:
        grouped = df

    # Auto-generate title
    if title is None:
        title_map = {
            "bar": f"{y_col} by {x_col}",
            "line": f"{y_col} Trend over {x_col}",
            "pie": f"{y_col} Distribution by {x_col}",
            "scatter": f"{y_col} vs {x_col}",
            "histogram": f"Distribution of {x_col}",
        }
        title = title_map.get(chart_type, f"{y_col} by {x_col}")

    # Create the chart
    if chart_type == "bar":
        fig = px.bar(
            grouped, x=x_col, y=y_col, title=title,
            color=y_col, color_continuous_scale="Purples",
            template=CHART_TEMPLATE,
        )
    elif chart_type == "line":
        fig = px.line(
            grouped, x=x_col, y=y_col, title=title,
            markers=True, template=CHART_TEMPLATE,
        )
        fig.update_traces(
            line=dict(width=3, color="#8b5cf6"),
            marker=dict(size=8, color="#a78bfa"),
        )
    elif chart_type == "pie":
        fig = px.pie(
            grouped, names=x_col, values=y_col, title=title,
            color_discrete_sequence=CHART_COLOR_SEQUENCE,
            template=CHART_TEMPLATE,
        )
        fig.update_traces(textposition="inside", textinfo="percent+label")
    elif chart_type == "scatter":
        fig = px.scatter(
            grouped, x=x_col, y=y_col, title=title,
            template=CHART_TEMPLATE,
            color_discrete_sequence=["#8b5cf6"],
        )
        fig.update_traces(marker=dict(size=10, opacity=0.7))
    elif chart_type == "histogram":
        fig = px.histogram(
            grouped, x=x_col, title=title,
            template=CHART_TEMPLATE,
            color_discrete_sequence=["#6366f1"],
        )
    else:
        fig = px.bar(
            grouped, x=x_col, y=y_col, title=title,
            template=CHART_TEMPLATE,
        )

    # Apply premium styling
    fig.update_layout(
        font=dict(family="Inter, system-ui, sans-serif", size=13),
        title_font_size=18,
        title_x=0.5,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=40, r=40, t=60, b=40),
        legend=dict(
            bgcolor="rgba(0,0,0,0.3)",
            bordercolor="rgba(255,255,255,0.1)",
            borderwidth=1,
        ),
    )

    logger.info("Chart generated successfully: %s", chart_type)
    return fig
