"""Tests for chart generation and column detection."""

import pytest
import pandas as pd

from src.charts import detect_chart_type, detect_columns, generate_chart


# ── Fixtures ─────────────────────────────────────

@pytest.fixture
def sample_df():
    """Create a sample DataFrame for testing."""
    return pd.DataFrame({
        "Month": ["Jan", "Feb", "Mar", "Apr", "May"],
        "Product": ["A", "B", "A", "B", "A"],
        "Revenue": [1000, 1500, 1200, 1800, 900],
        "Profit": [200, 400, 300, 500, 150],
        "Units": [50, 75, 60, 90, 45],
    })


# ── Chart Type Detection ────────────────────────

class TestDetectChartType:

    def test_line_from_trend(self):
        assert detect_chart_type("show me the revenue trend") == "line"

    def test_line_from_over_time(self):
        assert detect_chart_type("revenue over time") == "line"

    def test_pie_from_distribution(self):
        assert detect_chart_type("show distribution of products") == "pie"

    def test_pie_from_keyword(self):
        assert detect_chart_type("make a pie chart") == "pie"

    def test_scatter_from_correlation(self):
        assert detect_chart_type("show correlation between X and Y") == "scatter"

    def test_scatter_from_vs(self):
        assert detect_chart_type("revenue vs profit") == "scatter"

    def test_histogram_from_keyword(self):
        assert detect_chart_type("histogram of prices") == "histogram"

    def test_default_is_bar(self):
        assert detect_chart_type("show me revenue by month") == "bar"

    def test_empty_string_defaults_to_bar(self):
        assert detect_chart_type("") == "bar"


# ── Column Detection ────────────────────────────

class TestDetectColumns:

    def test_explicit_columns(self, sample_df):
        x, y = detect_columns("show Revenue by Month", sample_df)
        assert x == "Month"
        assert y == "Revenue"

    def test_fallback_to_first_columns(self, sample_df):
        x, y = detect_columns("show me something", sample_df)
        assert x == "Month"  # First text column
        assert y == "Revenue"  # First numeric column

    def test_numeric_y_text_x(self, sample_df):
        x, y = detect_columns("Profit by Product", sample_df)
        assert x == "Product"
        assert y == "Profit"

    def test_only_numeric_mentioned(self, sample_df):
        x, y = detect_columns("show Units", sample_df)
        assert y == "Units"
        assert x == "Month"  # Fallback

    def test_only_text_mentioned(self, sample_df):
        x, y = detect_columns("by Product", sample_df)
        assert x == "Product"
        assert y == "Revenue"  # Fallback


# ── Chart Generation ────────────────────────────

class TestGenerateChart:

    def test_bar_chart_returns_figure(self, sample_df):
        fig = generate_chart("bar", "Month", "Revenue", sample_df)
        assert fig is not None
        assert hasattr(fig, "data")

    def test_line_chart_returns_figure(self, sample_df):
        fig = generate_chart("line", "Month", "Revenue", sample_df)
        assert fig is not None

    def test_pie_chart_returns_figure(self, sample_df):
        fig = generate_chart("pie", "Product", "Revenue", sample_df)
        assert fig is not None

    def test_scatter_chart_returns_figure(self, sample_df):
        fig = generate_chart("scatter", "Revenue", "Profit", sample_df)
        assert fig is not None

    def test_histogram_returns_figure(self, sample_df):
        fig = generate_chart("histogram", "Revenue", "Profit", sample_df)
        assert fig is not None

    def test_custom_title(self, sample_df):
        fig = generate_chart("bar", "Month", "Revenue", sample_df, title="My Custom Title")
        assert fig.layout.title.text == "My Custom Title"

    def test_unknown_type_defaults_to_bar(self, sample_df):
        fig = generate_chart("unknown_type", "Month", "Revenue", sample_df)
        assert fig is not None
