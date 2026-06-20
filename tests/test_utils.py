"""Tests for utility functions."""

import pytest
import pandas as pd

from src.utils import (
    format_dataframe_info,
    sanitize_user_input,
    validate_csv,
    validate_file_size,
)
from src.config import FileValidationError


# ── File Size Validation ─────────────────────────

class TestValidateFileSize:

    def test_within_limit(self):
        data = b"x" * (1024 * 1024)  # 1 MB
        validate_file_size(data, max_size_mb=5, file_name="test.csv")

    def test_exceeds_limit(self):
        data = b"x" * (6 * 1024 * 1024)  # 6 MB
        with pytest.raises(FileValidationError, match="exceeds the 5 MB limit"):
            validate_file_size(data, max_size_mb=5, file_name="test.csv")

    def test_exact_limit(self):
        data = b"x" * (5 * 1024 * 1024)  # Exactly 5 MB
        validate_file_size(data, max_size_mb=5, file_name="test.csv")

    def test_empty_file(self):
        validate_file_size(b"", max_size_mb=5, file_name="empty.csv")


# ── Input Sanitization ──────────────────────────

class TestSanitizeUserInput:

    def test_strips_whitespace(self):
        assert sanitize_user_input("  hello world  ") == "hello world"

    def test_removes_null_bytes(self):
        assert sanitize_user_input("hello\x00world") == "helloworld"

    def test_truncates_long_input(self):
        long_text = "a" * 3000
        result = sanitize_user_input(long_text, max_length=100)
        assert len(result) == 100

    def test_empty_string(self):
        assert sanitize_user_input("") == ""

    def test_none_returns_empty(self):
        assert sanitize_user_input(None) == ""

    def test_normal_text_unchanged(self):
        assert sanitize_user_input("What is the total revenue?") == "What is the total revenue?"


# ── CSV Validation ───────────────────────────────

class TestValidateCSV:

    def test_valid_csv(self, tmp_path):
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("Name,Value\nAlice,100\nBob,200\n")
        df = validate_csv(open(csv_file, "r"))
        assert df.shape == (2, 2)

    def test_empty_csv_raises(self, tmp_path):
        csv_file = tmp_path / "empty.csv"
        csv_file.write_text("")
        with pytest.raises(FileValidationError):
            validate_csv(open(csv_file, "r"))

    def test_single_column_raises(self, tmp_path):
        csv_file = tmp_path / "single.csv"
        csv_file.write_text("Name\nAlice\nBob\n")
        with pytest.raises(FileValidationError, match="at least 2 columns"):
            validate_csv(open(csv_file, "r"))

    def test_headers_only_raises(self, tmp_path):
        csv_file = tmp_path / "headers.csv"
        csv_file.write_text("Name,Value\n")
        with pytest.raises(FileValidationError, match="no data rows"):
            validate_csv(open(csv_file, "r"))


# ── DataFrame Info ───────────────────────────────

class TestFormatDataframeInfo:

    def test_basic_info(self):
        df = pd.DataFrame({"Name": ["A", "B"], "Value": [1, 2]})
        info = format_dataframe_info(df)
        assert "Rows" in info
        assert "Columns" in info
        assert "Numeric" in info
        assert "Text" in info

    def test_all_numeric(self):
        df = pd.DataFrame({"X": [1, 2], "Y": [3, 4]})
        info = format_dataframe_info(df)
        assert "X" in info
        assert "Y" in info
