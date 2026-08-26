"""
File validation and utility helpers.

Handles upload validation, input sanitization, and temp file management.
"""

import logging
import re
import tempfile
from pathlib import Path
from typing import Optional

import pandas as pd

from src.config import FileValidationError

logger = logging.getLogger("financial_analyst.utils")

# Allowed file extensions
ALLOWED_CSV_EXTENSIONS = {".csv"}
ALLOWED_PDF_EXTENSIONS = {".pdf"}

# Encodings to try in order when reading CSV files
_CSV_ENCODINGS = ["utf-8", "latin-1", "cp1252", "iso-8859-1"]


def validate_file_size(file_bytes: bytes, max_size_mb: int, file_name: str = "file") -> None:
    """
    Validate that a file does not exceed the maximum allowed size.

    Args:
        file_bytes: Raw file content.
        max_size_mb: Maximum allowed size in megabytes.
        file_name: Name for error messages.

    Raises:
        FileValidationError: If file exceeds size limit.
    """
    size_mb = len(file_bytes) / (1024 * 1024)
    if size_mb > max_size_mb:
        raise FileValidationError(
            f"'{file_name}' is {size_mb:.1f} MB, which exceeds the {max_size_mb} MB limit. "
            f"Please upload a smaller file."
        )
    logger.info("File size OK: %s (%.2f MB)", file_name, size_mb)


def validate_csv(file) -> pd.DataFrame:
    """
    Validate and parse a CSV file upload with automatic encoding detection.

    Tries multiple encodings (utf-8, latin-1, cp1252) to handle international
    characters. Strips whitespace from column names and drops fully empty columns.

    Args:
        file: Streamlit UploadedFile object.

    Returns:
        Parsed and cleaned DataFrame.

    Raises:
        FileValidationError: If the CSV is malformed or empty.
    """
    df = None
    last_error = None

    # Try each encoding in order
    for encoding in _CSV_ENCODINGS:
        try:
            file.seek(0)
            df = pd.read_csv(file, encoding=encoding)
            logger.info("CSV parsed successfully with encoding: %s", encoding)
            break
        except UnicodeDecodeError:
            last_error = f"Encoding {encoding} failed"
            continue
        except pd.errors.EmptyDataError:
            raise FileValidationError("The CSV file is empty. Please upload a file with data.")
        except pd.errors.ParserError as e:
            raise FileValidationError(
                f"Could not parse the CSV file. Please check the format.\nDetails: {e}"
            )
        except Exception as e:
            last_error = str(e)
            continue

    if df is None:
        raise FileValidationError(
            f"Error reading CSV — could not decode with any supported encoding.\nDetails: {last_error}"
        )

    if df.empty:
        raise FileValidationError("The CSV file contains headers but no data rows.")

    # Clean up column names: strip whitespace
    df.columns = df.columns.str.strip()

    # Drop fully empty columns (all NaN)
    empty_cols = [c for c in df.columns if df[c].isna().all()]
    if empty_cols:
        df = df.drop(columns=empty_cols)
        logger.info("Dropped %d empty columns: %s", len(empty_cols), empty_cols)

    # Drop unnamed columns (pandas artifact from extra commas)
    unnamed = [c for c in df.columns if str(c).startswith("Unnamed")]
    if unnamed:
        df = df.drop(columns=unnamed)
        logger.info("Dropped %d unnamed columns", len(unnamed))

    if len(df.columns) < 2:
        raise FileValidationError(
            "The CSV file needs at least 2 columns for meaningful analysis."
        )

    logger.info("CSV validated: %d rows × %d columns", df.shape[0], df.shape[1])
    return df


def save_temp_file(file_bytes: bytes, suffix: str = ".pdf") -> str:
    """
    Save uploaded bytes to a secure temporary file.

    Args:
        file_bytes: Raw file content.
        suffix: File extension.

    Returns:
        Path to the temporary file.
    """
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(file_bytes)
    tmp.close()
    logger.info("Temp file saved: %s", tmp.name)
    return tmp.name


def cleanup_temp_file(path: str) -> None:
    """Safely remove a temporary file."""
    try:
        p = Path(path)
        if p.exists():
            p.unlink()
            logger.debug("Temp file cleaned up: %s", path)
    except OSError as e:
        logger.warning("Could not clean up temp file %s: %s", path, e)


def sanitize_user_input(text: str, max_length: int = 2000) -> str:
    """
    Sanitize user input text.

    - Strips leading/trailing whitespace
    - Truncates to max_length
    - Removes null bytes

    Args:
        text: Raw user input.
        max_length: Maximum allowed character count.

    Returns:
        Sanitized string.
    """
    if not text:
        return ""
    text = text.strip()
    text = text.replace("\x00", "")
    if len(text) > max_length:
        text = text[:max_length]
        logger.warning("User input truncated to %d characters", max_length)
    return text


def format_dataframe_info(df: pd.DataFrame) -> str:
    """Generate a human-readable summary of a DataFrame with rich details."""
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    text_cols = df.select_dtypes(include="object").columns.tolist()
    date_cols = df.select_dtypes(include="datetime").columns.tolist()

    # Memory usage
    mem_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)

    # Null info
    total_nulls = df.isna().sum().sum()
    total_cells = df.shape[0] * df.shape[1]
    null_pct = (total_nulls / total_cells * 100) if total_cells > 0 else 0

    parts = [
        f"**Rows:** {df.shape[0]:,}  |  **Columns:** {df.shape[1]}  |  **Memory:** {mem_mb:.2f} MB",
        f"**Numeric:** {', '.join(numeric_cols) if numeric_cols else 'None'}",
        f"**Text:** {', '.join(text_cols) if text_cols else 'None'}",
    ]
    if date_cols:
        parts.append(f"**Date:** {', '.join(date_cols)}")

    if total_nulls > 0:
        parts.append(f"**Missing values:** {total_nulls:,} ({null_pct:.1f}%)")
    else:
        parts.append("**Missing values:** None (clean)")

    return "\n".join(parts)


def get_csv_quick_insights(df: pd.DataFrame) -> list:
    """
    Generate quick stat insights for the top numeric columns.

    Returns a list of dicts: [{"col": name, "min": x, "max": y, "mean": z}, ...]
    """
    numeric_cols = df.select_dtypes(include="number").columns.tolist()[:4]
    insights = []
    for col in numeric_cols:
        try:
            insights.append({
                "col": col,
                "min": df[col].min(),
                "max": df[col].max(),
                "mean": df[col].mean(),
            })
        except Exception:
            continue
    return insights
