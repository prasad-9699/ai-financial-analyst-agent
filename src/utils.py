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
    Validate and parse a CSV file upload.

    Args:
        file: Streamlit UploadedFile object.

    Returns:
        Parsed DataFrame.

    Raises:
        FileValidationError: If the CSV is malformed or empty.
    """
    try:
        df = pd.read_csv(file)
    except pd.errors.EmptyDataError:
        raise FileValidationError("The CSV file is empty. Please upload a file with data.")
    except pd.errors.ParserError as e:
        raise FileValidationError(
            f"Could not parse the CSV file. Please check the format.\nDetails: {e}"
        )
    except Exception as e:
        raise FileValidationError(f"Error reading CSV: {e}")

    if df.empty:
        raise FileValidationError("The CSV file contains headers but no data rows.")

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
    """Generate a human-readable summary of a DataFrame."""
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    text_cols = df.select_dtypes(include="object").columns.tolist()
    date_cols = df.select_dtypes(include="datetime").columns.tolist()

    parts = [
        f"**Rows:** {df.shape[0]:,}  |  **Columns:** {df.shape[1]}",
        f"**Numeric:** {', '.join(numeric_cols) if numeric_cols else 'None'}",
        f"**Text:** {', '.join(text_cols) if text_cols else 'None'}",
    ]
    if date_cols:
        parts.append(f"**Date:** {', '.join(date_cols)}")

    return "\n".join(parts)
