"""
Centralized configuration, validation, and logging setup.

All configurable values live here — no magic strings scattered across the app.
"""

import logging
import os
import sys
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


# ══════════════════════════════════════════════════════
# CUSTOM EXCEPTIONS
# ══════════════════════════════════════════════════════

class ConfigurationError(Exception):
    """Raised when required configuration is missing or invalid."""


class LLMConnectionError(Exception):
    """Raised when LLM API is unreachable or returns an error."""


class FileValidationError(Exception):
    """Raised when an uploaded file fails validation."""


class RAGProcessingError(Exception):
    """Raised when the PDF RAG pipeline encounters an error."""


# ══════════════════════════════════════════════════════
# APPLICATION CONFIG
# ══════════════════════════════════════════════════════

@dataclass(frozen=True)
class AppConfig:
    """Immutable application configuration loaded from environment."""

    # LLM
    groq_api_key: str = ""
    model_name: str = "llama-3.3-70b-versatile"
    model_temperature: float = 0.0

    # File limits (MB)
    max_csv_size_mb: int = 50
    max_pdf_size_mb: int = 20

    # RAG settings
    chunk_size: int = 1000
    chunk_overlap: int = 200
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    retriever_top_k: int = 4

    # Logging
    log_level: str = "INFO"

    # App metadata
    app_title: str = "AI Financial Analyst Agent"
    app_icon: str = "📊"


def load_config() -> AppConfig:
    """
    Load configuration from environment variables with validation.

    Raises:
        ConfigurationError: If required variables are missing.
    """
    api_key = os.getenv("GROQ_API_KEY", "").strip()

    if not api_key or api_key == "your_groq_api_key_here":
        raise ConfigurationError(
            "GROQ_API_KEY is not set. Please create a .env file with your API key.\n"
            "Get one free at: https://console.groq.com/keys\n"
            "Then add to .env: GROQ_API_KEY=your_key_here"
        )

    return AppConfig(
        groq_api_key=api_key,
        model_name=os.getenv("MODEL_NAME", "llama-3.3-70b-versatile"),
        model_temperature=float(os.getenv("MODEL_TEMPERATURE", "0")),
        max_csv_size_mb=int(os.getenv("MAX_CSV_SIZE_MB", "50")),
        max_pdf_size_mb=int(os.getenv("MAX_PDF_SIZE_MB", "20")),
        chunk_size=int(os.getenv("CHUNK_SIZE", "1000")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "200")),
        embedding_model=os.getenv(
            "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        ),
        retriever_top_k=int(os.getenv("RETRIEVER_TOP_K", "4")),
        log_level=os.getenv("LOG_LEVEL", "INFO"),
    )


# ══════════════════════════════════════════════════════
# LOGGING SETUP
# ══════════════════════════════════════════════════════

def setup_logging(level: str = "INFO") -> logging.Logger:
    """Configure structured logging for the application."""
    log_level = getattr(logging, level.upper(), logging.INFO)

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )

    # Suppress noisy third-party loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("langchain").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    logging.getLogger("faiss").setLevel(logging.WARNING)

    logger = logging.getLogger("financial_analyst")
    logger.setLevel(log_level)
    return logger
