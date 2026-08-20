"""Tests for configuration module."""

import os
import pytest

from src.config import AppConfig, ConfigurationError, load_config, setup_logging


class TestAppConfig:

    def test_default_values(self):
        config = AppConfig(groq_api_key="test_key")
        assert config.model_name == "qwen/qwen3.6-27b"
        assert config.model_temperature == 0.0
        assert config.max_csv_size_mb == 50
        assert config.max_pdf_size_mb == 20
        assert config.chunk_size == 1000
        assert config.chunk_overlap == 200
        assert config.retriever_top_k == 4
        assert config.tesseract_cmd == "tesseract"

    def test_immutable(self):
        config = AppConfig(groq_api_key="test_key")
        with pytest.raises(AttributeError):
            config.model_name = "different-model"


def _set_required_env(monkeypatch):
    """Helper to set all required env vars for load_config()."""
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test_key_12345")
    monkeypatch.setenv("SUPABASE_URL", "https://test.supabase.co")
    monkeypatch.setenv("SUPABASE_KEY", "test_anon_key_12345")


class TestLoadConfig:

    def test_missing_api_key_raises(self, monkeypatch):
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        with pytest.raises(ConfigurationError, match="GROQ_API_KEY"):
            load_config()

    def test_placeholder_api_key_raises(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "your_groq_api_key_here")
        with pytest.raises(ConfigurationError, match="GROQ_API_KEY"):
            load_config()

    def test_missing_supabase_url_raises(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "gsk_test_key_12345")
        monkeypatch.delenv("SUPABASE_URL", raising=False)
        monkeypatch.delenv("SUPABASE_KEY", raising=False)
        with pytest.raises(ConfigurationError, match="SUPABASE_URL"):
            load_config()

    def test_missing_supabase_key_raises(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "gsk_test_key_12345")
        monkeypatch.setenv("SUPABASE_URL", "https://test.supabase.co")
        monkeypatch.delenv("SUPABASE_KEY", raising=False)
        with pytest.raises(ConfigurationError, match="SUPABASE_KEY"):
            load_config()

    def test_valid_config(self, monkeypatch):
        _set_required_env(monkeypatch)
        config = load_config()
        assert config.groq_api_key == "gsk_test_key_12345"
        assert config.supabase_url == "https://test.supabase.co"
        assert config.supabase_key == "test_anon_key_12345"

    def test_custom_model_name(self, monkeypatch):
        _set_required_env(monkeypatch)
        monkeypatch.setenv("MODEL_NAME", "llama-3.1-8b-instant")
        config = load_config()
        assert config.model_name == "llama-3.1-8b-instant"

    def test_custom_tesseract_cmd(self, monkeypatch):
        _set_required_env(monkeypatch)
        monkeypatch.setenv("TESSERACT_CMD", r"C:\Tesseract\tesseract.exe")
        config = load_config()
        assert config.tesseract_cmd == r"C:\Tesseract\tesseract.exe"


class TestSetupLogging:

    def test_returns_logger(self):
        logger = setup_logging("INFO")
        assert logger.name == "financial_analyst"

    def test_debug_level(self):
        import logging
        logger = setup_logging("DEBUG")
        assert logger.level == logging.DEBUG

