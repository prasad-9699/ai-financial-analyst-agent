"""Tests for configuration module."""

import os
import pytest

from src.config import AppConfig, ConfigurationError, load_config, setup_logging


class TestAppConfig:

    def test_default_values(self):
        config = AppConfig(groq_api_key="test_key")
        assert config.model_name == "llama-3.3-70b-versatile"
        assert config.model_temperature == 0.0
        assert config.max_csv_size_mb == 50
        assert config.max_pdf_size_mb == 20
        assert config.chunk_size == 1000
        assert config.chunk_overlap == 200
        assert config.retriever_top_k == 4

    def test_immutable(self):
        config = AppConfig(groq_api_key="test_key")
        with pytest.raises(AttributeError):
            config.model_name = "different-model"


class TestLoadConfig:

    def test_missing_api_key_raises(self, monkeypatch):
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        with pytest.raises(ConfigurationError, match="GROQ_API_KEY"):
            load_config()

    def test_placeholder_api_key_raises(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "your_groq_api_key_here")
        with pytest.raises(ConfigurationError, match="GROQ_API_KEY"):
            load_config()

    def test_valid_api_key(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "gsk_test_key_12345")
        config = load_config()
        assert config.groq_api_key == "gsk_test_key_12345"

    def test_custom_model_name(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "gsk_test_key_12345")
        monkeypatch.setenv("MODEL_NAME", "llama-3.1-8b-instant")
        config = load_config()
        assert config.model_name == "llama-3.1-8b-instant"


class TestSetupLogging:

    def test_returns_logger(self):
        logger = setup_logging("INFO")
        assert logger.name == "financial_analyst"

    def test_debug_level(self):
        import logging
        logger = setup_logging("DEBUG")
        assert logger.level == logging.DEBUG
