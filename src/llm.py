"""
LLM factory with caching, retry logic, and automatic model fallback.

Handles transient Groq API failures gracefully with exponential backoff.
If the primary model is overloaded, automatically falls back to an
alternative model.
"""

import logging
import re
import time

from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.config import AppConfig, LLMConnectionError

logger = logging.getLogger("financial_analyst.llm")

# Fallback models to try when the primary model is overloaded (in order)
FALLBACK_MODELS = [
    "qwen/qwen3.6-27b",
    "openai/gpt-oss-20b",
]


def strip_thinking_tags(text: str) -> str:
    """Remove <think>...</think> blocks and any partial thinking tags from model output."""
    # Remove complete <think>...</think> blocks
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    # Remove unclosed <think> blocks (tag opened but never closed)
    cleaned = re.sub(r"<think>.*", "", cleaned, flags=re.DOTALL)
    # Remove any stray closing tags
    cleaned = re.sub(r"</think>", "", cleaned)
    return cleaned.strip()


def create_llm(config: AppConfig) -> ChatGroq:
    """
    Create a configured ChatGroq LLM instance.

    Args:
        config: Application configuration with API key and model settings.

    Returns:
        Configured ChatGroq instance.

    Raises:
        LLMConnectionError: If the LLM cannot be initialized.
    """
    try:
        llm = ChatGroq(
            model=config.model_name,
            temperature=config.model_temperature,
            groq_api_key=config.groq_api_key,
            max_retries=2,
            request_timeout=60,
        )
        logger.info("LLM initialized: model=%s, temp=%s", config.model_name, config.model_temperature)
        return llm
    except Exception as e:
        logger.error("Failed to initialize LLM: %s", e)
        raise LLMConnectionError("Could not initialize LLM: %s" % e) from e


def _try_invoke(llm, messages_or_prompt, config=None):
    """
    Try invoking the LLM. On overload errors, automatically retry with
    fallback models.

    Args:
        llm: Primary ChatGroq instance.
        messages_or_prompt: Either a string prompt or list of messages.
        config: AppConfig (needed to create fallback LLMs).

    Returns:
        The LLM response content as a string.
    """
    # Try primary model first
    try:
        response = llm.invoke(messages_or_prompt)
        return strip_thinking_tags(response.content)
    except Exception as e:
        error_str = str(e).lower()
        is_overloaded = any(kw in error_str for kw in [
            "overloaded", "rate_limit", "rate limit", "429", "503",
            "too many requests", "capacity", "service unavailable",
        ])

        if not is_overloaded:
            raise  # Not an overload error — don't try fallbacks

        logger.warning("Primary model overloaded: %s", e)

    # Try fallback models
    for fallback_model in FALLBACK_MODELS:
        try:
            logger.info("Trying fallback model: %s", fallback_model)
            fallback_llm = ChatGroq(
                model=fallback_model,
                temperature=getattr(config, "model_temperature", 0.0) if config else 0.0,
                groq_api_key=llm.groq_api_key,
                max_retries=2,
                request_timeout=60,
            )
            time.sleep(1)  # Brief pause before fallback
            response = fallback_llm.invoke(messages_or_prompt)
            logger.info("Fallback model %s succeeded", fallback_model)
            return strip_thinking_tags(response.content)
        except Exception as fallback_err:
            logger.warning("Fallback model %s also failed: %s", fallback_model, fallback_err)
            continue

    # All models failed
    raise LLMConnectionError(
        "All AI models are currently overloaded. Please wait a moment and try again."
    )


@retry(
    stop=stop_after_attempt(2),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(Exception),
    reraise=True,
)
def invoke_llm_with_retry(llm, prompt, config=None):
    """
    Invoke the LLM with automatic retry and model fallback.

    Args:
        llm: The ChatGroq instance.
        prompt: The user prompt.
        config: Optional AppConfig for fallback model creation.

    Returns:
        The LLM response content as a string.
    """
    return _try_invoke(llm, prompt, config)


@retry(
    stop=stop_after_attempt(2),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(Exception),
    reraise=True,
)
def chat_with_history(llm, prompt, history, config=None):
    """
    Invoke the LLM with full conversation history for context-aware responses.

    Args:
        llm: The ChatGroq instance.
        prompt: The current user prompt.
        history: List of message dicts with 'role' and 'content' keys.
        config: Optional AppConfig for fallback model creation.

    Returns:
        The LLM response content as a string.
    """
    messages = [
        SystemMessage(content=(
            "/no_think "
            "You are a helpful AI Financial Analyst assistant. "
            "You have memory of the full conversation. "
            "Refer to previous messages when relevant. "
            "Current Year: 2026. "
            "IMPORTANT: Do NOT include any <think> tags or reasoning process in your response. "
            "Give the final answer directly."
        ))
    ]

    # Add conversation history (last 20 exchanges to stay within context limits)
    for msg in history[-40:]:
        content = msg.get("content", "")
        if not content:
            continue
        if msg["role"] == "user":
            messages.append(HumanMessage(content=content))
        elif msg["role"] == "assistant":
            messages.append(AIMessage(content=content))

    # Add current prompt
    messages.append(HumanMessage(content=prompt))

    return _try_invoke(llm, messages, config)
