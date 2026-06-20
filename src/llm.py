"""
LLM factory with caching and retry logic.

Handles transient Groq API failures gracefully with exponential backoff.
"""

import logging

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
            max_retries=3,
            request_timeout=60,
        )
        logger.info("LLM initialized: model=%s, temp=%s", config.model_name, config.model_temperature)
        return llm
    except Exception as e:
        logger.error("Failed to initialize LLM: %s", e)
        raise LLMConnectionError("Could not initialize LLM: %s" % e) from e


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=15),
    retry=retry_if_exception_type(Exception),
    reraise=True,
)
def invoke_llm_with_retry(llm, prompt):
    """
    Invoke the LLM with automatic retry on transient failures.

    Args:
        llm: The ChatGroq instance.
        prompt: The user prompt.

    Returns:
        The LLM response content as a string.
    """
    try:
        response = llm.invoke(prompt)
        return response.content
    except Exception as e:
        logger.warning("LLM invocation failed (will retry): %s", e)
        raise


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=15),
    retry=retry_if_exception_type(Exception),
    reraise=True,
)
def chat_with_history(llm, prompt, history):
    """
    Invoke the LLM with full conversation history for context-aware responses.

    Args:
        llm: The ChatGroq instance.
        prompt: The current user prompt.
        history: List of message dicts with 'role' and 'content' keys.

    Returns:
        The LLM response content as a string.
    """
    messages = [
        SystemMessage(content=(
            "You are a helpful AI Financial Analyst assistant. "
            "You have memory of the full conversation. "
            "Refer to previous messages when relevant."
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

    try:
        response = llm.invoke(messages)
        return response.content
    except Exception as e:
        logger.warning("LLM chat invocation failed (will retry): %s", e)
        raise
