"""
LLM client initialization for the parent coaching chatbot.

This file belongs to the parent_coach_bot module and is responsible for
creating and returning an authenticated OpenAI client instance.

Supports both standard OpenAI (api.openai.com) and Azure OpenAI endpoints.
The endpoint is used as-is from AZURE_OPENAI_ENDPOINT; no path normalization
is applied so that standard OpenAI URLs (e.g. https://api.openai.com/v1) work
correctly alongside Azure URLs.

Update this file when authentication, endpoint, or HTTP timeout behavior changes.
"""

import os
import logging
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

logger = logging.getLogger(__name__)


def _get_config(key: str, default: str = "") -> str:
    """
    Read a config value from os.environ first, then st.secrets as fallback.

    This supports both local development (via .env) and Streamlit Cloud
    (where secrets live in st.secrets, not os.environ).

    Args:
        key: Environment / secret key name.
        default: Value to return if key is not found anywhere.

    Returns:
        Config value string, stripped of surrounding quotes.
    """
    value = os.getenv(key, "")
    if not value:
        try:
            import streamlit as st  # noqa: PLC0415
            value = str(st.secrets.get(key, default))
        except Exception:
            value = default
    return value.strip('"').strip("'")


def initialize_client() -> OpenAI:
    """
    Load configuration and initialize an authenticated OpenAI client.

    Reads credentials from os.environ (populated by .env locally) or
    st.secrets (populated by the Streamlit Cloud secrets dashboard).

    Returns:
        Authenticated OpenAI client pointed at the configured endpoint.

    Raises:
        RuntimeError: If endpoint or API key cannot be found.
    """
    env_path = Path(__file__).parent / ".env"
    load_dotenv(env_path, override=True)

    endpoint = _get_config("AZURE_OPENAI_ENDPOINT")
    api_key = _get_config("AZURE_OPENAI_API_KEY")

    if not endpoint:
        raise RuntimeError("AZURE_OPENAI_ENDPOINT is not set.")

    if not api_key:
        raise RuntimeError("AZURE_OPENAI_API_KEY is not set.")

    logger.info("Authenticating with API key.")
    return OpenAI(
        base_url=endpoint,
        api_key=api_key,
        timeout=get_timeout_seconds(),
    )


def get_deployment() -> str:
    """
    Return the configured OpenAI deployment / model name.

    Returns:
        Deployment name string.

    Raises:
        RuntimeError: If AZURE_OPENAI_DEPLOYMENT_NAME is not set.
    """
    name = _get_config("AZURE_OPENAI_DEPLOYMENT_NAME")
    if not name:
        raise RuntimeError("AZURE_OPENAI_DEPLOYMENT_NAME is not set.")
    return name


def get_temperature() -> float:
    """
    Return the configured sampling temperature.

    Returns:
        Temperature as float (default 0.7).
    """
    return float(_get_config("OPENAI_TEMPERATURE", "0.7"))


def get_max_tokens() -> int:
    """
    Return the configured max tokens for completions.

    Returns:
        Max tokens as int (default 4000).
    """
    return int(_get_config("OPENAI_MAX_TOKENS", "4000"))


def get_timeout_seconds() -> float:
    """
    Return HTTP client timeout for OpenAI requests (connect + read).

    Prevents APITimeoutError on slow networks or TLS handshakes. Override with
    OPENAI_TIMEOUT_SECONDS in .env or Streamlit secrets.

    Returns:
        Timeout in seconds (default 120).
    """
    return float(_get_config("OPENAI_TIMEOUT_SECONDS", "120"))
