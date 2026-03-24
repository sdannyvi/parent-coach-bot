"""
LLM client initialization for the parent coaching chatbot.

This file belongs to the parent_coach_bot module and is responsible for
creating and returning an authenticated OpenAI client instance.

Supports both standard OpenAI (api.openai.com) and Azure OpenAI endpoints.
The endpoint is used as-is from AZURE_OPENAI_ENDPOINT; no path normalization
is applied so that standard OpenAI URLs (e.g. https://api.openai.com/v1) work
correctly alongside Azure URLs.

Update this file only when the authentication mechanism or endpoint changes.
"""

import os
import logging
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from azure.identity import DefaultAzureCredential, get_bearer_token_provider

logger = logging.getLogger(__name__)


def initialize_client() -> OpenAI:
    """
    Load environment variables and initialize an authenticated OpenAI client.

    Authentication order:
    1. API key (AZURE_OPENAI_API_KEY) if present.
    2. Azure AD via DefaultAzureCredential (uses AZURE_TENANT_ID,
       AZURE_CLIENT_ID, AZURE_CLIENT_SECRET env vars or a logged-in Azure CLI).

    Returns:
        Authenticated OpenAI client pointed at the configured endpoint.

    Raises:
        RuntimeError: If required environment variables are missing or auth fails.
    """
    env_path = Path(__file__).parent / ".env"
    load_dotenv(env_path, override=True)

    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "").strip('"').strip("'")
    cognitive_services_url = os.getenv("AZURE_COGNITIVE_SERVICES_URL", "").strip('"').strip("'")
    api_key = os.getenv("AZURE_OPENAI_API_KEY", "").strip('"').strip("'")

    if not endpoint:
        raise RuntimeError("AZURE_OPENAI_ENDPOINT is not set in the environment.")

    if api_key:
        logger.info("Authenticating with API key.")
        return OpenAI(base_url=endpoint, api_key=api_key)

    if cognitive_services_url:
        logger.info("Authenticating with Azure AD (DefaultAzureCredential).")
        credential = DefaultAzureCredential()
        token_provider = get_bearer_token_provider(credential, cognitive_services_url)
        # The openai SDK accepts a callable as api_key when using token providers
        return OpenAI(base_url=endpoint, api_key=token_provider)

    raise RuntimeError(
        "No authentication method available. "
        "Set AZURE_OPENAI_API_KEY or AZURE_COGNITIVE_SERVICES_URL."
    )


def get_deployment() -> str:
    """
    Return the configured Azure OpenAI deployment name.

    Returns:
        Deployment name string.

    Raises:
        RuntimeError: If AZURE_OPENAI_DEPLOYMENT_NAME is not set.
    """
    name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "").strip('"').strip("'")
    if not name:
        raise RuntimeError("AZURE_OPENAI_DEPLOYMENT_NAME is not set in the environment.")
    return name


def get_temperature() -> float:
    """
    Return the configured sampling temperature.

    Returns:
        Temperature as float (default 0.7).
    """
    return float(os.getenv("OPENAI_TEMPERATURE", "0.7"))


def get_max_tokens() -> int:
    """
    Return the configured max tokens for completions.

    Returns:
        Max tokens as int (default 4000).
    """
    return int(os.getenv("OPENAI_MAX_TOKENS", "4000"))
