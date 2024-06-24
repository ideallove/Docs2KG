"""
This is allowing user to switch between different LLM models

Mainly is to switch between AzureOpenAI and OpenAI
"""

import os
from typing import Optional


def get_openai_client(
    llm_provider_name: Optional[str] = "openai",
    azure_endpoint: Optional[str] = None,
    azure_api_version: Optional[str] = None,
):
    """
    Get the OpenAI client based on the provider name
    Args:
        llm_provider_name (str): The provider name, default is "openai"
        azure_endpoint (str): The Azure endpoint, default is None, if AzureOpenAI is used, then this is required
        azure_api_version (str): The Azure API version, default is None, if AzureOpenAI is used, then this is required

    Returns:
        OpenAI: The OpenAI or AzureOpenAI client

    """
    if llm_provider_name is None:
        llm_provider_name = os.getenv("LLM_PROVIDER_NAME", "openai")

    if llm_provider_name == "openai":
        from openai import OpenAI

        return OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
        )
    elif llm_provider_name == "azure_openai":
        # TODO: later if requested, add intra id authentication
        if azure_endpoint is None:
            raise ValueError("Azure endpoint is required for AzureOpenAI")
        if azure_api_version is None:
            raise ValueError("Azure API version is required for AzureOpenAI")
        from openai import AzureOpenAI

        return AzureOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            azure_endpoint=azure_endpoint,
            api_version=azure_api_version,
        )
    else:
        raise ValueError(f"Unknown LLM provider name: {llm_provider_name}")
