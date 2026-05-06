"""LLM provider abstraction — Ollama (default) or Groq.

Select the provider via the LLM_PROVIDER environment variable:
  LLM_PROVIDER=ollama  (default)  — uses ChatOllama + local qwen2.5-coder:30b
  LLM_PROVIDER=groq               — uses ChatGroq + GROQ_API_KEY

All other env vars apply per-provider:
  Ollama: OLLAMA_URL, CHAT_MODEL, OLLAMA_TIMEOUT
  Groq:   GROQ_API_KEY, GROQ_MODEL

Three public helpers cover every LLM interaction pattern:
  chat_model()  — plain chat client
  with_tools()  — binds tools so the model emits tool_calls
  structured()  — validates LLM output against a Pydantic schema
"""
from __future__ import annotations

import os

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Module-level defaults — read at import time for the Ollama path so that
# tests that monkeypatch and reload the module still get stable constants.
# ---------------------------------------------------------------------------

_CHAT_MODEL: str = os.environ.get("CHAT_MODEL", "qwen2.5-coder:30b")

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def chat_model(temperature: float = 0) -> BaseChatModel:
    """Return a configured chat model for the active provider.

    Reads LLM_PROVIDER, GROQ_API_KEY, OLLAMA_URL etc. at call time so that
    the provider can be switched between tests without reloading the module.

    Args:
        temperature: Sampling temperature (0 = deterministic for planning/tools,
                     0.2 for free-form text generation).
    """
    provider = os.environ.get("LLM_PROVIDER", "ollama").lower()

    if provider == "groq":
        from langchain_groq import ChatGroq

        return ChatGroq(
            model=os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile"),
            api_key=os.environ.get("GROQ_API_KEY", ""),  # type: ignore[arg-type]
            temperature=temperature,
        )

    # Default: Ollama
    from langchain_ollama import ChatOllama

    model_name = os.environ.get("CHAT_MODEL", _CHAT_MODEL)
    # Disable qwen3's extended thinking — it produces <think>...</think> tokens
    # that break LangChain's tool-call parsing.
    extra: dict[str, object] = {}
    if "qwen3" in model_name.lower():
        extra["think"] = False

    return ChatOllama(
        model=model_name,
        base_url=os.environ.get("OLLAMA_URL", "http://localhost:11434"),
        temperature=temperature,
        timeout=int(os.environ.get("OLLAMA_TIMEOUT", "120")),
        **extra,
    )


def with_tools(model: BaseChatModel, tools: list[BaseTool]) -> Runnable:
    """Return the model with the given tools bound."""
    return model.bind_tools(tools)


def structured(model: BaseChatModel, schema: type[BaseModel]) -> Runnable:
    """Return a Runnable that validates LLM output against *schema*."""
    return model.with_structured_output(schema)
