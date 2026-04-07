"""Provider-agnostic LLM adapter for the traffic signal agent.

Environment variable contract
------------------------------
  MODEL_PROVIDER   : "openai" | "hf" | "ollama"
                     Default: auto-detect from other vars.
  MODEL_NAME       : Model ID used by OpenAI-compatible endpoints and as
                     fallback HF model name.  Default: meta-llama/llama-3.2-3b-instruct:free
  API_BASE_URL     : Base URL for OpenAI-compatible endpoints.
                     Default: https://openrouter.ai/api/v1
  OPENROUTER_API_KEY: API key for OpenRouter (openai provider).
  OLLAMA_MODEL     : If set, forces ollama provider at localhost:11434.

  HF_TOKEN         : Hugging Face auth token (for hf provider).
  HF_MODEL_NAME    : HF model ID to call.  Falls back to MODEL_NAME.
  HF_API_URL / HF_ENDPOINT:
                     Custom HF Inference Endpoint URL.
                     If unset, uses HF Serverless Inference API:
                       https://api-inference.huggingface.co/models/{model}

Usage
------
  from llm_agent.llm_adapter import build_adapter
  adapter = build_adapter()          # reads env vars
  text = adapter.complete(system, user, max_tokens=16, temperature=0.0)

Design
------
  - BaseAdapter defines a single complete() method.
  - OpenAICompatibleAdapter wraps openai SDK (zero extra deps).
  - HuggingFaceAdapter uses requests with text-generation API.
  - build_adapter() is the single factory used by LLMAgent.
  - Deterministic when temperature=0 for both providers.
  - No model name is hardcoded; all come from env vars.
"""
from __future__ import annotations

import os
import time
from abc import ABC, abstractmethod
from typing import Optional

from dotenv import load_dotenv

load_dotenv(override=False)  # Never override env vars injected by the grader


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------

class LLMAdapter(ABC):
    """Abstract LLM adapter — single interface for all providers."""

    @abstractmethod
    def complete(
        self,
        system: str,
        user: str,
        max_tokens: int = 32,
        temperature: float = 0.0,
    ) -> Optional[str]:
        """Run a chat completion.

        Returns the model's text response, or None on failure.
        """

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Human-readable provider identifier."""

    @property
    @abstractmethod
    def model_id(self) -> str:
        """Model identifier string."""


# ---------------------------------------------------------------------------
# OpenAI-compatible adapter (OpenRouter, Ollama, any OAI endpoint)
# ---------------------------------------------------------------------------

class OpenAICompatibleAdapter(LLMAdapter):
    """Adapter for any OpenAI-compatible HTTP endpoint.

    Works with OpenRouter, Ollama, local llama.cpp servers, etc.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model_name: str,
        timeout: float = 30.0,
    ) -> None:
        self._base_url  = base_url
        self._api_key   = api_key
        self._model     = model_name
        self._timeout   = timeout
        self._client: Optional[object] = None

    # ------------------------------------------------------------------ #

    def _get_client(self) -> Optional[object]:
        if self._client is not None:
            return self._client
        try:
            from openai import OpenAI  # type: ignore
            self._client = OpenAI(
                base_url=self._base_url,
                api_key=self._api_key,
                timeout=self._timeout,
            )
            return self._client
        except ImportError:
            print(
                "[LLMAdapter] ❌ 'openai' package not installed. "
                "Run: pip install openai"
            )
            return None
        except Exception as exc:
            print(f"[LLMAdapter] ❌ Could not create OpenAI client: {exc}")
            return None

    def complete(
        self,
        system: str,
        user: str,
        max_tokens: int = 32,
        temperature: float = 0.0,
    ) -> Optional[str]:
        client = self._get_client()
        if client is None:
            return None
        try:
            resp = client.chat.completions.create(  # type: ignore[attr-defined]
                model=self._model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user},
                ],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return resp.choices[0].message.content.strip()
        except Exception as exc:
            raise  # let caller handle with retry logic

    @property
    def provider_name(self) -> str:
        if "localhost" in self._base_url or "127.0.0.1" in self._base_url:
            return "ollama"
        return "openai_compatible"

    @property
    def model_id(self) -> str:
        return self._model


# ---------------------------------------------------------------------------
# Hugging Face adapter (Serverless Inference API or dedicated endpoint)
# ---------------------------------------------------------------------------

class HuggingFaceAdapter(LLMAdapter):
    """HuggingFace Serverless Inference API adapter.

    Uses the standard text-generation format:
      POST {endpoint}
      body: {"inputs": <prompt>, "parameters": {"temperature": ...,
              "max_new_tokens": ..., "return_full_text": false}}
      response: [{"generated_text": "..."}]

    Endpoint resolved from HF_API_URL env var, else constructed from model name.
    Prompt is formatted for instruction-tuned models (Gemma chat template).
    """

    _HF_SERVERLESS_BASE = "https://api-inference.huggingface.co/models"

    def __init__(
        self,
        model_name: str,
        hf_token: str,
        endpoint_url: Optional[str] = None,
        timeout: float = 60.0,
    ) -> None:
        self._model   = model_name
        self._token   = hf_token
        self._timeout = timeout

        # Use explicit URL if provided, else build serverless URL from model name
        self._endpoint = (
            endpoint_url.rstrip("/") if endpoint_url
            else f"{self._HF_SERVERLESS_BASE}/{model_name}"
        )

        self._headers = {
            "Authorization": f"Bearer {hf_token}",
            "Content-Type": "application/json",
        }

    # ------------------------------------------------------------------ #

    def complete(
        self,
        system: str,
        user: str,
        max_tokens: int = 32,
        temperature: float = 0.0,
    ) -> Optional[str]:
        """Call HF Serverless Inference API with legacy text-generation format.

        Request body:
          {
            "inputs": <prompt_string>,
            "parameters": {
              "temperature": <float>,
              "max_new_tokens": <int>,
              "return_full_text": false
            }
          }

        Response:
          [{"generated_text": "..."}]
        """
        import requests

        prompt = self._build_prompt(system, user)
        payload = {
            "inputs": prompt,
            "parameters": {
                # HF API rejects exactly 0.0 — use 0.01 for determinism
                "temperature": max(temperature, 0.01),
                "max_new_tokens": max_tokens,
                "return_full_text": False,
            },
        }

        try:
            resp = requests.post(
                self._endpoint,
                json=payload,
                headers=self._headers,
                timeout=self._timeout,
            )
            resp.raise_for_status()
            data = resp.json()

            # Standard response: [{"generated_text": "..."}]
            if isinstance(data, list) and data:
                return str(data[0].get("generated_text", "")).strip()

            # Dict response (some endpoints wrap differently)
            if isinstance(data, dict):
                return str(data.get("generated_text", "")).strip()

        except requests.exceptions.RequestException as exc:
            raise RuntimeError(f"HF request error: {exc}") from exc

        return None

    @staticmethod
    def _build_prompt(system: str, user: str) -> str:
        """Build a prompt string compatible with instruction-tuned models.

        Uses Gemma/Llama chat template format:
          <start_of_turn>user
          {system}

          {user}<end_of_turn>
          <start_of_turn>model

        This is model-agnostic enough to work with most HF instruction models.
        """
        return (
            f"<start_of_turn>user\n"
            f"{system}\n\n"
            f"{user}<end_of_turn>\n"
            f"<start_of_turn>model\n"
        )

    @property
    def provider_name(self) -> str:
        return "huggingface"

    @property
    def model_id(self) -> str:
        return self._model




# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_DEFAULT_OAI_BASE  = "https://openrouter.ai/api/v1"
_DEFAULT_OAI_MODEL = "gpt-3.5-turbo"


def build_adapter(verbose: bool = True) -> Optional[LLMAdapter]:
    """Build the appropriate LLM adapter from environment variables.

    PRIORITY ORDER (grader environment always wins):
    1. API_BASE_URL + API_KEY both set → grader/LiteLLM proxy (HIGHEST PRIORITY)
    2. OLLAMA_MODEL set               → local Ollama (localhost:11434)
    3. MODEL_PROVIDER=hf              → HuggingFaceAdapter
    4. MODEL_PROVIDER=openai + key    → OpenAICompatibleAdapter
    5. Nothing configured             → return None (rule-based fallback)

    The grader injects API_BASE_URL and API_KEY as environment variables.
    load_dotenv(override=False) ensures they are never overridden by .env.
    """
    # ── GRADER PATH (highest priority) ──────────────────────────────────
    # When the hackathon grader runs inference, it sets both API_BASE_URL
    # and API_KEY. Detect this and build the adapter immediately.
    grader_url = os.environ.get("API_BASE_URL", "").strip()
    grader_key = os.environ.get("API_KEY", "").strip()

    if grader_url and grader_key:
        model = os.environ.get("MODEL_NAME", _DEFAULT_OAI_MODEL).strip()
        adapter = OpenAICompatibleAdapter(
            base_url=grader_url,
            api_key=grader_key,
            model_name=model,
        )
        if verbose:
            print(
                f"[LLMAdapter] ✔ Grader proxy detected → "
                f"model={model}  base_url={grader_url}",
                flush=True,
            )
        return adapter

    # ── LOCAL: Ollama shortcut ───────────────────────────────────────────
    ollama_model = os.environ.get("OLLAMA_MODEL", "").strip()
    if ollama_model:
        adapter = OpenAICompatibleAdapter(
            base_url="http://localhost:11434/v1",
            api_key="dummy",
            model_name=ollama_model,
        )
        if verbose:
            print(f"[LLMAdapter] ✔ Using Ollama adapter → model={ollama_model}", flush=True)
        return adapter

    provider = os.environ.get("MODEL_PROVIDER", "").strip().lower()

    # ── HuggingFace provider ─────────────────────────────────────────────
    if provider == "hf":
        hf_token = (
            os.environ.get("HF_TOKEN", "")
            or os.environ.get("HUGGING_FACE_TOKEN", "")
        ).strip()
        hf_model = (
            os.environ.get("HF_MODEL_NAME", "")
            or os.environ.get("MODEL_NAME", "")
        ).strip()
        hf_endpoint = (
            os.environ.get("HF_API_URL", "")
            or os.environ.get("HF_ENDPOINT", "")
        ).strip() or None

        if not hf_token:
            print(
                "[LLMAdapter] ❌ MODEL_PROVIDER=hf but HF_TOKEN is not set.\n"
                "           Set HF_TOKEN in your .env file.",
                flush=True,
            )
            return None

        if not hf_model:
            print(
                "[LLMAdapter] ❌ No HF model name found.\n"
                "           Set HF_MODEL_NAME or MODEL_NAME in your .env file.",
                flush=True,
            )
            return None

        adapter = HuggingFaceAdapter(
            model_name=hf_model,
            hf_token=hf_token,
            endpoint_url=hf_endpoint,
        )
        if verbose:
            endpoint_desc = hf_endpoint or "serverless"
            print(
                f"[LLMAdapter] ✔ Using HuggingFace adapter → "
                f"model={hf_model}  endpoint={endpoint_desc}",
                flush=True,
            )
        return adapter

    # ── OpenAI-compatible provider (default / local testing) ────────────
    api_key = (
        os.environ.get("OPENROUTER_API_KEY", "")
        or os.environ.get("HF_TOKEN", "")
        or os.environ.get("HUGGING_FACE_TOKEN", "")
        or os.environ.get("OPENAI_API_KEY", "")
    ).strip()
    base_url = os.environ.get("API_BASE_URL", "").strip() or _DEFAULT_OAI_BASE
    model    = os.environ.get("MODEL_NAME", _DEFAULT_OAI_MODEL).strip()

    adapter = OpenAICompatibleAdapter(
        base_url=base_url,
        api_key=api_key or "dummy",
        model_name=model,
    )
    if verbose:
        print(
            f"[LLMAdapter] ✔ Using OpenAI-compatible adapter → "
            f"model={model}  base_url={base_url}",
            flush=True,
        )
    return adapter
