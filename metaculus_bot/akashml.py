"""AkashML model access: open-weight models served via the Akash Network.

AkashML exposes an OpenAI-compatible REST API at ``https://api.akashml.com/v1``
(``/chat/completions``, ``/models``). Models are addressed in this repo with an
``akashml/<model-id>`` slug, where ``<model-id>`` is AkashML's own Model ID
verbatim (e.g. ``akashml/zai-org/GLM-5.3``). ``build_llm_with_openrouter_fallback``
recognises the prefix and delegates here, so ``llm_configs.py`` keeps a single
builder for every slot.

Under the hood the call goes to litellm's generic ``openai/`` provider with
AkashML's base URL and the personal ``AKASHML_API_KEY``. There is no donated
key and no fallback: AkashML is not behind OpenRouter, so an AkashML failure
surfaces like any other forecaster error.

Pricing is registered with litellm (``register_akashml_pricing``) so the
framework's ``MonetaryCostManager`` reports real spend for AkashML calls
instead of $0 — otherwise per-run cost telemetry would silently undercount.
Rates are USD per 1M tokens as published on AkashML's Models page (2026-10);
they are subject to change, so re-check ``GET /v1/models`` before relying on
the numbers for a budget decision.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

import litellm
from forecasting_tools import GeneralLlm

from metaculus_bot.constants import AKASHML_API_KEY_ENV

logger: logging.Logger = logging.getLogger(__name__)

AKASHML_PREFIX: str = "akashml/"
AKASHML_BASE_URL: str = "https://api.akashml.com/v1"
# litellm provider used to speak to AkashML's OpenAI-compatible endpoint.
_LITELLM_PROVIDER_PREFIX: str = "openai/"


@dataclass(frozen=True)
class AkashMlModel:
    """One AkashML text model: context window and USD prices per 1M tokens."""

    context_tokens: int
    input_per_m: float
    output_per_m: float
    cache_read_per_m: float | None  # None = no cache-read rate (all input bills at full rate)


# Text models from AkashML's Models page. Keys are AkashML Model IDs verbatim.
AKASHML_MODELS: dict[str, AkashMlModel] = {
    "meta-llama/Llama-3.3-70B-Instruct": AkashMlModel(131_072, 0.20, 0.52, 0.10),
    "moonshotai/Kimi-K3": AkashMlModel(1_000_000, 1.20, 14.00, 1.20),
    "openai/gpt-oss-120b": AkashMlModel(131_072, 0.037, 0.187, 0.037),
    "openai/gpt-oss-20b": AkashMlModel(131_072, 0.02, 0.10, None),
    "Qwen/Qwen3.6-35B-A3B": AkashMlModel(262_144, 0.10, 0.90, 0.05),
    "Qwen/Qwen3.8-27B": AkashMlModel(262_144, 0.225, 1.98, 0.05),
    "zai-org/GLM-5.3": AkashMlModel(1_000_000, 0.19, 4.40, 0.19),
}


def is_akashml_model(model: str) -> bool:
    return model.startswith(AKASHML_PREFIX)


def akashml_model_id(model: str) -> str:
    """``akashml/zai-org/GLM-5.3`` -> ``zai-org/GLM-5.3`` (AkashML's own Model ID)."""
    if not is_akashml_model(model):
        raise ValueError(f"Not an AkashML slug (expected '{AKASHML_PREFIX}<model-id>'): {model!r}")
    model_id = model.removeprefix(AKASHML_PREFIX)
    if not model_id:
        raise ValueError(f"AkashML slug is missing a model id: {model!r}")
    return model_id


def _litellm_model_name(model_id: str) -> str:
    return f"{_LITELLM_PROVIDER_PREFIX}{model_id}"


_pricing_registered: bool = False


def register_akashml_pricing() -> None:
    """Register AkashML per-token prices with litellm so cost tracking is not $0.

    Idempotent. Registered under the exact litellm model name we call with
    (``openai/<model-id>``); a model missing from ``AKASHML_MODELS`` still works
    but is logged as cost-untracked by forecasting-tools' ``ModelTracker``.
    """
    entries: dict[str, dict[str, Any]] = {}
    for model_id, spec in AKASHML_MODELS.items():
        entry: dict[str, Any] = {
            "litellm_provider": "openai",
            "mode": "chat",
            "max_input_tokens": spec.context_tokens,
            "input_cost_per_token": spec.input_per_m / 1_000_000,
            "output_cost_per_token": spec.output_per_m / 1_000_000,
        }
        # No cache-read rate means cached input bills at the full input rate.
        cache_read_per_m = spec.cache_read_per_m if spec.cache_read_per_m is not None else spec.input_per_m
        entry["cache_read_input_token_cost"] = cache_read_per_m / 1_000_000
        entries[_litellm_model_name(model_id)] = entry
    litellm.register_model(entries)
    global _pricing_registered  # noqa: PLW0603  # one-shot registration flag
    _pricing_registered = True


class AkashMlLlm(GeneralLlm):
    """A GeneralLlm that calls an AkashML-hosted model.

    ``self.model`` keeps the ``akashml/<model-id>`` slug, so logs, the comment
    ``Model:`` line and ``FORECASTER_MODEL_NAMES`` identify the host unambiguously
    (``openai/gpt-oss-120b`` on AkashML is not OpenAI's API). Only the litellm
    call is pointed at ``openai/<model-id>`` + ``AKASHML_BASE_URL``.
    """

    def __init__(self, *, model: str, api_key: str | None = None, **kwargs: Any) -> None:
        model_id = akashml_model_id(model)
        if "base_url" in kwargs:
            raise ValueError("base_url is fixed for AkashML models; do not pass it")
        resolved_key = api_key or os.getenv(AKASHML_API_KEY_ENV)
        if not resolved_key:
            # Not fatal at construction: llm_configs builds every slot at import time,
            # and an unconfigured key must not break runs that never call this model.
            logger.warning(
                "AKASHML_KEY_MISSING: %s is not set; calls to %s will fail with an auth error.",
                AKASHML_API_KEY_ENV,
                model,
            )
        if model_id not in AKASHML_MODELS:
            logger.warning(
                "AKASHML_UNPRICED_MODEL: %s is not in AKASHML_MODELS, so its spend will not be cost-tracked.",
                model_id,
            )
        if not _pricing_registered:
            register_akashml_pricing()
        super().__init__(
            model=_litellm_model_name(model_id),
            api_key=resolved_key,
            base_url=AKASHML_BASE_URL,
            **kwargs,
        )
        self.model = model


def build_akashml_llm(model: str, **kwargs: Any) -> GeneralLlm:
    """Construct an ``AkashMlLlm`` for an ``akashml/<model-id>`` slug."""
    return AkashMlLlm(model=model, **kwargs)
