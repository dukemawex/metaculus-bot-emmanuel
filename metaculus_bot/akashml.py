"""AkashML model access: open-weight models served via the Akash Network.

AkashML exposes an OpenAI-compatible REST API at ``https://api.akashml.com/v1``
(``/chat/completions``, ``/models``). Models are addressed in this repo with an
``akashml/<model-id>`` slug, where ``<model-id>`` is AkashML's own Model ID
verbatim (e.g. ``akashml/zai-org/GLM-5.3``). ``build_llm_with_openrouter_fallback``
recognises the prefix and delegates here, so ``llm_configs.py`` keeps a single
builder for every slot.

Under the hood the call goes to litellm's generic ``openai/`` provider with
AkashML's base URL and the personal ``AKASHML_API_KEY``. AkashML is not behind
OpenRouter, so the donated key never applies. A plain ``AkashMlLlm`` surfaces a
failure like any other forecaster error; ``AkashMlWithFreeFallbackLlm`` (used by
the forecaster roster) retries once on a free OpenRouter model instead.

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

from metaculus_bot.constants import AKASHML_API_KEY_ENV, OPENROUTER_API_KEY_ENV

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


# Free OpenRouter backup for each AkashML forecaster slot. None of the AkashML models
# has a `:free` twin on OpenRouter (checked against /api/v1/models 2026-10-10), so each
# slot gets a DIFFERENT strong free model to keep the ensemble diverse when AkashML is
# down. All three have >=262k context and a completion cap above the forecaster's 64k
# max_tokens. Free models are rate-limited per account; a throttled one just fails
# the slot.
AKASHML_FREE_FALLBACKS: dict[str, str] = {
    "zai-org/GLM-5.3": "openrouter/nvidia/nemotron-3-ultra-550b-a55b:free",
    "moonshotai/Kimi-K3": "openrouter/thinkingmachines/inkling:free",
    "Qwen/Qwen3.8-27B": "openrouter/nvidia/nemotron-3-super-120b-a12b:free",
}

# Run counter of AkashML -> OpenRouter free fallbacks. Diagnostic, NOT folded into the
# alertable total: the backup is free and by design, so it must not redden CI on its own.
# The per-event AKASHML_FALLBACK warning makes every occurrence grep-able in run logs.
_akashml_fallback_count: int = 0


def get_akashml_fallback_count() -> int:
    return _akashml_fallback_count


def reset_akashml_fallback_count() -> None:
    """Reset the counter to zero. Used by tests; not for production code."""
    global _akashml_fallback_count  # noqa: PLW0603  # module-global run counter is the design (AGENTS.md)
    _akashml_fallback_count = 0


def fallback_served_marker(fallback_model: str) -> str:
    """Line prepended to an answer the free fallback produced.

    The forecaster's ``Model:`` line names the configured slot (the AkashML slug), and the
    LLM singletons are shared by concurrent questions, so no attribute on the object can
    say which model served THIS call. Stamping the answer itself keeps attribution honest
    in the published rationale. Value extraction reads the trailing JSON block, so a
    leading line does not disturb it.
    """
    return f"[AkashML unavailable — this forecast was produced by fallback model {fallback_model}]"


class AkashMlWithFreeFallbackLlm(AkashMlLlm):
    """AkashML primary; on ANY failure, retry once on a free OpenRouter model.

    The fallback runs on the personal ``OPENROUTER_API_KEY`` as a plain ``GeneralLlm``,
    deliberately NOT via the donated-key wrapper: most ``:free`` models are served by
    providers outside the donated key's allowed list, which would 404 and bump the
    alertable donated->personal counter on every call (see ablation/forecaster_lineup.py).
    """

    def __init__(self, *, model: str, fallback_model: str, api_key: str | None = None, **kwargs: Any) -> None:
        super().__init__(model=model, api_key=api_key, **kwargs)
        self.fallback_model = fallback_model
        openrouter_key = os.getenv(OPENROUTER_API_KEY_ENV)
        if not openrouter_key:
            logger.warning(
                "AKASHML_FALLBACK_UNAVAILABLE: %s is not set; %s has no free fallback.",
                OPENROUTER_API_KEY_ENV,
                model,
            )
        self._fallback_llm: GeneralLlm | None = (
            GeneralLlm(model=fallback_model, api_key=openrouter_key, **kwargs) if openrouter_key else None
        )

    async def invoke(self, prompt: Any, system_prompt: str | None = None) -> str:  # type: ignore[override]
        try:
            return await super().invoke(prompt, system_prompt)
        except Exception as e:
            if self._fallback_llm is None:
                raise
            # No await before the increment, so concurrent fallbacks cannot race it.
            global _akashml_fallback_count  # noqa: PLW0603  # module-global run counter is the design (AGENTS.md)
            _akashml_fallback_count += 1
            logger.warning(
                "AKASHML_FALLBACK: model=%s failed, retrying on free OpenRouter model %s. error=%s: %s",
                self.model,
                self.fallback_model,
                type(e).__name__,
                str(e)[:500],
            )
            answer = await self._fallback_llm.invoke(prompt, system_prompt)
            return f"{fallback_served_marker(self.fallback_model)}\n\n{answer}"


def build_akashml_llm(model: str, fallback_model: str | None = None, **kwargs: Any) -> GeneralLlm:
    """Construct an AkashML LLM for an ``akashml/<model-id>`` slug.

    With ``fallback_model`` (an ``openrouter/...:free`` slug), failures retry once there.
    """
    if fallback_model is None:
        return AkashMlLlm(model=model, **kwargs)
    return AkashMlWithFreeFallbackLlm(model=model, fallback_model=fallback_model, **kwargs)
