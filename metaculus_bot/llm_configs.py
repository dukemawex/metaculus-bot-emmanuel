"""Centralised model configuration for TemplateForecaster.

Keeping these objects in a single module avoids merge-conflicts and makes it
possible to tweak/benchmark models without touching application code.
"""

from typing import Any

from forecasting_tools import GeneralLlm

from metaculus_bot.akashml import AKASHML_FREE_FALLBACKS, build_akashml_llm
from metaculus_bot.fallback_openrouter import build_llm_with_openrouter_fallback

__all__ = [
    "DISAGREEMENT_ANALYZER_LLM",
    "FORECASTER_LLMS",
    "FORECASTER_MODEL_NAMES",
    "MARKET_QUERY_AUTHOR_LLM_CONFIG",
    "MARKET_RANKER_LLM_CONFIG",
    "PARSER_LLM",
    "RESEARCHER_LLM",
    "STACKER_FALLBACK_LLM",
    "STACKER_LLM",
    "SUMMARIZER_LLM",
]
# Reasoning models ignore (or degrade under) explicit sampling params, so we
# defer to provider defaults. temperature=None is explicit but redundant on
# ft 0.2.92, whose GeneralLlm ctor already defaults temperature to None (0.2.54
# injected 0 when the arg was omitted); top_p flows via **kwargs and is never set.
REASONING_MODEL_CONFIG: dict[str, Any] = {
    "temperature": None,
    "max_tokens": 64_000,  # Prevent truncation; all current forecasters/stackers support 64k output
    "stream": False,
    "timeout": 480,
    "allowed_tries": 3,
}
# Low-effort utility slots (parser, summarizer, analyzer). Same sampling-param
# rationale as REASONING_MODEL_CONFIG: temperature=None defers to provider
# defaults (redundant on ft 0.2.92, whose ctor default is already None); top_p
# left unset.
UTILITY_MODEL_CONFIG: dict[str, Any] = {
    "temperature": None,
    "max_tokens": 32_000,
    "stream": False,
    "timeout": 300,
    "allowed_tries": 3,
}
ACCEPTABLE_QUANTS = [
    "fp8",
    "fp16",
    "bf16",
    "fp32",
    "unknown",
]

# Per-instance allowed_tries=1 override (Round-2): forecaster .invoke is wrapped
# in the broad retry gated on TRANSIENT_RETRY_MAX_ELAPSED_S (forecaster_runners.py)
# so we can impose the universal "never retry a slow failure" deadline-safety rule
# that forecasting-tools' un-gated tenacity cannot. Spread per-instance (NOT by mutating
# REASONING_MODEL_CONFIG) so PARSER_LLM / STACKER configs are untouched.
_FORECASTER_CONFIG = {**REASONING_MODEL_CONFIG, "allowed_tries": 1}

FORECASTER_LLMS: list[GeneralLlm] = [
    # 2026-10-10 (operator decision): the forecaster roster is the three AkashML
    # open-weight models, each with a free OpenRouter backup. Replaces the
    # gpt-5.6-sol / claude-opus-4.8 / gemini-3.1-pro-preview triple, so no forecaster
    # call bills OpenRouter any more. Primary calls go to AkashML's OpenAI-compatible
    # endpoint on AKASHML_API_KEY; on ANY failure a slot retries once on its
    # AKASHML_FREE_FALLBACKS model (personal OPENROUTER_API_KEY, `:free`, $0) and stamps
    # the answer with the model that served it (metaculus_bot/akashml.py). No reasoning
    # kwargs: OpenRouter's `reasoning` shape is not in AkashML's documented parameter set,
    # so these run at provider defaults. gpt-oss deliberately excluded (operator).
    # (Dates anchor config eras for residual analysis — this is a new era boundary from
    # its merge-to-main date.)
    build_akashml_llm(
        "akashml/zai-org/GLM-5.3",
        fallback_model=AKASHML_FREE_FALLBACKS["zai-org/GLM-5.3"],
        **_FORECASTER_CONFIG,
    ),
    build_akashml_llm(
        "akashml/moonshotai/Kimi-K3",
        fallback_model=AKASHML_FREE_FALLBACKS["moonshotai/Kimi-K3"],
        **_FORECASTER_CONFIG,
    ),
    build_akashml_llm(
        "akashml/Qwen/Qwen3.8-27B",
        fallback_model=AKASHML_FREE_FALLBACKS["Qwen/Qwen3.8-27B"],
        **_FORECASTER_CONFIG,
    ),
]


def _forecaster_display_name(llm: GeneralLlm) -> str:
    """Short label for a forecaster (e.g. 'claude-opus-4.8') — strips the 'openrouter/<provider>/' prefix.

    Used by performance_analysis.parsing to map 'Forecaster N' labels in bot comments
    back to a model name without having to hand-maintain a parallel list.
    """
    return llm.model.rsplit("/", 1)[-1]


FORECASTER_MODEL_NAMES: list[str] = [_forecaster_display_name(llm) for llm in FORECASTER_LLMS]

# Summarizer: compresses raw AskNews article markdown into an analyst briefing
# (AskNews-only; all other providers already emit LLM prose). sol → terra
# 2026-07-18 operator decision: AskNews is an auxiliary/augmenting source
# (content audit: 16% unique content vs native-search 54% / gap-fill 59%), so
# the absolute-frontier tier isn't warranted. The role audit
# (scratch/research_role_audit_2026-07-17/) had sol 1st but verdict "MARGINAL
# EDGE" with terra 2nd (one attribution blur, no fabrications), and 4/5 briefing
# failures in the AskNews quality audit (scratch/asknews_quality_audit_2026-07-18/)
# were prompt-era (mini summarizer + missing no-forecast rule), not model-tier.
# Terra: -43% cost, ~50s vs ~118s wall. Effort stays low (latency).
# allowed_tries=1 (Round-2): the summarizer invoke is wrapped in the broad,
# elapsed-gated retry (orchestrator._summarize_asknews) to impose the universal
# "never retry a slow failure" deadline rule. Per-instance override so PARSER_LLM (which
# also uses UTILITY_MODEL_CONFIG) keeps its allowed_tries=3.
SUMMARIZER_LLM: GeneralLlm = build_llm_with_openrouter_fallback(
    "openrouter/openai/gpt-5.6-terra",
    reasoning={"effort": "low"},
    **{**UTILITY_MODEL_CONFIG, "allowed_tries": 1},
)
# Parser: deterministic extraction of percentiles/JSON from rationales — a
# capability-saturated task, so it rides the cheapest tier that saturates it and
# keeps allowed_tries=3 for robustness. mini → luna 2026-08-03: the per-token
# comparison that used to favor mini inverted. Luna is $0.20/$1.20 vs mini's
# $0.75/$4.50 per 1M, so the newer model is also the ~3.75x cheaper one. (The
# models API showed $0.10/$0.60 behind a "50% off" badge on 2026-08-03; a live
# call on 2026-08-04 billed at double that, so the promo does not apply on this
# route — see the ranker cost comment below. The swap still wins, by less.)
# Effort unchanged at low.
PARSER_LLM: GeneralLlm = build_llm_with_openrouter_fallback(
    "openrouter/openai/gpt-5.6-luna",
    reasoning={"effort": "low"},
    **UTILITY_MODEL_CONFIG,
)
# Researcher slot in the forecasting-tools LLM config dict. Effectively dead
# code in our pipeline — we use research providers (AskNews/Gemini/native_search)
# rather than the framework's researcher path — but the slot must be populated
# to avoid silent framework defaults. Aliasing to SUMMARIZER_LLM rather than
# constructing a duplicate config: same model, same effort, same job tier, no
# reason to maintain two parallel definitions.
RESEARCHER_LLM = SUMMARIZER_LLM

# Stacker meta-model for conditional stacking (invoked only on high-disagreement questions).
#
# allowed_tries=1: a single attempt at REASONING_MODEL_CONFIG's timeout, no
# retries. The outer STACKER_SOFT_DEADLINE catches wholly stuck calls; on failure
# we fall back to STACKER_FALLBACK_LLM rather than burning two more full-timeout
# attempts against the same Anthropic API that just stalled. Retrying against the same
# provider after a stall rarely succeeds (we're almost certainly re-rolling a
# dice with the same distribution), and the budget is better spent on a
# different-provider fallback.
STACKER_LLM: GeneralLlm = build_llm_with_openrouter_fallback(
    # 2026-07-20: fable-5 → opus-4.8 (fable-5 pulled from BOTH roles after
    # content=None failures in the 2026-07-19 test_bot run — see the forecaster-slot
    # comment above + FUTURE.md). Stacking is prod-disabled, so this is
    # backtest/ablation-only exposure today. opus-4.8 uses effort-based adaptive
    # thinking, not a max_tokens budget. Live-verified OpenRouter effort enum:
    # none/minimal/low/medium/high/xhigh/max. effort=xhigh matches the opus-4.8
    # forecaster slot; "max" (one tier above xhigh) is deliberately held back for
    # latency — the stacker runs under STACKER_SOFT_DEADLINE.
    "openrouter/anthropic/claude-opus-4.8",
    reasoning={"effort": "xhigh"},
    extra_body={"verbosity": "high"},
    **{**REASONING_MODEL_CONFIG, "allowed_tries": 1},
)

# Fallback stacker used when the primary stacker times out or errors.
# Reasoning slot → strongest OpenAI tier (gpt-5.6-sol) at high effort;
# deliberately cross-provider from the Anthropic Fable primary so an Anthropic
# stall doesn't take both attempts down. Tighter timeout and single try since
# we're already running late on the critical path by the time this fires.
# Stays at high (not xhigh) for that same reason — the 2026-07-15 xhigh bump
# covers the primary stacker and forecaster slots, not this tighter-budget path.
STACKER_FALLBACK_LLM: GeneralLlm = build_llm_with_openrouter_fallback(
    "openrouter/openai/gpt-5.6-sol",
    reasoning={"effort": "high"},
    **{**REASONING_MODEL_CONFIG, "allowed_tries": 1, "timeout": 300},
)

# --- The prediction-market provider's two LLM stages ---
#
# Both are RAW DICTS rather than built GeneralLlm singletons, unlike PARSER_LLM and friends:
# the provider is gated OFF by default, so paying construction cost at import would be waste,
# and the tests patch `build_llm_with_openrouter_fallback` at the provider's one invocation
# helper. Both route `openrouter/openai/...` through that wrapper, which tries the donated
# Metaculus key first and falls back to the personal key on credential / credit / route errors
# — so prod spend lands on OAI_ANTH_OPENROUTER_KEY.
#
# `allowed_tries=1` is required, not decorative: the repo's elapsed-gated `llm_retry` wrapper
# (prediction_market._invoke_market_llm) is the SOLE retry layer, and leaving this unpinned
# inherits forecasting-tools' default of 2 with an UN-GATED `random.uniform(5, 10)` tenacity
# sleep — a large slice of PREDICTION_MARKET_TIMEOUT spent sleeping blind, which is exactly
# what llm_retry exists to eliminate. `temperature=None` defers reasoning models to provider
# defaults (redundant on ft 0.2.92, whose ctor default is already None); top_p left unset. Each
# litellm `timeout` sits ABOVE its elapsed-gated wall cap in constants.py, so the wall is the
# binding bound.
#
# Luna is the cheapest tier that saturates both tasks. The real rate on this route is $0.20/M in
# and $1.20/M out — TWICE the $0.10/$0.60 the bake-off read off the models API on 2026-08-03,
# where a "50% off" badge was displayed that has since lapsed or never applied here. A live
# ranking call reconciled the true rates to 7 significant figures against OpenRouter's own
# `upstream_inference_cost` (26,250 in / 685 out / a 25% cache-WRITE surcharge on the input,
# `scratch/market_port_2026-08-04/QA_DRY_RUN.md`), so this is measured rather than quoted.
#
# MEASURED cost per question: ranker $0.0074 (26k in at the median post-enrichment,
# full-PredictIt shape + ~685 out, cache write included); author ~1.4k in + ~300 out ≈ $0.0005.
# The two keyword calls they replace measured ~170 tok in / ~50 out ≈ $0.0001, so net new is
# ≈ +$0.008 per question — under a cent per run at the prod shape of 1-2 questions, and ~$0.24
# of ranker spend across a 30-question tournament run. The earlier ~$0.003-0.004 arithmetic in
# the port plan understated by 2.4x purely because of the promo price; the token shapes were
# right. This traffic is ~97% input, so the input rate is the whole cost.

# Prediction-market RANKER: one call per question over the whole ~380-440-candidate pool,
# emitting up to 8 ranked rows with a relation tier and a one-phrase label. Measured completion
# averages 589 tokens including reasoning, max 1,042 (scratch/bakeoff_run_2026-08-03/results/
# RANKED_ARM_RESULTS.md). max_tokens sits ~3x above that max because a TRUNCATED ranking is a
# fail-open — the whole ranking is lost, not just its tail — and luna's output tokens are cheap.
MARKET_RANKER_LLM_CONFIG: dict = {
    "model": "openrouter/openai/gpt-5.6-luna",
    "temperature": None,
    "max_tokens": 3000,
    "reasoning_effort": "low",
    "timeout": 90,
    "allowed_tries": 1,
}

# Prediction-market QUERY AUTHOR: one call per question emitting the domain vocabulary the
# question's own tokens cannot reach (up to 8 synonyms + 3 framings). Its output is ADDITIVE to
# a deterministic query set, so its failure costs recall nothing. Measured completion max 588
# tokens including reasoning; max_tokens sits ~2.5x above that.
MARKET_QUERY_AUTHOR_LLM_CONFIG: dict = {
    "model": "openrouter/openai/gpt-5.6-luna",
    "temperature": None,
    "max_tokens": 1500,
    "reasoning_effort": "low",
    "timeout": 45,
    "allowed_tries": 1,
}


# Tier-B auxiliary: read-and-synthesize work that needs taste but not deep
# reasoning. Identifies the crux of forecaster disagreement; output text seeds
# the targeted-search query downstream. Runs under CRUX_SOFT_DEADLINE;
# effort deliberately low since 2026-05-20 for latency — the tier was upgraded
# instead (smarter-model-at-lower-effort beats more effort on a smaller model).
# 2026-07-17: sol→terra per the role audit; terra 2nd (sol 3rd) at -49% cost;
# the role fires rarely (stacking disabled in prod).
# allowed_tries=1 (Round-2): the crux-analyzer invoke is wrapped in the broad,
# elapsed-gated retry (targeted.extract_disagreement_crux) to impose the universal
# "never retry a slow failure" deadline rule on the conditional-stacking critical path.
# Per-instance override so PARSER_LLM keeps its allowed_tries=3.
DISAGREEMENT_ANALYZER_LLM: GeneralLlm = build_llm_with_openrouter_fallback(
    "openrouter/openai/gpt-5.6-terra",
    reasoning={"effort": "low"},
    **{**UTILITY_MODEL_CONFIG, "allowed_tries": 1},
)
