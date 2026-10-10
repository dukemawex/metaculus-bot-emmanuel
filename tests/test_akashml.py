import litellm
import pytest
from litellm import ModelResponse, Usage

from metaculus_bot import akashml
from metaculus_bot.akashml import (
    AKASHML_BASE_URL,
    AKASHML_MODELS,
    AkashMlLlm,
    akashml_model_id,
    is_akashml_model,
    register_akashml_pricing,
)
from metaculus_bot.constants import AKASHML_API_KEY_ENV
from metaculus_bot.fallback_openrouter import FallbackOpenRouterLlm, build_llm_with_openrouter_fallback
from metaculus_bot.llm_configs import _forecaster_display_name


def test_slug_parsing() -> None:
    assert is_akashml_model("akashml/zai-org/GLM-5.3")
    assert not is_akashml_model("openrouter/openai/gpt-oss-120b")
    assert akashml_model_id("akashml/openai/gpt-oss-120b") == "openai/gpt-oss-120b"
    with pytest.raises(ValueError, match="Not an AkashML slug"):
        akashml_model_id("openrouter/zai-org/GLM-5.3")
    with pytest.raises(ValueError, match="missing a model id"):
        akashml_model_id("akashml/")


def test_builder_routes_akashml_slug_to_akashml_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(AKASHML_API_KEY_ENV, "akash-key")
    # Both OpenRouter keys set: an akashml slug must still never touch them.
    monkeypatch.setenv("OPENROUTER_API_KEY", "personal")
    monkeypatch.setenv("OAI_ANTH_OPENROUTER_KEY", "donated")

    llm = build_llm_with_openrouter_fallback("akashml/openai/gpt-oss-120b", max_tokens=100, allowed_tries=1)

    assert isinstance(llm, AkashMlLlm)
    assert not isinstance(llm, FallbackOpenRouterLlm)
    assert llm.model == "akashml/openai/gpt-oss-120b"
    assert llm.litellm_kwargs["model"] == "openai/openai/gpt-oss-120b"
    assert llm.litellm_kwargs["base_url"] == AKASHML_BASE_URL
    assert llm.litellm_kwargs["api_key"] == "akash-key"
    assert llm.litellm_kwargs["max_tokens"] == 100


def test_explicit_api_key_wins_over_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(AKASHML_API_KEY_ENV, "env-key")
    llm = AkashMlLlm(model="akashml/zai-org/GLM-5.3", api_key="explicit")
    assert llm.litellm_kwargs["api_key"] == "explicit"


def test_missing_key_warns_but_constructs(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    monkeypatch.delenv(AKASHML_API_KEY_ENV, raising=False)
    llm = AkashMlLlm(model="akashml/zai-org/GLM-5.3")
    assert llm.litellm_kwargs["api_key"] is None
    assert "AKASHML_KEY_MISSING" in caplog.text


def test_base_url_override_rejected() -> None:
    with pytest.raises(ValueError, match="base_url"):
        AkashMlLlm(model="akashml/zai-org/GLM-5.3", base_url="https://example.com")


def test_display_name_keeps_model_name() -> None:
    llm = AkashMlLlm(model="akashml/moonshotai/Kimi-K3", api_key="k")
    assert _forecaster_display_name(llm) == "Kimi-K3"


def test_pricing_registered_with_litellm() -> None:
    register_akashml_pricing()
    response = ModelResponse(
        model="zai-org/GLM-5.3",
        choices=[{"message": {"role": "assistant", "content": "hi"}}],
        usage=Usage(prompt_tokens=1_000_000, completion_tokens=1_000_000, total_tokens=2_000_000),
    )
    spec = AKASHML_MODELS["zai-org/GLM-5.3"]
    cost = litellm.completion_cost(completion_response=response, model="openai/zai-org/GLM-5.3")
    assert cost == pytest.approx(spec.input_per_m + spec.output_per_m)


def test_unpriced_model_warns(caplog: pytest.LogCaptureFixture) -> None:
    AkashMlLlm(model="akashml/some-org/brand-new-model", api_key="k")
    assert "AKASHML_UNPRICED_MODEL" in caplog.text


def test_catalogue_prices_are_positive() -> None:
    assert AKASHML_MODELS
    for spec in AKASHML_MODELS.values():
        assert spec.input_per_m > 0
        assert spec.output_per_m > 0
        assert spec.context_tokens > 0
    assert akashml.AKASHML_PREFIX == "akashml/"
