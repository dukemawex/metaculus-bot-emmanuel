import litellm
import pytest
from forecasting_tools.ai_models import general_llm as ft_general_llm
from litellm import ModelResponse, Usage

from metaculus_bot import akashml
from metaculus_bot.akashml import (
    AKASHML_BASE_URL,
    AKASHML_FREE_FALLBACKS,
    AKASHML_MODELS,
    AkashMlLlm,
    AkashMlWithFreeFallbackLlm,
    akashml_model_id,
    build_akashml_llm,
    fallback_served_marker,
    get_akashml_fallback_count,
    is_akashml_model,
    register_akashml_pricing,
    reset_akashml_fallback_count,
)
from metaculus_bot.constants import AKASHML_API_KEY_ENV
from metaculus_bot.fallback_openrouter import FallbackOpenRouterLlm, build_llm_with_openrouter_fallback
from metaculus_bot.llm_configs import FORECASTER_LLMS, _forecaster_display_name


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


# --- AkashML -> free OpenRouter fallback -------------------------------------------


@pytest.fixture
def _fallback_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(AKASHML_API_KEY_ENV, "akash-key")
    monkeypatch.setenv("OPENROUTER_API_KEY", "personal")
    monkeypatch.setenv("OAI_ANTH_OPENROUTER_KEY", "donated")
    reset_akashml_fallback_count()


def _install_fake_acompletion(monkeypatch: pytest.MonkeyPatch, fail_models: set[str]) -> list[dict]:
    """Patch ft's acompletion: raise for models in ``fail_models``, answer otherwise."""
    calls: list[dict] = []

    async def fake(**kwargs: object) -> ModelResponse:
        calls.append(kwargs)
        if kwargs["model"] in fail_models:
            raise litellm.exceptions.APIConnectionError(message="akash down", llm_provider="openai", model="m")
        return ModelResponse(
            model=str(kwargs["model"]),
            choices=[{"message": {"role": "assistant", "content": f"answer from {kwargs['model']}"}}],
            usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        )

    monkeypatch.setattr(ft_general_llm, "acompletion", fake)
    return calls


@pytest.mark.usefixtures("_fallback_env")
async def test_primary_success_does_not_touch_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _install_fake_acompletion(monkeypatch, fail_models=set())
    llm = build_akashml_llm("akashml/zai-org/GLM-5.3", fallback_model="openrouter/nvidia/x:free", allowed_tries=1)

    answer = await llm.invoke("forecast")

    assert answer == "answer from openai/zai-org/GLM-5.3"
    assert [c["model"] for c in calls] == ["openai/zai-org/GLM-5.3"]
    assert get_akashml_fallback_count() == 0


@pytest.mark.usefixtures("_fallback_env")
async def test_primary_failure_falls_back_to_free_model_on_personal_key(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    calls = _install_fake_acompletion(monkeypatch, fail_models={"openai/zai-org/GLM-5.3"})
    llm = build_akashml_llm("akashml/zai-org/GLM-5.3", fallback_model="openrouter/nvidia/x:free", allowed_tries=1)

    answer = await llm.invoke("forecast", system_prompt="sys")

    assert [c["model"] for c in calls] == ["openai/zai-org/GLM-5.3", "openrouter/nvidia/x:free"]
    primary, fallback = calls
    assert primary["base_url"] == AKASHML_BASE_URL
    assert primary["api_key"] == "akash-key"
    # Free fallback: personal key, OpenRouter's default endpoint, never the donated key.
    assert fallback["api_key"] == "personal"
    assert fallback.get("base_url") is None
    assert answer.startswith(fallback_served_marker("openrouter/nvidia/x:free"))
    assert answer.endswith("answer from openrouter/nvidia/x:free")
    assert get_akashml_fallback_count() == 1
    assert "AKASHML_FALLBACK" in caplog.text


@pytest.mark.usefixtures("_fallback_env")
async def test_both_failing_raises_fallback_error(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_acompletion(monkeypatch, fail_models={"openai/zai-org/GLM-5.3", "openrouter/nvidia/x:free"})
    llm = build_akashml_llm("akashml/zai-org/GLM-5.3", fallback_model="openrouter/nvidia/x:free", allowed_tries=1)

    with pytest.raises(litellm.exceptions.APIConnectionError):
        await llm.invoke("forecast")


async def test_no_openrouter_key_means_no_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(AKASHML_API_KEY_ENV, "akash-key")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    calls = _install_fake_acompletion(monkeypatch, fail_models={"openai/zai-org/GLM-5.3"})
    llm = build_akashml_llm("akashml/zai-org/GLM-5.3", fallback_model="openrouter/nvidia/x:free", allowed_tries=1)

    with pytest.raises(litellm.exceptions.APIConnectionError):
        await llm.invoke("forecast")
    assert len(calls) == 1


def test_every_catalogued_fallback_is_a_free_openrouter_slug() -> None:
    for model_id, fallback in AKASHML_FREE_FALLBACKS.items():
        assert model_id in AKASHML_MODELS
        assert fallback.startswith("openrouter/")
        assert fallback.endswith(":free")
    assert len(set(AKASHML_FREE_FALLBACKS.values())) == len(AKASHML_FREE_FALLBACKS)


def test_prod_roster_is_akashml_with_free_fallbacks() -> None:
    assert FORECASTER_LLMS
    for llm in FORECASTER_LLMS:
        assert isinstance(llm, AkashMlWithFreeFallbackLlm)
        assert is_akashml_model(llm.model)
        assert "gpt-oss" not in llm.model
        assert llm.fallback_model == AKASHML_FREE_FALLBACKS[akashml_model_id(llm.model)]
