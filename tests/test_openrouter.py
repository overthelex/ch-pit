import json

import httpx
import pytest

from chpit import openrouter
from chpit.provider import RetryableError


def _transport(handler):
    return httpx.MockTransport(handler)


def test_complete_parses_text_usage_served_model_and_cost():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        seen["auth"] = request.headers["authorization"]
        return httpx.Response(200, json={
            "model": "anthropic/claude-sonnet-5-20260801", "provider": "Anthropic",
            "choices": [{"message": {"role": "assistant", "content": "Art. 1 ..."},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 12, "completion_tokens": 7, "cost": 0.00042},
        })

    p = openrouter.OpenRouterProvider(api_key="k", transport=_transport(handler))
    c = p.complete("anthropic/claude-sonnet-5", [{"role": "user", "content": "q"}], max_tokens=99)
    assert c.text == "Art. 1 ..." and c.input_tokens == 12 and c.output_tokens == 7
    assert c.cost_usd == pytest.approx(0.00042) and c.served_model.endswith("20260801")
    assert c.provider_name == "Anthropic" and c.finish_reason == "stop"
    assert seen["auth"] == "Bearer k" and seen["body"]["max_tokens"] == 99
    assert seen["body"]["usage"] == {"include": True} and seen["body"]["temperature"] == 0.0


def test_complete_parses_tool_calls():
    def handler(request):
        return httpx.Response(200, json={"choices": [{"message": {
            "role": "assistant", "content": None,
            "tool_calls": [{"id": "c1", "type": "function", "function": {
                "name": "ch_get_act_article",
                "arguments": json.dumps({"sr_number": "220", "article": "336", "as_of": "2021-01-01"})}}]},
            "finish_reason": "tool_calls"}], "usage": {}})
    p = openrouter.OpenRouterProvider(api_key="k", transport=_transport(handler))
    c = p.complete("m", [], tools=[{"type": "function", "function": {"name": "x"}}])
    assert c.text == "" and c.tool_calls[0].name == "ch_get_act_article"
    assert c.tool_calls[0].arguments["as_of"] == "2021-01-01" and c.raw_message["tool_calls"]


@pytest.mark.parametrize("status", [429, 502, 503])
def test_retryable_statuses_raise_retryable_error(status):
    p = openrouter.OpenRouterProvider(api_key="k", transport=_transport(
        lambda r: httpx.Response(status, text="slow down")))
    with pytest.raises(RetryableError):
        p.complete("m", [])


def test_non_retryable_status_raises_runtime_error():
    p = openrouter.OpenRouterProvider(api_key="k", transport=_transport(
        lambda r: httpx.Response(400, text="bad request")))
    with pytest.raises(RuntimeError, match="HTTP 400"):
        p.complete("m", [])


def test_error_body_with_retryable_code_is_retryable():
    p = openrouter.OpenRouterProvider(api_key="k", transport=_transport(
        lambda r: httpx.Response(200, json={"error": {"code": 429, "message": "rate"}})))
    with pytest.raises(RetryableError):
        p.complete("m", [])


def test_fetch_prices_scales_to_usd_per_million():
    def handler(request):
        assert request.url.path.endswith("/models")
        return httpx.Response(200, json={"data": [
            {"id": "a/b", "pricing": {"prompt": "0.000003", "completion": "0.000015"}},
            {"id": "c/d", "pricing": {"prompt": "0", "completion": "0"}},
            {"id": "broken", "pricing": {"prompt": None}},
        ]})
    prices = openrouter.fetch_prices(transport=_transport(handler))
    assert prices["a/b"] == {"in": pytest.approx(3.0), "out": pytest.approx(15.0)}
    assert prices["c/d"] == {"in": 0.0, "out": 0.0} and "broken" not in prices


def test_missing_api_key_is_an_error(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(ValueError):
        openrouter.OpenRouterProvider()
