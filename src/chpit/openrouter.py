"""OpenRouter provider: POST /api/v1/chat/completions with usage
accounting, and the public price table from GET /api/v1/models.

Pin fully qualified slugs (`anthropic/claude-sonnet-5`, never a bare
alias) and read `served_model` back from every response: an alias can be
re-pointed by the provider, and only the served model names what actually
answered. `usage.cost` is requested (`usage: {include: true}`) so a line's
cost is what OpenRouter billed, not an estimate.
"""
from __future__ import annotations

import json
import os
from typing import Any

from chpit.provider import Completion, RetryableError, ToolCall

BASE_URL = "https://openrouter.ai/api/v1"
_RETRY_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}


def _client(timeout: float):
    try:
        import httpx
    except ImportError as exc:  # pragma: no cover
        raise ImportError("OpenRouterProvider needs httpx: pip install 'chpit[openrouter]'") from exc
    return httpx.Client(timeout=timeout)


class OpenRouterProvider:
    def __init__(self, api_key: str | None = None, base_url: str = BASE_URL,
                 timeout: float = 180.0, extra_body: dict[str, Any] | None = None,
                 transport: Any = None, app_title: str = "CH-PiT benchmark"):
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY is not set")
        self.base_url = base_url.rstrip("/")
        self.extra_body = extra_body or {}
        self.app_title = app_title
        if transport is not None:
            import httpx
            self._http = httpx.Client(timeout=timeout, transport=transport)
        else:
            self._http = _client(timeout)

    def complete(self, model: str, messages: list[dict[str, Any]], *,
                 tools: list[dict[str, Any]] | None = None, tool_choice: str | None = None,
                 max_tokens: int = 2048, temperature: float = 0.0) -> Completion:
        body: dict[str, Any] = {
            "model": model, "messages": messages, "max_tokens": max_tokens,
            "temperature": temperature, "usage": {"include": True},
        }
        if tools:
            body["tools"] = tools
        if tool_choice:
            body["tool_choice"] = tool_choice
        body.update(self.extra_body)
        try:
            resp = self._http.post(
                f"{self.base_url}/chat/completions", json=body,
                headers={"Authorization": f"Bearer {self.api_key}",
                         "HTTP-Referer": "https://lawrider.ch", "X-Title": self.app_title})
        except Exception as exc:  # httpx transport / timeout errors
            raise RetryableError(f"{type(exc).__name__}: {exc}") from exc
        if resp.status_code in _RETRY_STATUS:
            raise RetryableError(f"HTTP {resp.status_code}: {resp.text[:200]}")
        if resp.status_code != 200:
            raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:300]}")
        data = resp.json()
        if "error" in data and not data.get("choices"):
            err = data["error"]
            code = err.get("code") if isinstance(err, dict) else None
            msg = err.get("message") if isinstance(err, dict) else str(err)
            if code in _RETRY_STATUS:
                raise RetryableError(f"{code}: {msg}")
            raise RuntimeError(f"OpenRouter error {code}: {msg}")
        return parse_completion(data)

    def close(self) -> None:
        self._http.close()


def parse_completion(data: dict[str, Any]) -> Completion:
    choice = (data.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    text = message.get("content") or ""
    if isinstance(text, list):  # some providers return content parts
        text = "".join(p.get("text", "") for p in text if isinstance(p, dict))
    calls: list[ToolCall] = []
    for tc in message.get("tool_calls") or []:
        fn = tc.get("function") or {}
        raw_args = fn.get("arguments") or "{}"
        try:
            args = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args)
        except json.JSONDecodeError:
            args = {"_unparseable": raw_args}
        calls.append(ToolCall(id=tc.get("id", ""), name=fn.get("name", ""), arguments=args))
    usage = data.get("usage") or {}
    cost = usage.get("cost")
    return Completion(
        text=text, tool_calls=tuple(calls),
        input_tokens=int(usage.get("prompt_tokens") or 0),
        output_tokens=int(usage.get("completion_tokens") or 0),
        cost_usd=float(cost) if cost is not None else None,
        served_model=data.get("model"), provider_name=data.get("provider"),
        finish_reason=choice.get("finish_reason"),
        raw_message=message,
    )


def fetch_prices(base_url: str = BASE_URL, timeout: float = 30.0,
                 transport: Any = None) -> dict[str, dict[str, float]]:
    """{slug: {"in": USD per 1M prompt tokens, "out": USD per 1M completion
    tokens}} for every model OpenRouter lists. Public endpoint, no key."""
    import httpx
    http = httpx.Client(timeout=timeout, transport=transport) if transport else _client(timeout)
    resp = http.get(f"{base_url.rstrip('/')}/models")
    resp.raise_for_status()
    prices: dict[str, dict[str, float]] = {}
    for m in resp.json().get("data", []):
        pricing = m.get("pricing") or {}
        try:
            prices[m["id"]] = {"in": float(pricing.get("prompt", 0)) * 1e6,
                               "out": float(pricing.get("completion", 0)) * 1e6}
        except (TypeError, ValueError):
            continue
    return prices
