"""The model-provider seam every runner mode talks through.

`Provider.complete()` takes OpenAI-shape messages (`role`/`content`, plus
`tool_calls` / `tool_call_id` for the agentic loop) and returns a
`Completion`. Providers raise `RetryableError` for anything worth another
attempt (rate limit, 5xx, timeout, an empty body); the runner owns the
backoff and the per-item error record. Tests inject a fake provider.
"""
from __future__ import annotations

import dataclasses
from typing import Any, Protocol


class RetryableError(Exception):
    """Transient provider failure: retry with backoff."""


@dataclasses.dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclasses.dataclass(frozen=True)
class Completion:
    text: str
    tool_calls: tuple[ToolCall, ...] = ()
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float | None = None
    served_model: str | None = None
    provider_name: str | None = None
    finish_reason: str | None = None
    raw_message: dict[str, Any] | None = None  # the assistant message to append in a tool loop


class Provider(Protocol):
    def complete(self, model: str, messages: list[dict[str, Any]], *,
                 tools: list[dict[str, Any]] | None = None,
                 tool_choice: str | None = None,
                 max_tokens: int = 2048,
                 temperature: float = 0.0) -> Completion: ...
