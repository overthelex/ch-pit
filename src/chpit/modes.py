"""How a system answers one item. Each mode is a small object with a name,
a way to price an item before anything is spent, and `answer()`, which
gets a `call` closure (provider.complete with retries and usage
accounting, owned by the runner) and a `log` dict for per-item extras.

Modes:
  closed   the model sees only the item's `question` -- act, article, date --
           with no retrieval (the v2 Bedrock baseline, prompt unchanged).
  recite   no model at all: the current edition of the article as the
           point-in-time tool returns it WITHOUT a date. Correct on items
           whose gold is still current, wrong-version elsewhere.
  current  the same current-edition text handed to the model as context;
           measures copy fidelity of a wrong-edition context.
  pit      the edition valid on `as_of` handed to the model as context;
           its distance to the oracle's 1.000 is transcription loss.
  agentic  the model gets the point-in-time tools and decides itself
           whether to pass the date. See agentic.py.
"""
from __future__ import annotations

import math
from typing import Any, Callable, Protocol

from chpit.provider import Completion

CHARS_PER_TOKEN = 4
OUTPUT_TOKEN_OVERHEAD = 50

# Verbatim from the v2 Bedrock run so closed-book numbers stay comparable.
CLOSED_BOOK_SYSTEM_PROMPT = (
    "You are a Swiss legal database. Answer with the verbatim text of the "
    "requested article as in force on the given date, in the language of "
    "the question, nothing else."
)

RAG_SYSTEM_PROMPT = (
    "You are a Swiss legal database. The user asks for the verbatim text of "
    "an article as in force on a given date. A retrieved article text is "
    "provided between <context> tags. Answer with the verbatim article text "
    "from the context, in the language of the question, nothing else."
)

Call = Callable[..., Completion]


def tokens(chars: int) -> int:
    return math.ceil(chars / CHARS_PER_TOKEN)


class Mode(Protocol):
    name: str

    def estimate_tokens(self, item: dict[str, Any]) -> tuple[int, int]:
        """(input_tokens, output_tokens) for pricing ITEM before the run."""

    def answer(self, item: dict[str, Any], call: Call, log: dict[str, Any]) -> str:
        """The system's free-text answer to ITEM."""


class ClosedBook:
    name = "closed"
    system_prompt = CLOSED_BOOK_SYSTEM_PROMPT

    def estimate_tokens(self, item: dict[str, Any]) -> tuple[int, int]:
        return (tokens(len(self.system_prompt) + len(item["question"])),
                tokens(len(item["gold"]["text"])) + OUTPUT_TOKEN_OVERHEAD)

    def answer(self, item: dict[str, Any], call: Call, log: dict[str, Any]) -> str:
        completion = call([
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": item["question"]},
        ])
        return completion.text


class _Retrieval:
    """Shared plumbing for the modes that fetch an article from the
    point-in-time tool. FETCH(item, as_of) returns the tool's JSON."""

    def __init__(self, fetch: Callable[[dict[str, Any], str | None], dict[str, Any]]):
        self._fetch = fetch

    def _article_text(self, item: dict[str, Any], as_of: str | None,
                      log: dict[str, Any]) -> str:
        result = self._fetch(item, as_of)
        log["tool_result"] = {k: result.get(k) for k in ("error", "as_of")}
        version = result.get("version") or {}
        if version:
            log["tool_result"]["version_id"] = version.get("version_id")
            log["tool_result"]["date_applicability"] = version.get("date_applicability")
        if result.get("error"):
            return ""
        article = result.get("article") or {}
        return article.get("text") or ""


class Recite(_Retrieval):
    name = "recite"

    def estimate_tokens(self, item: dict[str, Any]) -> tuple[int, int]:
        return (0, 0)

    def answer(self, item: dict[str, Any], call: Call, log: dict[str, Any]) -> str:
        return self._article_text(item, None, log)


class _ContextRag(_Retrieval):
    system_prompt = RAG_SYSTEM_PROMPT
    dated = False

    def estimate_tokens(self, item: dict[str, Any]) -> tuple[int, int]:
        ctx = len(item["gold"]["text"])
        return (tokens(len(self.system_prompt) + len(item["question"]) + ctx + 40),
                tokens(len(item["gold"]["text"])) + OUTPUT_TOKEN_OVERHEAD)

    def answer(self, item: dict[str, Any], call: Call, log: dict[str, Any]) -> str:
        text = self._article_text(item, item["as_of"] if self.dated else None, log)
        if not text:
            log["context_empty"] = True
        completion = call([
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"{item['question']}\n\n<context>\n{text}\n</context>"},
        ])
        return completion.text


class CurrentRag(_ContextRag):
    name = "current"
    dated = False


class PitRag(_ContextRag):
    name = "pit"
    dated = True
