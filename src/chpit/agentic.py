"""The agentic baseline: the model gets the point-in-time tools and has to
decide, on its own, whether to pass the date the question names.

The system prompt says that the tools default to today's edition and that
the question names a date. It does NOT tell the model to pass `as_of`;
whether it does is exactly what this mode measures, so every tool call's
arguments are logged on the result line and `report --tools` reports the
share of items where the model asked for the right date.

Loop: up to `max_tool_calls` rounds of complete(tools=...) -> execute every
tool call via the MCP client (cached) -> append tool messages -> repeat;
when the model answers without tool calls, or the round budget is spent,
the last completion (forced with tool_choice="none" after the budget) is
the answer. Tool results are truncated to `max_result_chars` before they go
back into the context.

The vendored tool definitions (`tools/ch_tools.en.json`) carry English
descriptions, because the live server's are Ukrainian by product rule;
`check_schema_drift()` compares their input schemas to `tools/list` so a
server change fails the run loudly instead of silently changing what the
model sees.
"""
from __future__ import annotations

import json
import pathlib
import time
from typing import Any

from chpit.modes import Call, OUTPUT_TOKEN_OVERHEAD, tokens

TOOLS_FILE = pathlib.Path(__file__).parent / "tools" / "ch_tools.en.json"

AGENTIC_SYSTEM_PROMPT = (
    "You are a Swiss legal research assistant with tools over Fedlex, the "
    "federal law collection. The user asks for the verbatim text of an "
    "article as in force on a specific date. Use the tools to retrieve the "
    "article, then answer with the verbatim article text in the language of "
    "the question, nothing else. Note that a tool returns the edition in "
    "force today unless told otherwise."
)


def load_tools() -> list[dict[str, Any]]:
    return json.loads(TOOLS_FILE.read_text(encoding="utf-8"))


def check_schema_drift(live_tools: list[dict[str, Any]],
                       vendored: list[dict[str, Any]] | None = None) -> list[str]:
    """Names of vendored tools whose live input schema differs in property
    names or required list, or which the server no longer lists."""
    vendored = vendored or load_tools()
    live = {t["name"]: t for t in live_tools}
    drift: list[str] = []
    for tool in vendored:
        fn = tool["function"]
        name = fn["name"]
        if name not in live:
            drift.append(f"{name}: not on the server")
            continue
        live_schema = live[name].get("inputSchema") or {}
        live_props = set((live_schema.get("properties") or {}).keys())
        ours = set((fn["parameters"].get("properties") or {}).keys())
        if not ours <= live_props:
            drift.append(f"{name}: properties {sorted(ours - live_props)} not on the server")
        live_req = set(live_schema.get("required") or [])
        our_req = set(fn["parameters"].get("required") or [])
        if live_req - our_req:
            drift.append(f"{name}: server requires {sorted(live_req - our_req)}")
    return drift


class Agentic:
    name = "agentic"
    system_prompt = AGENTIC_SYSTEM_PROMPT

    def __init__(self, mcp: Any, max_tool_calls: int = 4, max_result_chars: int = 12000,
                 tools: list[dict[str, Any]] | None = None):
        self.mcp = mcp
        self.max_tool_calls = max_tool_calls
        self.max_result_chars = max_result_chars
        self.tools = tools or load_tools()

    def estimate_tokens(self, item: dict[str, Any]) -> tuple[int, int]:
        schemas = tokens(len(json.dumps(self.tools)))
        rounds = min(self.max_tool_calls, 3)
        ctx = tokens(len(item["gold"]["text"]) + 400)
        inp = sum(tokens(len(self.system_prompt) + len(item["question"])) + schemas + r * ctx
                  for r in range(rounds + 1))
        out = tokens(len(item["gold"]["text"])) + OUTPUT_TOKEN_OVERHEAD + 60 * rounds
        return inp, out

    def answer(self, item: dict[str, Any], call: Call, log: dict[str, Any]) -> str:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": item["question"]},
        ]
        calls_log: list[dict[str, Any]] = []
        answer = ""
        for round_no in range(self.max_tool_calls + 1):
            budget_left = round_no < self.max_tool_calls
            completion = call(messages, tools=self.tools if budget_left else None,
                              tool_choice=None if budget_left else "none")
            if not completion.tool_calls:
                answer = completion.text
                break
            messages.append(completion.raw_message or {
                "role": "assistant", "content": completion.text or None,
                "tool_calls": [{"id": tc.id, "type": "function",
                                "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)}}
                               for tc in completion.tool_calls]})
            for tc in completion.tool_calls:
                t0 = time.monotonic()
                entry: dict[str, Any] = {"round": round_no, "name": tc.name, "arguments": tc.arguments}
                try:
                    result = self.mcp.call_tool(tc.name, tc.arguments)
                    entry["ok"] = "error" not in result
                    if "error" in result:
                        entry["error"] = result.get("error")
                    text = json.dumps(result, ensure_ascii=False)
                except Exception as exc:  # noqa: BLE001 -- the model sees the failure
                    entry["ok"] = False
                    entry["error"] = f"{type(exc).__name__}: {exc}"
                    text = json.dumps({"error": entry["error"]})
                entry["latency_s"] = time.monotonic() - t0
                calls_log.append(entry)
                if len(text) > self.max_result_chars:
                    text = text[: self.max_result_chars] + "…[truncated]"
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": text})
        else:  # pragma: no cover - loop always breaks or exhausts with a forced answer
            pass

        log["tool_calls"] = calls_log
        log["n_tool_calls"] = len(calls_log)
        article_calls = [c for c in calls_log if c["name"] == "ch_get_act_article"]
        log["as_of_any"] = any("as_of" in c["arguments"] for c in article_calls)
        log["as_of_passed"] = any(c["arguments"].get("as_of") == item["as_of"] for c in article_calls)
        log["sr_article_correct"] = any(
            str(c["arguments"].get("sr_number")) == str(item["sr_number"])
            and str(c["arguments"].get("article")) == str(item["article_number"])
            for c in article_calls)
        return answer
