"""Run one mode over a sample with one or more models: cost gate, retries,
crash-safe append/resume, a run report per mode.

Cost gate: `estimate()` prices every item in the sample from the mode's
chars/4 token approximation and a price table the caller supplies
(`chpit prices` fetches OpenRouter's). `run()` prints that estimate as
JSON and returns without touching the provider unless `confirm=True`; the
CLI wires that to the environment variable CHPIT_CONFIRM=1, never a flag,
so a bare `chpit run` can never accidentally spend money. main() exits 2
on the gated path.

Concurrency: `workers` threads per model answer items in parallel; a single
writer appends lines in completion order (not sample order -- resume is by
id, and report.summarise does not care about order). Every line is written
and fsynced before the next is accepted.

Result line: {id, system, model, mode, lang, answer, input_tokens,
output_tokens, cost_usd, served_model, provider, latency_s, retries,
max_tokens_used, verdict{...}, error?, ...mode extras}. `system` is
`{model_short}/{mode}` so report.summarise groups per (lang, system).
"""
from __future__ import annotations

import concurrent.futures
import dataclasses
import datetime
import json
import logging
import os
import pathlib
import threading
import time
from typing import Any

from chpit import resume, sampling, score
from chpit.modes import Mode
from chpit.provider import Completion, Provider, RetryableError

log = logging.getLogger(__name__)

RESULTS_TEMPLATE = "results-{mode}-{model_short}.jsonl"
REPORT_TEMPLATE = "run-report-{mode}.json"

MAX_RETRIES = 5
BACKOFF_SECONDS = (1, 2, 4, 8, 16)
# Escalation ladder for an empty / truncated / unparseable body: a reply
# that hit the ceiling is billed in full, so a retry at the same ceiling
# buys the same failure again (measured on reasoning models, see CARD).
MAX_TOKENS_LADDER = (2048, 4096, 8192)


def model_short(model: str) -> str:
    return model.split("/")[-1]


def _iso(dt: datetime.datetime) -> str:
    return dt.astimezone(datetime.timezone.utc).isoformat()


def _now(now: datetime.datetime | None) -> str:
    return _iso(now) if now is not None else _iso(datetime.datetime.now(datetime.timezone.utc))


def price_for(prices: dict[str, dict[str, float]], model: str) -> dict[str, float]:
    price = prices.get(model)
    if price is None:
        raise ValueError(
            f"no price for model {model!r}. Pass a price table with the model "
            f"(USD per 1M tokens, {{'in': ..., 'out': ...}}); `chpit prices` "
            f"fetches OpenRouter's current table, or add it to --prices JSON.")
    return price


@dataclasses.dataclass(frozen=True)
class RunReport:
    confirmed: bool
    mode: str
    estimate: dict[str, Any]
    actual: dict[str, Any] | None
    started: str | None
    finished: str | None
    sample_size: int
    actual_total_usd: float | None = None


def estimate(items: list[dict[str, Any]], models: tuple[str, ...], mode: Mode,
             prices: dict[str, dict[str, float]]) -> dict[str, Any]:
    """{model: {items, input_tokens, output_tokens, usd}, ..., total_usd}."""
    result: dict[str, Any] = {}
    total_usd = 0.0
    input_tokens = 0
    output_tokens = 0
    for item in items:
        i, o = mode.estimate_tokens(item)
        input_tokens += i
        output_tokens += o
    for model in models:
        price = price_for(prices, model)
        usd = input_tokens / 1_000_000 * price["in"] + output_tokens / 1_000_000 * price["out"]
        result[model] = {"items": len(items), "input_tokens": input_tokens,
                         "output_tokens": output_tokens, "usd": usd}
        total_usd += usd
    result["total_usd"] = total_usd
    return result


def _make_call(provider: Provider, model: str, stats: dict[str, Any]):
    """A closure the mode calls: provider.complete with retries, backoff,
    the max_tokens ladder and usage accounting into STATS."""

    def call(messages: list[dict[str, Any]], *, tools: list[dict[str, Any]] | None = None,
             tool_choice: str | None = None, temperature: float = 0.0) -> Completion:
        attempt = 0
        ladder = 0
        while True:
            max_tokens = MAX_TOKENS_LADDER[ladder]
            t0 = time.monotonic()
            try:
                completion = provider.complete(model, messages, tools=tools, tool_choice=tool_choice,
                                               max_tokens=max_tokens, temperature=temperature)
            except RetryableError as exc:
                stats["latency_s"] += time.monotonic() - t0
                if attempt < MAX_RETRIES:
                    stats["retries"] = attempt + 1
                    time.sleep(BACKOFF_SECONDS[min(attempt, len(BACKOFF_SECONDS) - 1)])
                    attempt += 1
                    continue
                raise
            stats["latency_s"] += time.monotonic() - t0
            stats["input_tokens"] += completion.input_tokens
            stats["output_tokens"] += completion.output_tokens
            if completion.cost_usd is not None:
                stats["cost_usd"] = (stats.get("cost_usd") or 0.0) + completion.cost_usd
            stats["served_model"] = completion.served_model or stats.get("served_model")
            stats["provider"] = completion.provider_name or stats.get("provider")
            stats["calls"] += 1
            stats["max_tokens_used"] = max(stats["max_tokens_used"], max_tokens)
            truncated = (completion.finish_reason == "length"
                         or (not completion.text.strip() and not completion.tool_calls))
            if truncated and ladder + 1 < len(MAX_TOKENS_LADDER):
                ladder += 1
                stats["retries"] += 1
                continue
            stats["retries"] = max(stats["retries"], attempt)
            return completion

    return call


def answer_item(provider: Provider, model: str, mode: Mode, item: dict[str, Any],
                prices: dict[str, dict[str, float]] | None = None) -> dict[str, Any]:
    """Answer and score one item. Any exception from the provider (after
    retries) is recorded on the line as `error`, with an empty answer that
    is still scored, so an unreachable model shows as `ungrounded` rather
    than vanishing from the count."""
    stats: dict[str, Any] = {"latency_s": 0.0, "retries": 0, "input_tokens": 0,
                             "output_tokens": 0, "calls": 0, "max_tokens_used": 0,
                             "cost_usd": None, "served_model": None, "provider": None}
    extras: dict[str, Any] = {}
    answer = ""
    error: str | None = None
    try:
        answer = mode.answer(item, _make_call(provider, model, stats), extras)
    except Exception as exc:  # noqa: BLE001 -- recorded per item, not fatal
        error = f"{type(exc).__name__}: {exc}"

    if stats["cost_usd"] is None and prices and model in prices:
        p = prices[model]
        stats["cost_usd"] = (stats["input_tokens"] / 1e6 * p["in"]
                             + stats["output_tokens"] / 1e6 * p["out"])

    verdict = score.score(answer, item["gold"]["text"], item["distractor"]["text"])
    result: dict[str, Any] = {
        "id": item["id"],
        "system": f"{model_short(model)}/{mode.name}",
        "model": model,
        "mode": mode.name,
        "lang": item["lang"],
        "answer": answer,
        "input_tokens": stats["input_tokens"],
        "output_tokens": stats["output_tokens"],
        "cost_usd": stats["cost_usd"],
        "served_model": stats["served_model"],
        "provider": stats["provider"],
        "latency_s": stats["latency_s"],
        "retries": stats["retries"],
        "calls": stats["calls"],
        "max_tokens_used": stats["max_tokens_used"],
        "verdict": {
            "label": verdict.label,
            "gold_coverage": verdict.gold_coverage,
            "distractor_coverage": verdict.distractor_coverage,
            "shared_coverage": verdict.shared_coverage,
            "distractor_all_coverage": verdict.distractor_all_coverage,
        },
    }
    result.update(extras)
    if error is not None:
        result["error"] = error
    return result


def _actual_for(out_file: pathlib.Path, sample_size: int, model: str,
                prices: dict[str, dict[str, float]]) -> tuple[dict[str, Any], float]:
    all_results = resume.read_jsonl_file(out_file)
    last = resume.last_by_id(all_results)
    answered = sum(1 for r in last.values() if "error" not in r)
    errors = sum(1 for r in last.values() if "error" in r)
    input_tokens_sum = sum(r.get("input_tokens", 0) for r in all_results)
    output_tokens_sum = sum(r.get("output_tokens", 0) for r in all_results)
    reported = [r.get("cost_usd") for r in all_results if r.get("cost_usd") is not None]
    if reported and len(reported) == len(all_results):
        usd = float(sum(reported))
    else:
        price = price_for(prices, model)
        usd = input_tokens_sum / 1e6 * price["in"] + output_tokens_sum / 1e6 * price["out"]
    return ({"items": sample_size, "answered": answered, "errors": errors,
             "input_tokens": input_tokens_sum, "output_tokens": output_tokens_sum,
             "usd": usd}, usd)


def _write_report(out_path: pathlib.Path, mode: str, est: dict[str, Any],
                  actual: dict[str, Any], total_usd: float, started: str,
                  finished: str, sample_size: int) -> None:
    (out_path / REPORT_TEMPLATE.format(mode=mode)).write_text(json.dumps({
        "mode": mode, "estimate": est, "actual": actual, "actual_total_usd": total_usd,
        "sample_size": sample_size, "started": started, "finished": finished,
    }, ensure_ascii=False, indent=2))


def run(by_lang: dict[str, list[dict[str, Any]]], out_dir: str | pathlib.Path, *,
        mode: Mode, models: tuple[str, ...], prices: dict[str, dict[str, float]],
        langs: tuple[str, ...] = ("de", "fr", "it"), sample_per_lang: int = 0,
        seed: int = 20260825, provider: Provider | None = None, confirm: bool = False,
        workers: int = 1, now: datetime.datetime | None = None) -> RunReport:
    """Sample from BY_LANG (`sample_per_lang <= 0` = everything), price the
    sample, and -- only if CONFIRM -- answer it with every model in MODELS
    via PROVIDER, crash-safe and resumable (see module docstring)."""
    sample = sampling.sample_items(by_lang, langs, sample_per_lang, seed)
    est = estimate(sample, models, mode, prices)

    if not confirm:
        print(json.dumps(est, ensure_ascii=False))
        return RunReport(confirmed=False, mode=mode.name, estimate=est, actual=None,
                         started=None, finished=None, sample_size=len(sample))
    if provider is None:
        raise ValueError("a confirmed run needs a provider")

    out_path = pathlib.Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    started = _now(now)
    actual: dict[str, Any] = {}
    total_usd = 0.0

    for model in models:
        out_file = out_path / RESULTS_TEMPLATE.format(mode=mode.name, model_short=model_short(model))
        resume.truncate_partial_line(out_file)
        skip_ids = resume.done_ids(out_file)
        todo = [item for item in sample if item["id"] not in skip_ids]
        write_lock = threading.Lock()

        with out_file.open("a", encoding="utf-8") as f:
            def _write(result: dict[str, Any]) -> None:
                with write_lock:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
                    f.flush()
                    os.fsync(f.fileno())

            if workers <= 1:
                for item in todo:
                    _write(answer_item(provider, model, mode, item, prices))
            else:
                with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
                    futures = [pool.submit(answer_item, provider, model, mode, item, prices)
                               for item in todo]
                    for fut in concurrent.futures.as_completed(futures):
                        _write(fut.result())

        actual[model], usd = _actual_for(out_file, len(sample), model, prices)
        total_usd += usd
        _write_report(out_path, mode.name, est, actual, total_usd, started, _now(now), len(sample))

    finished = _now(now)
    _write_report(out_path, mode.name, est, actual, total_usd, started, finished, len(sample))
    return RunReport(confirmed=True, mode=mode.name, estimate=est, actual=actual,
                     actual_total_usd=total_usd, started=started, finished=finished,
                     sample_size=len(sample))
