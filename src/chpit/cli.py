"""`chpit` command line.

  chpit score  --answer FILE --item FILE            score one answer
  chpit report --results F [F...] --items DIR       tables per (lang, system)
  chpit sample --items DIR [--split core|full] ...  print the sample's ids
  chpit prices [--out prices.json]                  OpenRouter price table
  chpit run    --items DIR --out DIR --mode closed|current|pit|agentic
               --models a/b,c/d [--prices prices.json] ...
               (spends money only with CHPIT_CONFIRM=1)
  chpit recite --items DIR --out DIR                the no-model baseline
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import pathlib
import sys
from typing import Any

from chpit import items as items_mod
from chpit import report, runner, sampling, score

log = logging.getLogger("chpit")

DEFAULT_MODELS = ("anthropic/claude-sonnet-5", "anthropic/claude-haiku-4.5",
                  "openai/gpt-5.6-terra", "deepseek/deepseek-v4-pro")


def _langs(s: str) -> tuple[str, ...]:
    return tuple(p.strip() for p in s.split(",") if p.strip())


def _load_prices(path: str | None) -> dict[str, dict[str, float]]:
    if path:
        return json.loads(pathlib.Path(path).read_text())
    from chpit.openrouter import fetch_prices
    return fetch_prices()


def _mode(name: str, args: argparse.Namespace):
    from chpit import modes
    if name == "closed":
        return modes.ClosedBook()
    from chpit.mcp_client import McpClient, article_fetcher
    mcp = McpClient(url=args.mcp_url, bearer=os.environ.get("LAWRIDER_MCP_TOKEN", ""),
                    cache_file=pathlib.Path(args.out) / "mcp-cache.jsonl")
    fetch = article_fetcher(mcp)
    if name == "recite":
        return modes.Recite(fetch)
    if name == "current":
        return modes.CurrentRag(fetch)
    if name == "pit":
        return modes.PitRag(fetch)
    if name == "agentic":
        from chpit.agentic import Agentic
        return Agentic(mcp, max_tool_calls=args.max_tool_calls)
    raise SystemExit(f"unknown mode {name!r}")


def cmd_score(args: argparse.Namespace) -> int:
    item = json.loads(pathlib.Path(args.item).read_text())
    answer = pathlib.Path(args.answer).read_text()
    v = score.score(answer, item["gold"]["text"], item["distractor"]["text"])
    print(json.dumps(v.__dict__, ensure_ascii=False, indent=2))
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    argv = ["--results", *args.results]
    if args.items:
        argv += ["--items", args.items]
    if args.rescore:
        argv.append("--rescore")
    if args.out:
        argv += ["--out", args.out]
    if args.tools:
        argv.append("--tools")
    if args.hard_from:
        argv += ["--hard-from", args.hard_from]
    return report.main(argv)


def cmd_sample(args: argparse.Namespace) -> int:
    by_lang = items_mod.read_items(args.items, _langs(args.langs), args.split)
    sample = sampling.sample_items(by_lang, _langs(args.langs), args.sample_per_lang, args.seed)
    for it in sample:
        print(it["id"])
    print(f"# {len(sample)} items", file=sys.stderr)
    return 0


def cmd_prices(args: argparse.Namespace) -> int:
    prices = _load_prices(None)
    text = json.dumps(prices, indent=2, sort_keys=True)
    if args.out:
        pathlib.Path(args.out).write_text(text)
        print(f"{len(prices)} models -> {args.out}", file=sys.stderr)
    else:
        print(text)
    return 0


def cmd_run(args: argparse.Namespace, mode_name: str | None = None) -> int:
    mode_name = mode_name or args.mode
    langs = _langs(args.langs)
    models = _langs(args.models) if mode_name != "recite" else ("recite",)
    by_lang = items_mod.read_items(args.items, langs, args.split)
    prices = _load_prices(args.prices) if mode_name != "recite" else {"recite": {"in": 0.0, "out": 0.0}}
    confirm = os.environ.get("CHPIT_CONFIRM") == "1" or mode_name == "recite"
    mode = _mode(mode_name, args)
    provider: Any = None
    if confirm and mode_name != "recite":
        from chpit.openrouter import OpenRouterProvider
        extra: dict[str, Any] = {}
        if args.reasoning_effort == "none":
            extra["reasoning"] = {"exclude": True, "effort": "low"}
        elif args.reasoning_effort:
            extra["reasoning"] = {"effort": args.reasoning_effort}
        provider = OpenRouterProvider(extra_body=extra)
    elif mode_name == "recite":
        provider = _NoProvider()
    rep = runner.run(by_lang, args.out, mode=mode, models=models, prices=prices, langs=langs,
                     sample_per_lang=args.sample_per_lang, seed=args.seed, provider=provider,
                     confirm=confirm, workers=args.workers)
    if not rep.confirmed:
        log.info("cost estimate only (set CHPIT_CONFIRM=1 to run for real)")
        return 2
    log.info("run %s: sample=%d models=%s usd=%.4f", rep.mode, rep.sample_size,
             ",".join(models), rep.actual_total_usd or 0.0)
    return 0


class _NoProvider:
    def complete(self, *a: Any, **k: Any):  # pragma: no cover - recite never calls it
        raise RuntimeError("recite mode makes no model calls")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="chpit", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("score"); s.add_argument("--answer", required=True); s.add_argument("--item", required=True)
    s.set_defaults(fn=cmd_score)

    r = sub.add_parser("report")
    r.add_argument("--results", nargs="+", required=True); r.add_argument("--items")
    r.add_argument("--rescore", action="store_true"); r.add_argument("--out"); r.add_argument("--tools", action="store_true")
    r.add_argument("--hard-from", help="recite results file for the date-sensitive split")
    r.set_defaults(fn=cmd_report)

    def common(x: argparse.ArgumentParser) -> None:
        x.add_argument("--items", required=True); x.add_argument("--langs", default="de,fr,it")
        x.add_argument("--split", default="core", choices=sorted(items_mod.SPLIT_PREFIX))
        x.add_argument("--sample-per-lang", type=int, default=0, help="0 = every item of the split")
        x.add_argument("--seed", type=int, default=20260825)

    sm = sub.add_parser("sample"); common(sm); sm.set_defaults(fn=cmd_sample)

    pr = sub.add_parser("prices"); pr.add_argument("--out"); pr.set_defaults(fn=cmd_prices)

    def run_common(x: argparse.ArgumentParser) -> None:
        common(x)
        x.add_argument("--out", required=True)
        x.add_argument("--workers", type=int, default=4)
        x.add_argument("--mcp-url", default="https://mcp.lawrider.ch/v2/mcp")
        x.add_argument("--max-tool-calls", type=int, default=4)

    rn = sub.add_parser("run"); run_common(rn)
    rn.add_argument("--mode", required=True, choices=["closed", "current", "pit", "agentic"])
    rn.add_argument("--models", default=",".join(DEFAULT_MODELS))
    rn.add_argument("--prices", help="JSON {model: {in, out}} USD per 1M tokens; default: fetch from OpenRouter")
    rn.add_argument("--reasoning-effort", default="low", choices=["none", "low", "medium", "high"],
                    help="OpenRouter reasoning.effort for every call (default: low; recorded in the run report)")
    rn.set_defaults(fn=cmd_run)

    rc = sub.add_parser("recite"); run_common(rc)
    rc.set_defaults(fn=lambda a: cmd_run(a, "recite"))
    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
