"""Reading benchmark items.

A build directory holds `bench-{lang}.jsonl` (the full 5,000 per language)
and `core-{lang}.jsonl` (the fixed 500-per-language subset every published
baseline runs on). `read_items()` picks one of the two by SPLIT and returns
{lang: [item, ...]} in file order (id-sorted by the builder).
"""
from __future__ import annotations

import json
import pathlib
from typing import Any

SPLIT_PREFIX = {"full": "bench", "core": "core"}


def item_file(items_dir: str | pathlib.Path, lang: str, split: str = "core") -> pathlib.Path:
    try:
        prefix = SPLIT_PREFIX[split]
    except KeyError:
        raise ValueError(f"unknown split {split!r}; expected one of {sorted(SPLIT_PREFIX)}") from None
    return pathlib.Path(items_dir) / f"{prefix}-{lang}.jsonl"


def read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def read_items(items_dir: str | pathlib.Path, langs: tuple[str, ...],
               split: str = "core") -> dict[str, list[dict[str, Any]]]:
    """{lang: items} for every LANG from `{items_dir}/{split}-{lang}.jsonl`.
    A missing file is an error, not an empty language: a run that silently
    answered nothing for a language would still produce a report."""
    by_lang: dict[str, list[dict[str, Any]]] = {}
    for lang in langs:
        f = item_file(items_dir, lang, split)
        if not f.exists():
            raise FileNotFoundError(f"no benchmark items for lang {lang!r} (split {split!r}): {f}")
        by_lang[lang] = read_jsonl(f)
    return by_lang


def items_by_id(items_dir: str | pathlib.Path, langs: tuple[str, ...] = ("de", "fr", "it"),
                split: str = "full") -> dict[str, dict[str, Any]]:
    """{id: item} over every language, for scoring results back against
    their gold/distractor texts (report --rescore)."""
    out: dict[str, dict[str, Any]] = {}
    for lang, items in read_items(items_dir, langs, split).items():
        for item in items:
            out[item["id"]] = item
    return out
