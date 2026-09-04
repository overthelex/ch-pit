"""Stratified sampling of one language's items by `kind`.

`core-{lang}.jsonl` is already the fixed subset every published baseline
runs on, so a run over `core` passes `sample_per_lang=0` (everything).
Sampling is kept for pilots and for runs over the full files: independent
per language, `random.Random(f"{seed}:{lang}:llm")`, stratified by `kind`
(`before`/`after`) so a systematic bias between the two halves cannot hide
inside an unbalanced sample. Pure function of (seed, lang, item set).
"""
from __future__ import annotations

import random
from typing import Any

_KINDS = ("before", "after")


def sample_lang(items: list[dict[str, Any]], sample_per_lang: int,
                rng: random.Random) -> list[dict[str, Any]]:
    """Stratified sample of ITEMS (one language) by `kind`; each kind gets
    roughly half, a short bucket hands its shortfall to a shared shuffled
    leftover pool, so the size is always min(sample_per_lang, len(items)).
    Buckets are sorted by id before the shuffle, so the result depends only
    on the RNG seed and the item set, never on input order."""
    by_kind: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        by_kind.setdefault(item.get("kind", "unknown"), []).append(item)
    # Shuffle buckets in a fixed order so the RNG stream is consumed the
    # same way whatever order the items arrived in.
    for kind in sorted(by_kind):
        bucket = by_kind[kind]
        bucket.sort(key=lambda it: it["id"])
        rng.shuffle(bucket)

    kinds = [k for k in _KINDS if k in by_kind]
    kinds += sorted(k for k in by_kind if k not in _KINDS)
    if not kinds:
        return []

    total_available = sum(len(by_kind[k]) for k in kinds)
    target = min(sample_per_lang, total_available)

    base = target // len(kinds)
    remainder = target - base * len(kinds)
    quotas = {k: base for k in kinds}
    for k in kinds[:remainder]:
        quotas[k] += 1

    picked: list[dict[str, Any]] = []
    leftover: list[dict[str, Any]] = []
    for k in kinds:
        bucket = by_kind[k]
        take = min(quotas[k], len(bucket))
        picked.extend(bucket[:take])
        leftover.extend(bucket[take:])

    shortfall = target - len(picked)
    if shortfall > 0:
        rng.shuffle(leftover)
        picked.extend(leftover[:shortfall])

    picked.sort(key=lambda it: it["id"])
    return picked


def sample_items(by_lang: dict[str, list[dict[str, Any]]], langs: tuple[str, ...],
                 sample_per_lang: int, seed: int) -> list[dict[str, Any]]:
    """The concatenated, id-sorted sample over LANGS. `sample_per_lang <= 0`
    means every item (the `core` case)."""
    sample: list[dict[str, Any]] = []
    for lang in langs:
        items = by_lang.get(lang, [])
        if sample_per_lang <= 0:
            sample.extend(items)
        else:
            rng = random.Random(f"{seed}:{lang}:llm")
            sample.extend(sample_lang(items, sample_per_lang, rng))
    sample.sort(key=lambda it: it["id"])
    return sample
