# CH-PiT: Swiss Point-in-Time Law

A benchmark that asks one question about a legal AI system's answer: **was
it grounded in the version of the Swiss federal article that was actually
in force on the date asked?**

Fedlex keeps every consolidated edition of every federal act. CH-PiT turns
real amendment events into dated questions ("what did Art. X of act Y say
as of date D?") whose correct answer is one specific edition's text and
whose wrong answer is the adjacent edition's text, and ships a
deterministic scorer that tells the two apart from a free-text answer.

- Dataset: https://huggingface.co/datasets/overthelex/ch-pit (15,000 items,
  de/fr/it, plus the 500-per-language `core` subset)
- Card: [`CARD.md`](CARD.md) -- construction rules, fields, scorer thresholds
- Results: [`RESULTS.md`](RESULTS.md)

## Install

```
pip install "chpit @ git+https://github.com/overthelex/ch-pit.git@v2026.9.0"
```

The scorer is pure Python with no dependencies. Baseline runners need
`chpit[openrouter]`; publishing needs `chpit[hf]`.

## Score one answer

```python
from chpit import score
verdict = score.score(answer, item["gold"]["text"], item["distractor"]["text"])
verdict.label   # grounded_correct | grounded_wrong_version | ungrounded
```

## Licences

Code: MIT (`LICENSE`). Data: Fedlex, reuse free of charge with source
attribution (`DATA-LICENCE.md`).

## Cite

See `CARD.md`, "Citation".
