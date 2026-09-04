"""runner.run(): cost gate, scored lines, per-item errors, retries, the
max_tokens ladder, crash safety and resume -- with a fake provider."""
import json
import os
import pathlib
import threading
import time

import pytest

from chpit import cli, modes, resume, runner
from chpit.provider import Completion, RetryableError

GOLD = ("1 Die Kündigung ist nichtig, wenn sie missbräuchlich erfolgt.\n"
        "2 Die Kündigungsfrist beträgt drei Monate, sofern nichts anderes vereinbart wurde.\n"
        "3 Der Arbeitnehmer kann die Kündigung innerhalb von 180 Tagen gerichtlich anfechten.")
DISTRACTOR = ("1 Die Kündigung ist nichtig, wenn sie missbräuchlich erfolgt.\n"
              "2 Die Kündigungsfrist richtet sich nach den Bestimmungen des Einzelarbeitsvertrags.\n"
              "3 Der Arbeitnehmer kann die Kündigung nur durch eine schriftliche Klage anfechten.")
PRICES = {"m/one": {"in": 1.0, "out": 5.0}, "m/two": {"in": 3.0, "out": 15.0}}


def _item(i: int, lang: str = "de", kind: str = "after") -> dict:
    return {"id": f"{lang}{i:04d}", "lang": lang, "kind": kind, "as_of": "2022-01-01",
            "gold_is_current": False, "question": f"Wie lautet Art. {i} OR am 1. Januar 2022?",
            "gold": {"text": GOLD}, "distractor": {"text": DISTRACTOR}}


def _by_lang(n: int = 4) -> dict:
    return {"de": [_item(i, kind="before" if i % 2 else "after") for i in range(n)]}


class FakeProvider:
    """Answers with the gold text; records every call."""

    def __init__(self, answer: str = GOLD, fail_ids: set[str] | None = None,
                 retryable_first_n: int = 0, empty_first_n: int = 0, sleep: float = 0.0):
        self.calls: list[dict] = []
        self.answer = answer
        self.retryable_left = retryable_first_n
        self.empty_left = empty_first_n
        self.sleep = sleep

    def complete(self, model, messages, *, tools=None, tool_choice=None,
                 max_tokens=2048, temperature=0.0) -> Completion:
        self.calls.append({"model": model, "messages": messages, "max_tokens": max_tokens})
        if self.sleep:
            time.sleep(self.sleep)
        if self.retryable_left > 0:
            self.retryable_left -= 1
            raise RetryableError("429 rate limited")
        if self.empty_left > 0:
            self.empty_left -= 1
            return Completion(text="", finish_reason="length", input_tokens=10, output_tokens=max_tokens)
        return Completion(text=self.answer, input_tokens=100, output_tokens=200, cost_usd=0.001,
                          served_model=model + ":served", provider_name="Fake")


class NoCallProvider:
    def complete(self, *a, **k):
        raise AssertionError("provider must not be called on the gate path")


def test_gate_prints_estimate_and_never_calls_provider(tmp_path, capsys):
    rep = runner.run(_by_lang(), tmp_path, mode=modes.ClosedBook(), models=("m/one",),
                     prices=PRICES, provider=NoCallProvider(), confirm=False)
    assert rep.confirmed is False and rep.sample_size == 4
    est = json.loads(capsys.readouterr().out)
    assert est["m/one"]["items"] == 4 and est["total_usd"] > 0
    assert not list(tmp_path.iterdir())


def test_estimate_uses_chars_over_four_and_sums_models():
    items = [_item(1)]
    est = runner.estimate(items, ("m/one", "m/two"), modes.ClosedBook(), PRICES)
    i, o = modes.ClosedBook().estimate_tokens(items[0])
    assert est["m/one"]["input_tokens"] == i and est["m/one"]["output_tokens"] == o
    assert est["total_usd"] == pytest.approx(est["m/one"]["usd"] + est["m/two"]["usd"])


def test_unpriced_model_fails_on_the_gate_before_any_call(tmp_path):
    with pytest.raises(ValueError, match="no price for model 'm/nope'"):
        runner.run(_by_lang(), tmp_path, mode=modes.ClosedBook(), models=("m/nope",),
                   prices=PRICES, provider=NoCallProvider(), confirm=True)


def test_confirmed_run_writes_one_scored_line_per_item_with_usage(tmp_path):
    prov = FakeProvider()
    rep = runner.run(_by_lang(), tmp_path, mode=modes.ClosedBook(), models=("m/one",),
                     prices=PRICES, provider=prov, confirm=True)
    lines = resume.read_jsonl_file(tmp_path / "results-closed-one.jsonl")
    assert len(lines) == 4 and {l["verdict"]["label"] for l in lines} == {"grounded_correct"}
    assert lines[0]["system"] == "one/closed" and lines[0]["served_model"] == "m/one:served"
    assert lines[0]["cost_usd"] == pytest.approx(0.001)
    assert rep.actual["m/one"]["answered"] == 4 and rep.actual["m/one"]["errors"] == 0
    assert rep.actual_total_usd == pytest.approx(0.004)
    assert rep.actual["m/one"]["usd_from_tokens"] == pytest.approx(400 / 1e6 * 1.0 + 800 / 1e6 * 5.0)
    report = json.loads((tmp_path / "run-report-closed.json").read_text())
    assert report["mode"] == "closed" and report["actual"]["m/one"]["answered"] == 4
    # closed-book prompt verbatim, question as the user turn
    m = prov.calls[0]["messages"]
    assert m[0] == {"role": "system", "content": modes.CLOSED_BOOK_SYSTEM_PROMPT}
    assert m[1]["content"].startswith("Wie lautet Art.")


class FailOn:
    def __init__(self, inner, bad_id_prefix: str):
        self.inner, self.bad = inner, bad_id_prefix

    def complete(self, model, messages, **kw):
        if self.bad in messages[-1]["content"]:
            raise RuntimeError("boom")
        return self.inner.complete(model, messages, **kw)


def test_provider_error_is_recorded_per_item_and_run_continues(tmp_path):
    prov = FailOn(FakeProvider(), "Art. 2 ")
    rep = runner.run(_by_lang(), tmp_path, mode=modes.ClosedBook(), models=("m/one",),
                     prices=PRICES, provider=prov, confirm=True)
    lines = resume.read_jsonl_file(tmp_path / "results-closed-one.jsonl")
    bad = [l for l in lines if "error" in l]
    assert len(bad) == 1 and bad[0]["error"].startswith("RuntimeError: boom")
    assert bad[0]["verdict"]["label"] == "ungrounded" and bad[0]["answer"] == ""
    assert rep.actual["m/one"]["answered"] == 3 and rep.actual["m/one"]["errors"] == 1


def test_retryable_error_is_retried_and_latency_excludes_backoff(tmp_path, monkeypatch):
    slept = []
    monkeypatch.setattr(runner.time, "sleep", lambda s: slept.append(s))
    prov = FakeProvider(retryable_first_n=2, sleep=0.01)
    runner.run({"de": [_item(1)]}, tmp_path, mode=modes.ClosedBook(), models=("m/one",),
               prices=PRICES, provider=prov, confirm=True)
    line = resume.read_jsonl_file(tmp_path / "results-closed-one.jsonl")[0]
    # the patch also catches the fake provider's own 0.01 s work sleeps
    assert "error" not in line and line["retries"] == 2 and [x for x in slept if x >= 1] == [1, 2]
    assert len(prov.calls) == 3 and line["latency_s"] < 1.0


def test_empty_or_truncated_reply_escalates_the_max_tokens_ladder(tmp_path):
    prov = FakeProvider(empty_first_n=2)
    runner.run({"de": [_item(1)]}, tmp_path, mode=modes.ClosedBook(), models=("m/one",),
               prices=PRICES, provider=prov, confirm=True)
    assert [c["max_tokens"] for c in prov.calls] == [2048, 4096, 8192]
    line = resume.read_jsonl_file(tmp_path / "results-closed-one.jsonl")[0]
    assert line["max_tokens_used"] == 8192 and line["verdict"]["label"] == "grounded_correct"


class CrashOnNth:
    def __init__(self, n: int):
        self.n, self.calls = n, 0

    def complete(self, model, messages, **kw):
        self.calls += 1
        if self.calls == self.n:
            raise KeyboardInterrupt
        return Completion(text=GOLD, input_tokens=1, output_tokens=1)


def test_crash_leaves_earlier_lines_on_disk_and_rerun_resumes(tmp_path):
    with pytest.raises(KeyboardInterrupt):
        runner.run(_by_lang(), tmp_path, mode=modes.ClosedBook(), models=("m/one",),
                   prices=PRICES, provider=CrashOnNth(3), confirm=True)
    out = tmp_path / "results-closed-one.jsonl"
    assert len(resume.read_jsonl_file(out)) == 2
    prov = FakeProvider()
    runner.run(_by_lang(), tmp_path, mode=modes.ClosedBook(), models=("m/one",),
               prices=PRICES, provider=prov, confirm=True)
    assert len(prov.calls) == 2
    assert len({l["id"] for l in resume.read_jsonl_file(out)}) == 4


def test_errored_item_is_re_asked_on_resume_and_the_new_line_supersedes(tmp_path):
    out = tmp_path / "results-closed-one.jsonl"
    out.write_text(json.dumps({"id": "de0001", "system": "one/closed", "lang": "de",
                               "answer": "", "error": "old", "input_tokens": 0,
                               "output_tokens": 0, "verdict": {"label": "ungrounded"}}) + "\n")
    prov = FakeProvider()
    rep = runner.run({"de": [_item(1)]}, tmp_path, mode=modes.ClosedBook(), models=("m/one",),
                     prices=PRICES, provider=prov, confirm=True)
    lines = resume.read_jsonl_file(out)
    assert len(lines) == 2 and len(prov.calls) == 1
    assert rep.actual["m/one"]["answered"] == 1 and rep.actual["m/one"]["errors"] == 0


def test_each_line_is_fsynced(tmp_path, monkeypatch):
    synced = []
    monkeypatch.setattr(runner.os, "fsync", lambda fd: synced.append(fd))
    runner.run(_by_lang(3), tmp_path, mode=modes.ClosedBook(), models=("m/one",),
               prices=PRICES, provider=FakeProvider(), confirm=True)
    assert len(synced) == 3


class ConcurrencyProbe(FakeProvider):
    """Counts how many complete() calls overlap (fsync on the writer side
    makes wall-clock thresholds flaky, so parallelism is measured here)."""

    def __init__(self):
        super().__init__(sleep=0.05)
        self.active = 0
        self.peak = 0
        self.lock = threading.Lock()

    def complete(self, *a, **k):
        with self.lock:
            self.active += 1
            self.peak = max(self.peak, self.active)
        try:
            return super().complete(*a, **k)
        finally:
            with self.lock:
                self.active -= 1


def test_workers_run_in_parallel_and_still_write_every_line(tmp_path):
    prov = ConcurrencyProbe()
    runner.run(_by_lang(8), tmp_path, mode=modes.ClosedBook(), models=("m/one",),
               prices=PRICES, provider=prov, confirm=True, workers=8)
    assert prov.peak >= 4
    assert len({l["id"] for l in resume.read_jsonl_file(tmp_path / "results-closed-one.jsonl")}) == 8


def test_cli_run_exits_2_without_confirm(tmp_path, monkeypatch, capsys):
    items_dir = tmp_path / "items"; items_dir.mkdir()
    (items_dir / "core-de.jsonl").write_text("".join(json.dumps(_item(i)) + "\n" for i in range(2)))
    (tmp_path / "prices.json").write_text(json.dumps(PRICES))
    monkeypatch.delenv("CHPIT_CONFIRM", raising=False)
    rc = cli.main(["run", "--items", str(items_dir), "--out", str(tmp_path / "out"), "--mode", "closed",
                   "--models", "m/one", "--langs", "de", "--prices", str(tmp_path / "prices.json")])
    assert rc == 2 and json.loads(capsys.readouterr().out)["m/one"]["items"] == 2


def test_zero_reported_cost_falls_back_to_token_pricing(tmp_path):
    class ZeroCost(FakeProvider):
        def complete(self, *a, **k):
            c = super().complete(*a, **k)
            return Completion(text=c.text, input_tokens=c.input_tokens, output_tokens=c.output_tokens,
                              cost_usd=0.0, served_model=c.served_model)
    rep = runner.run(_by_lang(2), tmp_path, mode=modes.ClosedBook(), models=("m/one",),
                     prices=PRICES, provider=ZeroCost(), confirm=True)
    a = rep.actual["m/one"]
    assert a["usd_reported"] == 0.0 and a["usd"] == pytest.approx(a["usd_from_tokens"]) and a["usd"] > 0
