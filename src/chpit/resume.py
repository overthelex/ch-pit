"""Crash-safe result files.

A baseline run has already paid for every answer it received before a
crash, so results are never held in memory and written at the end: each
results file is opened in append mode and every line is written and
fsynced the moment it is scored. Re-running with the same output directory
reads the ids already answered and skips them. Three details make that
safe rather than approximately safe:

  * A line carrying `error` is NOT done: it is re-asked and the new line
    appended after the old one; readers keep the last error-free line per
    id (`last_by_id`), so the retry supersedes the failure without
    rewriting a file the run is still appending to.
  * A kill mid-write leaves a partial object with no trailing newline.
    `read_jsonl_file` drops an unparseable FINAL line only (anywhere else
    is corruption, not truncation); `truncate_partial_line` repairs the
    tail before the append handle is opened.
  * A complete object whose newline never landed is an answer already paid
    for, so `truncate_partial_line` adds the newline rather than cutting it.
"""
from __future__ import annotations

import json
import logging
import pathlib
from typing import Any

log = logging.getLogger(__name__)


def read_jsonl_file(path: pathlib.Path) -> list[dict[str, Any]]:
    """Every parseable JSON object in PATH; an unparseable final line is
    dropped with a warning, an unparseable earlier line raises."""
    lines: list[dict[str, Any]] = []
    if not path.exists():
        return lines
    raw = path.read_text(encoding="utf-8").splitlines()
    for i, line in enumerate(raw):
        line = line.strip()
        if not line:
            continue
        try:
            lines.append(json.loads(line))
        except json.JSONDecodeError:
            if i == len(raw) - 1:
                log.warning("%s: dropping a truncated final line (%d chars)", path, len(line))
                break
            raise
    return lines


def truncate_partial_line(path: pathlib.Path) -> None:
    """Repair PATH's tail when it does not end in a newline: a valid JSON
    object gets its newline back, anything else is cut."""
    if not path.exists():
        return
    data = path.read_bytes()
    if not data or data.endswith(b"\n"):
        return
    cut = data.rfind(b"\n")
    tail = data[cut + 1:]
    try:
        json.loads(tail.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        log.warning("%s: truncating %d trailing bytes with no newline", path, len(tail))
        path.write_bytes(data[:cut + 1] if cut >= 0 else b"")
    else:
        log.warning("%s: completing a %d-byte final line that lost its newline", path, len(tail))
        with path.open("ab") as f:
            f.write(b"\n")


def done_ids(results_file: pathlib.Path) -> set[str]:
    """Ids answered without error in RESULTS_FILE."""
    return {line["id"] for line in read_jsonl_file(results_file) if "error" not in line}


def last_by_id(lines: list[dict[str, Any]]) -> dict[Any, dict[str, Any]]:
    """The line that counts for each id: the last error-free one, or the
    last errored one when no attempt succeeded."""
    out: dict[Any, dict[str, Any]] = {}
    for r in lines:
        previous = out.get(r["id"])
        if previous is None or "error" not in r or "error" in previous:
            out[r["id"]] = r
    return out
