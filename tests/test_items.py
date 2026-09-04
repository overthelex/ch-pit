import json

import pytest

from chpit import items


def test_read_items_reads_core_and_full_splits(tmp_path):
    (tmp_path / "core-de.jsonl").write_text(json.dumps({"id": "a"}) + "\n\n" + json.dumps({"id": "b"}) + "\n")
    (tmp_path / "bench-de.jsonl").write_text("".join(json.dumps({"id": f"x{i}"}) + "\n" for i in range(5)))
    assert [i["id"] for i in items.read_items(tmp_path, ("de",), "core")["de"]] == ["a", "b"]
    assert len(items.read_items(tmp_path, ("de",), "full")["de"]) == 5


def test_read_items_missing_language_file_raises(tmp_path):
    (tmp_path / "core-de.jsonl").write_text("")
    with pytest.raises(FileNotFoundError, match="'fr'"):
        items.read_items(tmp_path, ("de", "fr"), "core")


def test_unknown_split_is_an_error(tmp_path):
    with pytest.raises(ValueError):
        items.item_file(tmp_path, "de", "dev")


def test_items_by_id_spans_languages(tmp_path):
    for lang in ("de", "fr"):
        (tmp_path / f"bench-{lang}.jsonl").write_text(json.dumps({"id": f"{lang}1"}) + "\n")
    assert set(items.items_by_id(tmp_path, ("de", "fr"))) == {"de1", "fr1"}
