import json

import pytest

from chpit import publish

GOLD = ("1 Die Kündigung ist nichtig, wenn sie missbräuchlich erfolgt.\n"
        "2 Die Kündigungsfrist beträgt drei Monate, sofern nichts anderes vereinbart wurde.")
DISTRACTOR = ("1 Die Kündigung ist nichtig, wenn sie missbräuchlich erfolgt.\n"
              "2 Die Kündigungsfrist richtet sich nach den Bestimmungen des Einzelarbeitsvertrags.")


def _item(lang: str, i: int, core: bool) -> dict:
    return {"build": "v1", "core": core, "id": f"{lang}{i:04d}", "lang": lang, "act_id": 7,
            "sr_number": "220", "abbreviation": "OR", "article_number": "336", "e_id": "art_336",
            "as_of": "2021-01-01", "kind": "after", "change_date": "2021-01-01",
            "question": "Wie lautet Art. 336 OR?", "gold_is_current": False,
            "gold": {"version_id": 1, "date_applicability": "2021-01-01", "date_end_applicability": None,
                     "eli": "https://x/1", "source": "fedlex", "text": GOLD},
            "distractor": {"version_id": 2, "date_applicability": "2015-01-01",
                           "date_end_applicability": "2020-12-31", "eli": "https://x/2",
                           "source": "fedlex", "text": DISTRACTOR},
            "source": "Fedlex (fedlex.admin.ch)", "licence": "Fedlex data may be reused free of charge with source attribution"}


def _build_dir(tmp_path, per_lang=4, core=2, version="v1"):
    d = tmp_path / "build"; d.mkdir()
    for lang in publish.LANGS:
        items = [_item(lang, i, i < core) for i in range(per_lang)]
        for it in items:
            it["build"] = version
        (d / f"bench-{lang}.jsonl").write_text("".join(json.dumps(it) + "\n" for it in items))
        (d / f"core-{lang}.jsonl").write_text("".join(json.dumps(it) + "\n" for it in items if it["core"]))
    (d / "build-report.json").write_text("{}")
    return d


def test_build_folder_writes_parquet_that_round_trips_and_a_card(tmp_path):
    d = _build_dir(tmp_path)
    out = tmp_path / "hf"
    stats = publish.build_folder(d, None, "v1", out, "---\nlicense: x\n---\n# CH-PiT\n\nbody", "| t |",
                                 expect_per_lang=4, core_per_lang=2)
    assert stats["de"] == {"full": 4, "core": 2}
    rows = publish.read_parquet(out / "data" / "fr" / "full-00000.parquet")
    assert len(rows) == 4 and rows[0] == _item("fr", 0, True) | {"build": "v1", "recite_label": None, "recite_as_of": None}
    assert len(publish.read_parquet(out / "data" / "fr" / "core-00000.parquet")) == 2
    card = (out / "README.md").read_text()
    assert card.startswith("---\nlicense: other") and "config_name: de" in card and "split: core" in card
    assert "Version v1" in card and "| t |" in card and "\n# CH-PiT\n" in card
    assert (out / "raw" / "bench-it.jsonl").exists() and (out / "raw" / "core-it.jsonl").exists()


@pytest.mark.parametrize("break_it,match", [
    (lambda d: (d / "bench-de.jsonl").write_text(""), "0 items"),
    (lambda d: (d / "core-de.jsonl").write_text(""), "core has 0"),
])
def test_validate_rejects_wrong_counts(tmp_path, break_it, match):
    d = _build_dir(tmp_path)
    break_it(d)
    with pytest.raises(publish.ValidationError, match=match):
        publish.validate(d, "v1", expect_per_lang=4, core_per_lang=2)


def test_validate_rejects_a_wrong_build_label_and_bad_core_flag(tmp_path):
    d = _build_dir(tmp_path)
    with pytest.raises(publish.ValidationError, match="build="):
        publish.validate(d, "v2", expect_per_lang=4, core_per_lang=2)
    items = [json.loads(l) for l in (d / "bench-de.jsonl").read_text().splitlines()]
    items[3]["core"] = True
    (d / "bench-de.jsonl").write_text("".join(json.dumps(it) + "\n" for it in items))
    with pytest.raises(publish.ValidationError, match="core flag"):
        publish.validate(d, "v1", expect_per_lang=4, core_per_lang=2)


def test_validate_checks_the_oracle_when_present(tmp_path):
    d = _build_dir(tmp_path)
    bad = {"id": "de0000", "system": "oracle", "lang": "de", "answer": "", "verdict": {
        "label": "ungrounded", "gold_coverage": 0, "distractor_coverage": 0, "shared_coverage": 0,
        "distractor_all_coverage": 0}}
    (d / "results-oracle.jsonl").write_text(json.dumps(bad) + "\n")
    with pytest.raises(publish.ValidationError, match="oracle is not 1.000"):
        publish.validate(d, "v1", expect_per_lang=4, core_per_lang=2)


def test_recite_labels_are_stamped_when_given(tmp_path):
    d = _build_dir(tmp_path)
    rec = tmp_path / "results-recite-recite.jsonl"
    rec.write_text(json.dumps({"id": "de0000", "system": "recite", "verdict": {"label": "grounded_wrong_version"}}) + "\n"
                   + json.dumps({"id": "de0001", "system": "recite", "error": "x", "verdict": {"label": "ungrounded"}}) + "\n")
    (tmp_path / "run-report-recite.json").write_text(json.dumps({"finished": "2026-09-04T12:00:00+00:00"}))
    out = tmp_path / "hf"
    stats = publish.build_folder(d, None, "v1", out, "# CH-PiT\n", expect_per_lang=4, core_per_lang=2, recite_file=rec)
    assert stats["recite_labels"] == 1
    rows = {r["id"]: r for r in publish.read_parquet(out / "data" / "de" / "full-00000.parquet")}
    assert rows["de0000"]["recite_label"] == "grounded_wrong_version" and rows["de0000"]["recite_as_of"] == "2026-09-04"
    assert rows["de0001"]["recite_label"] is None
    raw = [json.loads(l) for l in (out / "raw" / "bench-de.jsonl").read_text().splitlines()]
    assert raw[0]["recite_label"] == "grounded_wrong_version"
