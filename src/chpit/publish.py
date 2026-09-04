"""Build the Hugging Face dataset folder for one CH-PiT version and, when
asked, upload it and tag it.

Layout of the published repo (`overthelex/ch-pit`):

    README.md                     CARD.md with the dataset-card YAML header
    data/{lang}/core-00000.parquet   the 500-per-language `core` split
    data/{lang}/full-00000.parquet   all 5,000 items of that language
    raw/{core,bench}-{lang}.jsonl    the builder's files, byte for byte
    results/{version}/               every baseline's results-*.jsonl, run
                                     reports, report-all.json, mcp-cache.jsonl
    build-report.json

Configs are languages (`de`, `fr`, `it`), splits are `core` and `full`, so
`load_dataset("overthelex/ch-pit", "de", split="core")` is the sample every
published baseline ran on. `gold` and `distractor` are struct columns; a
parquet row is the same dict as its JSONL line.

Validation before anything is written: 5,000 ids per language, unique
across languages, every item has a gold-only unit, every item carries
`build == version`, and the oracle results summarise to 1.000 per language.
"""
from __future__ import annotations

import json
import pathlib
import re
import shutil
from typing import Any

from chpit import items as items_mod
from chpit import report, resume, score

LANGS = ("de", "fr", "it")
RESULT_GLOBS = ("results-*.jsonl", "run-report-*.json", "report-*.json", "mcp-cache.jsonl",
                "llm-run-report.json")


class ValidationError(ValueError):
    pass


def validate(items_dir: pathlib.Path, version: str, expect_per_lang: int = 5000,
             core_per_lang: int = 500) -> dict[str, Any]:
    seen: set[str] = set()
    stats: dict[str, Any] = {}
    for lang in LANGS:
        full = items_mod.read_items(items_dir, (lang,), "full")[lang]
        core = items_mod.read_items(items_dir, (lang,), "core")[lang]
        if len(full) != expect_per_lang:
            raise ValidationError(f"{lang}: {len(full)} items, expected {expect_per_lang}")
        if len(core) != core_per_lang:
            raise ValidationError(f"{lang}: core has {len(core)} items, expected {core_per_lang}")
        core_ids = {it["id"] for it in core}
        for it in full:
            if it["id"] in seen:
                raise ValidationError(f"duplicate id {it['id']}")
            seen.add(it["id"])
            if it.get("build") != version:
                raise ValidationError(f"{it['id']}: build={it.get('build')!r}, expected {version!r}")
            if bool(it.get("core")) != (it["id"] in core_ids):
                raise ValidationError(f"{it['id']}: core flag disagrees with core-{lang}.jsonl")
            gold_only, _, _ = score.discriminating_units(it["gold"]["text"], it["distractor"]["text"])
            if not gold_only:
                raise ValidationError(f"{it['id']}: no gold-only unit")
        stats[lang] = {"full": len(full), "core": len(core)}
    oracle = items_dir / "results-oracle.jsonl"
    if oracle.exists():
        lines = resume.read_jsonl_file(oracle)
        summary = report.summarise(lines, items_mod.items_by_id(items_dir, LANGS, "full"))
        for lang in LANGS:
            row = summary.get(lang, {}).get("oracle", {}).get("all", {})
            if row.get("share_correct") != 1.0:
                raise ValidationError(f"oracle is not 1.000 on {lang}: {row}")
        stats["oracle"] = "1.000 on every language"
    else:
        stats["oracle"] = "results-oracle.jsonl not present, oracle check skipped"
    return stats


def _schema():
    import pyarrow as pa
    edition = pa.struct([
        ("version_id", pa.int64()), ("date_applicability", pa.string()),
        ("date_end_applicability", pa.string()), ("eli", pa.string()),
        ("source", pa.string()), ("text", pa.string()),
    ])
    return pa.schema([
        ("build", pa.string()), ("core", pa.bool_()), ("id", pa.string()), ("lang", pa.string()),
        ("act_id", pa.int64()), ("sr_number", pa.string()), ("abbreviation", pa.string()),
        ("article_number", pa.string()), ("e_id", pa.string()), ("as_of", pa.string()),
        ("kind", pa.string()), ("change_date", pa.string()), ("question", pa.string()),
        ("gold_is_current", pa.bool_()), ("gold", edition), ("distractor", edition),
        ("source", pa.string()), ("licence", pa.string()),
        ("recite_label", pa.string()), ("recite_as_of", pa.string()),
    ])


def write_parquet(items: list[dict[str, Any]], path: pathlib.Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(items, schema=_schema())
    pq.write_table(table, path, compression="zstd")


def read_parquet(path: pathlib.Path) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq
    return pq.read_table(path).to_pylist()


_YAML_HEADER = re.compile(r"\A---\n.*?\n---\n", re.DOTALL)


def dataset_card(card_md: str, version: str, stats: dict[str, Any], results_md: str = "") -> str:
    """CARD.md with a dataset-card YAML header (configs = languages, splits
    core/full) and a short version block after the title."""
    body = _YAML_HEADER.sub("", card_md, count=1)
    configs = "\n".join(
        f"- config_name: {lang}\n"
        f"{'  default: true' + chr(10) if lang == 'de' else ''}"
        f"  data_files:\n"
        f"  - split: core\n    path: data/{lang}/core-*.parquet\n"
        f"  - split: full\n    path: data/{lang}/full-*.parquet"
        for lang in LANGS)
    header = (
        "---\n"
        "license: other\n"
        "license_name: fedlex-open-data\n"
        "license_link: https://www.fedlex.admin.ch/\n"
        "language:\n- de\n- fr\n- it\n"
        "task_categories:\n- question-answering\n"
        "pretty_name: \"Swiss Point-in-Time Law (CH-PiT)\"\n"
        "size_categories:\n- 10K<n<100K\n"
        "tags:\n- legal\n- swiss-law\n- point-in-time\n- benchmark\n- fedlex\n"
        f"configs:\n{configs}\n"
        "---\n")
    per_lang = ", ".join(f"{l}: {stats[l]['full']} (core {stats[l]['core']})" for l in LANGS if l in stats)
    version_block = (
        f"\n> **Version {version}.** Items per language: {per_lang}. "
        f"Oracle: {stats.get('oracle', 'n/a')}. Load with "
        f"`load_dataset(\"overthelex/ch-pit\", \"de\", split=\"core\")`; "
        f"scorer and runners: https://github.com/overthelex/ch-pit.\n")
    if results_md:
        version_block += "\n### Results on `core`\n\n" + results_md + "\n"
    # insert after the H1 title line
    lines = body.lstrip("\n").split("\n", 1)
    return header + lines[0] + "\n" + version_block + ("\n" + lines[1] if len(lines) > 1 else "")


def recite_labels(recite_file: pathlib.Path) -> tuple[dict[str, str], str | None]:
    """{id: label} from a recite run (last error-free line per id) and the
    run date, for the `recite_label` / `recite_as_of` columns."""
    lines = resume.read_jsonl_file(recite_file)
    last = resume.last_by_id(lines)
    labels = {rid: r["verdict"]["label"] for rid, r in last.items() if "error" not in r}
    run_report = recite_file.parent / "run-report-recite.json"
    as_of = None
    if run_report.exists():
        as_of = (json.loads(run_report.read_text()).get("finished") or "")[:10] or None
    return labels, as_of


def build_folder(items_dir: pathlib.Path, results_dir: pathlib.Path | None, version: str,
                 out_dir: pathlib.Path, card_md: str, results_md: str = "",
                 expect_per_lang: int = 5000, core_per_lang: int = 500,
                 recite_file: pathlib.Path | None = None) -> dict[str, Any]:
    stats = validate(items_dir, version, expect_per_lang, core_per_lang)
    labels: dict[str, str] = {}
    recite_as_of: str | None = None
    if recite_file is not None:
        labels, recite_as_of = recite_labels(recite_file)
        stats["recite_labels"] = len(labels)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    for lang in LANGS:
        full = items_mod.read_items(items_dir, (lang,), "full")[lang]
        for it in full:
            it["recite_label"] = labels.get(it["id"])
            it["recite_as_of"] = recite_as_of if it["id"] in labels else None
        core = [it for it in full if it.get("core")]
        write_parquet(full, out_dir / "data" / lang / "full-00000.parquet")
        write_parquet(core, out_dir / "data" / lang / "core-00000.parquet")
        (out_dir / "raw").mkdir(parents=True, exist_ok=True)
        for split, rows in (("full", full), ("core", core)):
            with (out_dir / "raw" / items_mod.item_file(items_dir, lang, split).name).open("w", encoding="utf-8") as f:
                for it in rows:
                    f.write(json.dumps(it, ensure_ascii=False) + "\n")
    for name in ("build-report.json", "results-oracle.jsonl"):
        if (items_dir / name).exists():
            dest = out_dir / ("results" if name.startswith("results") else ".") / version if name.startswith("results") else out_dir
            dest.mkdir(parents=True, exist_ok=True)
            shutil.copy2(items_dir / name, dest / name)
    if results_dir is not None and results_dir.exists():
        dest = out_dir / "results" / version
        dest.mkdir(parents=True, exist_ok=True)
        for pattern in RESULT_GLOBS:
            for f in sorted(results_dir.glob(pattern)):
                shutil.copy2(f, dest / f.name)
    (out_dir / "README.md").write_text(dataset_card(card_md, version, stats, results_md), encoding="utf-8")
    # round-trip check: a parquet row is the (stamped) JSONL dict
    sample = items_mod.read_items(out_dir / "raw", ("de",), "core")["de"][0]
    back = read_parquet(out_dir / "data" / "de" / "core-00000.parquet")[0]
    if back != sample:
        raise ValidationError("parquet round-trip differs from the JSONL line for " + sample["id"])
    return stats


def upload(out_dir: pathlib.Path, repo_id: str, version: str, token: str | None = None,
           private: bool = True) -> str:
    from huggingface_hub import HfApi
    api = HfApi(token=token)
    api.create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)
    api.upload_folder(folder_path=str(out_dir), repo_id=repo_id, repo_type="dataset",
                      commit_message=f"CH-PiT {version}")
    try:
        api.create_tag(repo_id, tag=version, repo_type="dataset", tag_message=f"CH-PiT {version}")
    except Exception as exc:  # noqa: BLE001 -- a re-publish of the same version keeps the tag
        return f"uploaded; tag {version} not created: {exc}"
    return f"uploaded and tagged {version}"


def main(argv: list[str] | None = None) -> int:
    import argparse
    p = argparse.ArgumentParser(description="Build (and optionally upload) the HF dataset folder")
    p.add_argument("--items", required=True, help="build dir with bench-/core-{lang}.jsonl, build-report.json, results-oracle.jsonl")
    p.add_argument("--results", help="dir with the baseline results-*.jsonl and run reports")
    p.add_argument("--version", required=True, help="e.g. v2026.09")
    p.add_argument("--out", required=True, help="folder to build")
    p.add_argument("--card", default="CARD.md")
    p.add_argument("--results-md", help="markdown file with the results table to embed in the card")
    p.add_argument("--recite", help="results-recite-recite.jsonl: stamps recite_label / recite_as_of on every item")
    p.add_argument("--repo", default="overthelex/ch-pit")
    p.add_argument("--upload", action="store_true", help="upload to --repo and tag --version")
    p.add_argument("--public", action="store_true", help="create the repo public (default private)")
    args = p.parse_args(argv)
    results_md = pathlib.Path(args.results_md).read_text(encoding="utf-8") if args.results_md else ""
    stats = build_folder(pathlib.Path(args.items), pathlib.Path(args.results) if args.results else None,
                         args.version, pathlib.Path(args.out),
                         pathlib.Path(args.card).read_text(encoding="utf-8"), results_md,
                         recite_file=pathlib.Path(args.recite) if args.recite else None)
    print(json.dumps(stats, ensure_ascii=False))
    if args.upload:
        print(upload(pathlib.Path(args.out), args.repo, args.version, private=not args.public))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
