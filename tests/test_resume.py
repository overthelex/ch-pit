import json

from chpit import resume


def _write(path, lines: list[str]) -> None:
    path.write_bytes("".join(lines).encode("utf-8"))


def test_read_jsonl_drops_only_a_truncated_final_line(tmp_path):
    p = tmp_path / "r.jsonl"
    _write(p, ['{"id": "a"}\n', '{"id": "b"}\n', '{"id": "c", "ans'])
    assert [l["id"] for l in resume.read_jsonl_file(p)] == ["a", "b"]


def test_read_jsonl_raises_on_corruption_in_the_middle(tmp_path):
    p = tmp_path / "r.jsonl"
    _write(p, ['{"id": "a"}\n', '{"id": "b", bro\n', '{"id": "c"}\n'])
    try:
        resume.read_jsonl_file(p)
    except json.JSONDecodeError:
        return
    raise AssertionError("corruption in the middle must raise")


def test_truncate_partial_line_cuts_an_unparseable_suffix(tmp_path):
    p = tmp_path / "r.jsonl"
    _write(p, ['{"id": "a"}\n', '{"id": "b", "ans'])
    resume.truncate_partial_line(p)
    assert p.read_bytes() == b'{"id": "a"}\n'


def test_truncate_partial_line_keeps_a_complete_object_missing_its_newline(tmp_path):
    p = tmp_path / "r.jsonl"
    _write(p, ['{"id": "a"}\n', '{"id": "b"}'])
    resume.truncate_partial_line(p)
    assert p.read_bytes() == b'{"id": "a"}\n{"id": "b"}\n'


def test_truncate_partial_line_removes_a_lone_unparseable_line(tmp_path):
    p = tmp_path / "r.jsonl"
    _write(p, ['{"id": "a", "an'])
    resume.truncate_partial_line(p)
    assert p.read_bytes() == b""


def test_done_ids_ignores_errored_lines(tmp_path):
    p = tmp_path / "r.jsonl"
    _write(p, ['{"id": "a"}\n', '{"id": "b", "error": "x"}\n'])
    assert resume.done_ids(p) == {"a"}


def test_last_by_id_prefers_the_last_error_free_line():
    lines = [{"id": "a", "error": "x"}, {"id": "a", "answer": "ok"}, {"id": "a", "error": "y"}]
    assert resume.last_by_id(lines)["a"] == {"id": "a", "answer": "ok"}
