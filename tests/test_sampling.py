import random

from chpit import sampling


def _items(n_before: int, n_after: int) -> list[dict]:
    items = [{"id": f"b{i:04d}", "kind": "before"} for i in range(n_before)]
    items += [{"id": f"a{i:04d}", "kind": "after"} for i in range(n_after)]
    return items


def test_sample_lang_is_deterministic_and_order_independent():
    items = _items(50, 50)
    a = sampling.sample_lang(list(items), 20, random.Random("x"))
    b = sampling.sample_lang(list(reversed(items)), 20, random.Random("x"))
    assert [i["id"] for i in a] == [i["id"] for i in b]
    assert len(a) == 20


def test_sample_lang_is_stratified_by_kind():
    picked = sampling.sample_lang(_items(80, 20), 30, random.Random(1))
    kinds = [i["kind"] for i in picked]
    assert kinds.count("before") == 15 and kinds.count("after") == 15


def test_sample_lang_odd_size_and_shortfall():
    picked = sampling.sample_lang(_items(80, 3), 31, random.Random(1))
    kinds = [i["kind"] for i in picked]
    assert len(picked) == 31 and kinds.count("after") == 3


def test_sample_lang_caps_at_available():
    assert len(sampling.sample_lang(_items(2, 2), 100, random.Random(0))) == 4


def test_sample_items_zero_means_everything_sorted_by_id():
    by_lang = {"de": _items(3, 3), "fr": _items(2, 1)}
    out = sampling.sample_items(by_lang, ("de", "fr"), 0, 1)
    assert len(out) == 9 and [i["id"] for i in out] == sorted(i["id"] for i in out)


def test_sample_items_uses_the_per_language_seed():
    by_lang = {"de": _items(40, 40)}
    a = sampling.sample_items(by_lang, ("de",), 10, 20260825)
    b = sampling.sample_items(by_lang, ("de",), 10, 20260825)
    c = sampling.sample_items(by_lang, ("de",), 10, 1)
    assert a == b and a != c
