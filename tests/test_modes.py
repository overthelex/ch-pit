"""recite / current / pit with a fake fetcher, and the agentic loop with a
fake MCP client and a scripted provider."""
import json

from chpit import agentic, modes, report
from chpit.provider import Completion, ToolCall

GOLD = ("1 Die Kündigung ist nichtig, wenn sie missbräuchlich erfolgt.\n"
        "2 Die Kündigungsfrist beträgt drei Monate, sofern nichts anderes vereinbart wurde.\n"
        "3 Der Arbeitnehmer kann die Kündigung innerhalb von 180 Tagen gerichtlich anfechten.")
CURRENT = ("1 Die Kündigung ist nichtig, wenn sie missbräuchlich erfolgt.\n"
           "2 Die Kündigungsfrist richtet sich nach den Bestimmungen des Einzelarbeitsvertrags.\n"
           "3 Der Arbeitnehmer kann die Kündigung nur durch eine schriftliche Klage anfechten.")
ITEM = {"id": "de0001", "lang": "de", "kind": "before", "as_of": "2020-12-31", "sr_number": "220",
        "article_number": "336", "gold_is_current": False,
        "question": "Wie lautet Art. 336 OR (SR 220) am 31. Dezember 2020?",
        "gold": {"text": GOLD}, "distractor": {"text": CURRENT}}


def fake_fetch(item, as_of):
    """Dated lookups return the gold (old) edition; undated ones today's."""
    text = GOLD if as_of else CURRENT
    return {"as_of": as_of or "today", "version": {"version_id": 1 if as_of else 2,
            "date_applicability": "2015-01-01" if as_of else "2021-01-01"},
            "article": {"text": text}}


def echo_call(messages, **kw):
    """A 'model' that copies the context back verbatim."""
    content = messages[-1]["content"]
    ctx = content.split("<context>\n", 1)[1].rsplit("\n</context>", 1)[0] if "<context>" in content else ""
    return Completion(text=ctx, input_tokens=1, output_tokens=1)


def test_recite_returns_the_current_edition_without_a_model():
    log = {}
    answer = modes.Recite(fake_fetch).answer(ITEM, lambda *a, **k: (_ for _ in ()).throw(AssertionError("no model")), log)
    assert answer == CURRENT and log["tool_result"]["version_id"] == 2


def test_current_rag_hands_the_undated_edition_as_context():
    log = {}
    assert modes.CurrentRag(fake_fetch).answer(ITEM, echo_call, log) == CURRENT
    assert log["tool_result"]["as_of"] == "today"


def test_pit_rag_hands_the_dated_edition_as_context():
    log = {}
    assert modes.PitRag(fake_fetch).answer(ITEM, echo_call, log) == GOLD
    assert log["tool_result"]["as_of"] == "2020-12-31"


def test_empty_tool_result_is_logged():
    log = {}
    mode = modes.PitRag(lambda item, as_of: {"error": "no_edition_for_date"})
    assert mode.answer(ITEM, echo_call, log) == "" and log["context_empty"] is True
    assert log["tool_result"]["error"] == "no_edition_for_date"


class FakeMcp:
    def __init__(self):
        self.calls = []

    def call_tool(self, name, arguments):
        self.calls.append((name, arguments))
        if name != "ch_get_act_article":
            return {"editions": []}
        return fake_fetch(None, arguments.get("as_of"))


class ScriptedProvider:
    """Round 1: call the tool (with or without as_of); round 2: answer with
    the tool's text."""

    def __init__(self, pass_as_of: bool):
        self.pass_as_of = pass_as_of
        self.rounds = 0

    def __call__(self, messages, *, tools=None, tool_choice=None, temperature=0.0):
        self.rounds += 1
        if self.rounds == 1:
            assert tools and tool_choice is None
            args = {"sr_number": "220", "article": "336", "lang": "de"}
            if self.pass_as_of:
                args["as_of"] = "2020-12-31"
            return Completion(text="", tool_calls=(ToolCall("c1", "ch_get_act_article", args),),
                              raw_message={"role": "assistant", "content": None, "tool_calls": [
                                  {"id": "c1", "type": "function", "function": {
                                      "name": "ch_get_act_article", "arguments": json.dumps(args)}}]})
        tool_msg = messages[-1]
        assert tool_msg["role"] == "tool" and tool_msg["tool_call_id"] == "c1"
        return Completion(text=json.loads(tool_msg["content"])["article"]["text"])


def test_agentic_logs_whether_the_model_passed_the_date():
    mcp = FakeMcp()
    mode = agentic.Agentic(mcp, max_tool_calls=4)
    log = {}
    assert mode.answer(ITEM, ScriptedProvider(pass_as_of=True), log) == GOLD
    assert log["as_of_passed"] and log["as_of_any"] and log["sr_article_correct"] and log["n_tool_calls"] == 1
    log2 = {}
    assert mode.answer(ITEM, ScriptedProvider(pass_as_of=False), log2) == CURRENT
    assert not log2["as_of_passed"] and not log2["as_of_any"]


def test_agentic_forces_an_answer_after_the_tool_budget():
    class Looper:
        def __init__(self):
            self.n = 0

        def __call__(self, messages, *, tools=None, tool_choice=None, temperature=0.0):
            self.n += 1
            if tools is not None:
                args = {"sr_number": "220", "article": "336"}
                return Completion(text="", tool_calls=(ToolCall(f"c{self.n}", "ch_get_act_article", args),))
            assert tool_choice == "none"
            return Completion(text="final")

    mode = agentic.Agentic(FakeMcp(), max_tool_calls=2)
    log = {}
    assert mode.answer(ITEM, Looper(), log) == "final" and log["n_tool_calls"] == 2


def test_schema_drift_detects_missing_property_and_new_required():
    live = [{"name": "ch_get_act_article", "inputSchema": {"properties": {"sr_number": {}, "article": {}},
                                                           "required": ["sr_number", "article", "canton"]}},
            {"name": "ch_get_act_history", "inputSchema": {"properties": {"sr_number": {}, "article": {}, "lang": {}},
                                                           "required": ["sr_number"]}}]
    drift = agentic.check_schema_drift(live)
    assert any("ch_get_act_article: properties" in d for d in drift)
    assert any("ch_get_act_article: server requires ['canton']" in d for d in drift)
    assert any("ch_get_act_text: not on the server" in d for d in drift)


def test_summarise_tools_and_markdown():
    lines = [
        {"id": "a", "system": "m/agentic", "tool_calls": [{}], "n_tool_calls": 1, "as_of_any": True,
         "as_of_passed": True, "sr_article_correct": True, "verdict": {"label": "grounded_correct"}},
        {"id": "b", "system": "m/agentic", "tool_calls": [{}], "n_tool_calls": 1, "as_of_any": False,
         "as_of_passed": False, "sr_article_correct": True, "verdict": {"label": "grounded_wrong_version"}},
        {"id": "c", "system": "m/closed", "verdict": {"label": "ungrounded"}},
    ]
    s = report.summarise_tools(lines)
    assert set(s) == {"m/agentic"} and s["m/agentic"]["n"] == 2
    assert s["m/agentic"]["share_as_of_correct"] == 0.5 and s["m/agentic"]["share_correct_when_as_of"] == 1.0
    assert s["m/agentic"]["share_correct_when_no_as_of"] == 0.0
    assert "| m/agentic | 2 |" in report.markdown_tools(s)


def test_recite_hard_split_in_summarise_and_markdown():
    items = {f"i{k}": {"lang": "de", "kind": "after", "gold_is_current": False} for k in range(4)}
    v = lambda lab: {"label": lab, "gold_coverage": 1.0, "distractor_coverage": 0.0}
    recite = [
        {"id": "i0", "system": "recite", "lang": "de", "verdict": v("grounded_correct")},
        {"id": "i1", "system": "recite", "lang": "de", "verdict": v("grounded_wrong_version")},
        {"id": "i2", "system": "recite", "lang": "de", "verdict": v("ungrounded")},
        {"id": "i3", "system": "recite", "lang": "de", "error": "x", "verdict": v("ungrounded")},
    ]
    hard, easy = report.hard_ids_from_recite(recite)
    assert hard == {"i1", "i2"} and easy == {"i0"}
    sys_lines = [{"id": f"i{k}", "system": "m/pit", "lang": "de", "verdict": v(lab)}
                 for k, lab in enumerate(["grounded_correct", "grounded_correct", "ungrounded", "grounded_correct"])]
    s = report.summarise(sys_lines, items, hard, easy)["de"]["m/pit"]["all"]
    assert s["n_recite_hard"] == 2 and s["share_correct_recite_hard"] == 0.5
    assert s["n_recite_easy"] == 1 and s["share_correct_recite_easy"] == 1.0
    md = report.markdown(report.summarise(sys_lines, items, hard, easy))
    assert "correct % (recite-hard)" in md and "| 50.0 | 2 |" in md
    assert "recite-hard" not in report.markdown(report.summarise(sys_lines, items))
