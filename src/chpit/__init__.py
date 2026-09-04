"""CH-PiT: Swiss point-in-time law benchmark -- the scorer, question
templates, report aggregation and the `core` split. Pure Python, stdlib only.

    from chpit import score
    verdict = score.score(answer, item["gold"]["text"], item["distractor"]["text"])
"""
from chpit import core_split, report, score, templates  # noqa: F401
from chpit.score import Verdict, discriminating_units, normalise, units  # noqa: F401
from chpit.score import score as score_answer  # noqa: F401
from chpit.templates import question  # noqa: F401

__version__ = "2026.9.0"
