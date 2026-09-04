"""Tests for chpit.templates: question formatting in de/fr/it. Pure."""
import datetime

from chpit import templates



def test_format_date_de():
    assert templates.format_date(datetime.date(2020, 12, 31), "de") == "31. Dezember 2020"


def test_format_date_fr():
    assert templates.format_date(datetime.date(2020, 12, 31), "fr") == "31 décembre 2020"


def test_format_date_it():
    assert templates.format_date(datetime.date(2020, 12, 31), "it") == "31 dicembre 2020"


def test_format_date_no_leading_zero():
    assert templates.format_date(datetime.date(2021, 1, 1), "de") == "1. Januar 2021"


def test_question_de():
    q = templates.question("de", "336", "OR", "220", datetime.date(2020, 12, 31))
    assert q == (
        "Wie lautet Art. 336 OR (SR 220) in der am 31. Dezember 2020 "
        "geltenden Fassung? Zitiere den Wortlaut."
    )


def test_question_fr():
    q = templates.question("fr", "336", "CO", "220", datetime.date(2020, 12, 31))
    assert q == (
        "Quel est le texte de l'art. 336 CO (RS 220) en vigueur le "
        "31 décembre 2020 ? Citez-le mot à mot."
    )


def test_question_it():
    q = templates.question("it", "336", "CO", "220", datetime.date(2020, 12, 31))
    assert q == (
        "Qual è il testo dell'art. 336 CO (RS 220) in vigore il "
        "31 dicembre 2020? Citalo alla lettera."
    )
