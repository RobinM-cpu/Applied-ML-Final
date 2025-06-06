import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from job_fraud_detection.data.multimodal_preprocess import strip_html, non_latin_ratio, clean_and_mark, detect_desc_lang
import pandas as pd


def test_strip_html():
    raw = "<p>Hello world</p> http://example.com &nbsp;"
    cleaned = strip_html(raw)
    assert "http" not in cleaned
    assert "<p>" not in cleaned
    assert "Hello world" in cleaned
    print("OK")


def test_non_latin_ratio():
    assert non_latin_ratio("Hello world") == 0.0
    assert non_latin_ratio("Всем привет") > 0.9
    assert non_latin_ratio("") == 1.0
    assert 0.0 < non_latin_ratio("Job offer in Нидерландах") < 1.0
    print("OK")


def test_detect_desc_lang():
    result = detect_desc_lang("This is English text.")
    assert result == "en"
    result = detect_desc_lang("")
    assert result == "unknown"
    print("OK")


def test_clean_and_mark():
    row = {
        "title": "Dev",
        "description": "Great <b>job</b>!",
        "requirements": "Python",
        "company_profile": "Cool startup",
        "benefits": "Snacks & ping pong"
    }
    text = clean_and_mark(row)
    assert "[TITLE]" in text
    assert "[DESC]" in text
    assert "Great job!" in text
    assert "<b>" not in text
    print("OK")


if __name__ == "__main__":
    test_clean_and_mark()
    test_non_latin_ratio()
    test_strip_html()
    test_detect_desc_lang()