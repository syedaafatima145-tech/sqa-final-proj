"""
NEGATIVE / EDGE CASE TESTS — Flask /translate endpoint
Tool: Pytest + requests
Run: pytest tests/negative/test_edge_cases.py -v

AI USAGE LOG:
Prompt: "Generate negative and edge case tests for a multilingual translation API"
Changes made: Added specific checks for our M2M100 model behavior with empty strings,
              very long inputs, unicode edge cases, and missing fields.
              Removed tests assuming 400 status for same-language (our app returns a result anyway).
"""

import pytest
import requests

BASE_URL = "http://127.0.0.1:5000"


@pytest.fixture(scope="session", autouse=True)
def check_server():
    try:
        requests.get(f"{BASE_URL}/translate", timeout=5)
    except requests.exceptions.ConnectionError:
        pytest.skip("Flask backend not running")


class TestNegativeCases:

    # TEST 1 — Empty text returns error
    def test_empty_text_returns_error(self):
        payload = {"text": "", "source_lang": "en", "target_lang": "ur"}
        r = requests.post(f"{BASE_URL}/translate", json=payload, timeout=10)
        data = r.json()
        assert "error" in data or r.status_code == 400

    # TEST 2 — Missing text field returns error
    def test_missing_text_field_returns_error(self):
        payload = {"source_lang": "en", "target_lang": "ur"}
        r = requests.post(f"{BASE_URL}/translate", json=payload, timeout=10)
        data = r.json()
        assert "error" in data or r.status_code in [400, 500]

    # TEST 3 — Whitespace-only text returns error
    def test_whitespace_only_text_returns_error(self):
        payload = {"text": "     ", "source_lang": "en", "target_lang": "ur"}
        r = requests.post(f"{BASE_URL}/translate", json=payload, timeout=10)
        data = r.json()
        assert "error" in data or r.status_code == 400

    # TEST 4 — Invalid language code does not crash server
    def test_invalid_language_code_does_not_crash(self):
        payload = {"text": "Hello", "source_lang": "xx", "target_lang": "yy"}
        r = requests.post(f"{BASE_URL}/translate", json=payload, timeout=30)
        assert r.status_code in [200, 400, 500]
        assert r.json() is not None  # must return JSON, not crash

    # TEST 5 — Very long text (500 words) does not crash server
    def test_very_long_text_does_not_crash(self):
        long_text = "Hello world. " * 100  # ~1300 chars
        payload = {"text": long_text, "source_lang": "en", "target_lang": "ur"}
        r = requests.post(f"{BASE_URL}/translate", json=payload, timeout=120)
        assert r.status_code in [200, 400, 500]
        assert r.json() is not None

    # TEST 6 — Numbers only input
    def test_numbers_only_input(self):
        payload = {"text": "12345 67890", "source_lang": "en", "target_lang": "ur"}
        r = requests.post(f"{BASE_URL}/translate", json=payload, timeout=60)
        assert r.status_code == 200
        assert "result" in r.json()

    # TEST 7 — Mixed script input (code-switching)
    def test_code_switched_input_en_ur(self):
        payload = {"text": "Hello میں ٹھیک ہوں", "source_lang": "en", "target_lang": "ur"}
        r = requests.post(f"{BASE_URL}/translate", json=payload, timeout=60)
        assert r.status_code == 200
        assert "result" in r.json()

    # TEST 8 — Special characters in input
    def test_special_characters_input(self):
        payload = {"text": "Hello! How are you? #test @user", "source_lang": "en", "target_lang": "ur"}
        r = requests.post(f"{BASE_URL}/translate", json=payload, timeout=60)
        assert r.status_code == 200

    # TEST 9 — Empty JSON body
    def test_empty_json_body(self):
        r = requests.post(f"{BASE_URL}/translate", json={}, timeout=10)
        assert r.status_code in [400, 500]
        assert r.json() is not None

    # TEST 10 — Single character input
    def test_single_character_input(self):
        payload = {"text": "A", "source_lang": "en", "target_lang": "ur"}
        r = requests.post(f"{BASE_URL}/translate", json=payload, timeout=60)
        assert r.status_code == 200
        assert "result" in r.json()
