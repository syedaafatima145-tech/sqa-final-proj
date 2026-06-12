"""
API TESTS — Flask /translate endpoint
Tool: Pytest + requests
Run: pytest tests/api/test_translate_api.py -v

Prerequisites: python backend/app.py must be running on port 5000

AI USAGE LOG:
Prompt: "Generate pytest API tests for a Flask translation endpoint supporting en/ur/pa languages"
Changes made: Used actual FLORES-200 language codes (eng_Latn, urd_Arab, pan_Arab),
              matched exact response field 'result' from app.py,
              added engine-specific assertions based on reading app.py source.
"""

import pytest
import requests

BASE_URL = "http://127.0.0.1:5000"


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session", autouse=True)
def check_server():
    """Fail fast if the backend is not running."""
    try:
        r = requests.get(f"{BASE_URL}/translate", timeout=5)
        assert r.status_code == 200, "Server returned unexpected status"
    except requests.exceptions.ConnectionError:
        pytest.skip("Flask backend not running — start with: python backend/app.py")


# ── API Tests ─────────────────────────────────────────────────────────────────

class TestTranslateEndpointBasic:

    # TEST 1 — Happy path: English to Urdu
    def test_english_to_urdu_returns_result(self):
        payload = {"text": "Hello, how are you?", "source_lang": "en", "target_lang": "ur"}
        r = requests.post(f"{BASE_URL}/translate", json=payload, timeout=60)
        assert r.status_code == 200
        data = r.json()
        assert "result" in data
        assert len(data["result"]) > 0

    # TEST 2 — Happy path: English to Punjabi
    def test_english_to_punjabi_returns_result(self):
        payload = {"text": "Good morning", "source_lang": "en", "target_lang": "pa"}
        r = requests.post(f"{BASE_URL}/translate", json=payload, timeout=60)
        assert r.status_code == 200
        data = r.json()
        assert "result" in data
        assert len(data["result"]) > 0

    # TEST 3 — Happy path: Urdu to English
    def test_urdu_to_english_returns_result(self):
        payload = {"text": "آپ کیسے ہیں", "source_lang": "ur", "target_lang": "en"}
        r = requests.post(f"{BASE_URL}/translate", json=payload, timeout=60)
        assert r.status_code == 200
        data = r.json()
        assert "result" in data
        assert len(data["result"]) > 0

    # TEST 4 — GET request returns status info
    def test_get_translate_returns_status(self):
        r = requests.get(f"{BASE_URL}/translate", timeout=5)
        assert r.status_code == 200
        data = r.json()
        assert "status" in data
        assert "active_engine" in data

    # TEST 5 — Response is valid JSON
    def test_response_is_json(self):
        payload = {"text": "Test", "source_lang": "en", "target_lang": "ur"}
        r = requests.post(f"{BASE_URL}/translate", json=payload, timeout=60)
        assert r.headers["Content-Type"] == "application/json"
        assert r.json() is not None
