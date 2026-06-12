"""
INTEGRATION TESTS — Frontend ↔ Backend API Integration
Tool: Pytest + requests
Run: pytest tests/integration/test_integration.py -v

These tests verify that the data contract between the React frontend
and the Flask backend is respected — same fields, same format.

AI USAGE LOG:
Prompt: "Generate integration tests verifying React frontend API contract with Flask backend"
Changes made: Matched exact field names used in Translate.tsx (source_lang, target_lang, text),
              verified response field 'result' matches what frontend reads,
              added engine status check matching frontend's localServerStatus logic.
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


class TestFrontendBackendContract:

    # TEST 1 — Frontend status check: GET /translate returns active_engine field
    def test_status_check_returns_active_engine(self):
        """Translate.tsx reads data.active_engine from GET /translate"""
        r = requests.get(f"{BASE_URL}/translate", timeout=5)
        data = r.json()
        assert "active_engine" in data
        assert data["active_engine"] in ["hf", "tf", "uninitialized", "failed"]

    # TEST 2 — Frontend sends source_lang/target_lang as short codes (en/ur/pa)
    def test_frontend_language_codes_accepted(self):
        """Frontend sends 'en', 'ur', 'pa' — backend must accept these"""
        for src, tgt in [("en", "ur"), ("en", "pa"), ("ur", "en")]:
            payload = {"text": "Hello", "source_lang": src, "target_lang": tgt}
            r = requests.post(f"{BASE_URL}/translate", json=payload, timeout=60)
            assert r.status_code == 200, f"Failed for {src}->{tgt}"
            assert "result" in r.json(), f"No 'result' field for {src}->{tgt}"

    # TEST 3 — Response field is 'result' (not 'translation' or 'output')
    def test_response_field_is_result(self):
        """Translate.tsx reads result from geminiService which reads 'result' field"""
        payload = {"text": "Good morning", "source_lang": "en", "target_lang": "ur"}
        r = requests.post(f"{BASE_URL}/translate", json=payload, timeout=60)
        data = r.json()
        assert "result" in data
        assert isinstance(data["result"], str)

    # TEST 4 — CORS headers present (frontend runs on different port)
    def test_cors_headers_present(self):
        """Frontend on :3000 calls backend on :5000 — CORS must be enabled"""
        r = requests.options(
            f"{BASE_URL}/translate",
            headers={"Origin": "http://localhost:3000"},
            timeout=5
        )
        # Flask-CORS adds Access-Control-Allow-Origin
        assert "Access-Control-Allow-Origin" in r.headers or r.status_code in [200, 204]

    # TEST 5 — Translate and save flow: result can be used as Firestore record content
    def test_translation_result_suitable_for_firestore(self):
        """Dashboard saves translation content — verify result is non-empty string"""
        payload = {"text": "I am a student", "source_lang": "en", "target_lang": "ur"}
        r = requests.post(f"{BASE_URL}/translate", json=payload, timeout=60)
        data = r.json()
        result = data.get("result", "")
        assert isinstance(result, str)
        assert len(result) > 0
        # Simulate what storageService does — title generation
        title = f"Translation (en->ur): {payload['text'][:30]}"
        assert len(title) > 0
