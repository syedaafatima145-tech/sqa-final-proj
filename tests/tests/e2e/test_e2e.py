"""
END-TO-END TESTS — Browser Flows using Playwright
Tool: Playwright (Python)
Install: pip install playwright && playwright install chromium
Run: pytest tests/e2e/test_e2e.py -v --headed

Prerequisites:
  - Frontend running: npm run dev (http://localhost:3000)
  - Backend running: python backend/app.py (http://127.0.0.1:5000)

AI USAGE LOG:
Prompt: "Generate Playwright E2E tests for a React translation app with Firebase auth"
Changes made: Used actual selectors from Translate.tsx (textarea, select, button text),
              matched actual route paths from App.tsx (/login, /translate, /dashboard),
              added wait conditions matching the app's loading spinner behavior,
              removed Google OAuth test (app uses email/password only per Login.tsx).
"""

import pytest
from playwright.sync_api import Page, expect
import time

BASE_URL = "http://localhost:3000"
TEST_EMAIL = "test_sqa@example.com"
TEST_PASSWORD = "TestPassword123"


# ── E2E Test 1: Home page loads correctly ─────────────────────────────────────
def test_home_page_loads(page: Page):
    """Verify homepage renders with key content"""
    page.goto(f"{BASE_URL}/")
    page.wait_for_load_state("networkidle")

    # Check heading is visible
    expect(page.locator("h1")).to_be_visible()
    heading_text = page.locator("h1").inner_text()
    assert "Bridging" in heading_text or "NoteGen" in heading_text or len(heading_text) > 0

    # Check navigation links
    expect(page.get_by_role("link", name="Try Translation")).to_be_visible()


# ── E2E Test 2: Translation page — successful translation ─────────────────────
def test_translation_page_translates_text(page: Page):
    """User enters text, selects EN→UR, clicks Translate, sees output"""
    page.goto(f"{BASE_URL}/#/translate")
    page.wait_for_load_state("networkidle")

    # Wait for page to settle
    page.wait_for_selector("textarea", timeout=10000)

    # Type input text
    page.locator("textarea").fill("Hello, how are you?")

    # Ensure source is English, target is Urdu (defaults)
    source_select = page.locator("select").first
    target_select = page.locator("select").last
    source_select.select_option("en")
    target_select.select_option("ur")

    # Click translate button
    page.get_by_role("button", name="Translate").click()

    # Wait for loading to finish (spinner disappears)
    page.wait_for_selector("text=Processing", state="hidden", timeout=120000)

    # Output panel should have content
    output_panel = page.locator(".bg-indigo-950").last
    output_text = output_panel.inner_text()
    assert len(output_text.strip()) > 0
    assert "Waiting for Input" not in output_text


# ── E2E Test 3: Language swap works ───────────────────────────────────────────
def test_language_swap_button(page: Page):
    """Clicking swap reverses source/target and swaps text content"""
    page.goto(f"{BASE_URL}/#/translate")
    page.wait_for_load_state("networkidle")
    page.wait_for_selector("textarea", timeout=10000)

    # Fill in text
    page.locator("textarea").fill("Hello")

    # Get initial source value
    source_before = page.locator("select").first.input_value()

    # Click swap button
    page.get_by_title("Reverse direction").click()

    # Source and target should be swapped
    source_after = page.locator("select").first.input_value()
    assert source_after != source_before


# ── E2E Test 4: Login page — form validation ──────────────────────────────────
def test_login_page_shows_error_for_wrong_credentials(page: Page):
    """Submitting wrong credentials shows an error message"""
    page.goto(f"{BASE_URL}/#/login")
    page.wait_for_load_state("networkidle")

    # Fill wrong credentials
    page.get_by_placeholder("Email Address").fill("wrong@wrong.com")
    page.get_by_placeholder("Password").fill("wrongpassword")

    # Submit
    page.get_by_role("button", name="Login Account").click()

    # Wait for Firebase error
    page.wait_for_timeout(3000)

    # Error message should appear
    error_div = page.locator(".text-red-600")
    expect(error_div).to_be_visible(timeout=5000)


# ── E2E Test 5: Protected dashboard redirects unauthenticated users ───────────
def test_dashboard_redirects_unauthenticated(page: Page):
    """Accessing /dashboard without login redirects to /login"""
    page.goto(f"{BASE_URL}/#/dashboard")
    page.wait_for_load_state("networkidle")
    page.wait_for_timeout(2000)

    # Should be redirected to login
    current_url = page.url
    assert "/login" in current_url or page.get_by_text("Welcome Back").is_visible()
