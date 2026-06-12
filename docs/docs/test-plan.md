# Test Plan — NoteGen FYP
**Course:** Software Quality Assurance (SQA) | BSCS Semester 8  
**Application Under Test:** NoteGen — Multilingual Summarization & Translation Platform  
**Author:** [Your Name]  
**Date:** June 2026

---

## 1. Objectives
- Verify all 6 translation directions (EN↔UR, EN↔PA, UR↔PA) produce correct output
- Validate Firebase Authentication (login, signup, protected routes)
- Confirm Gemini summarization API returns non-empty results
- Test Dashboard CRUD operations (save, delete, search records)
- Detect edge cases: empty input, same-language translation, very long text, special characters

---

## 2. Scope

### In Scope
- `/translate` page — local Flask model + Gemini engine toggle
- `/summarize` page — Gemini API summarization
- `/login` page — Firebase email/password auth
- `/dashboard` page — history records (save, delete, search, download)
- Flask backend API (`/translate` endpoint)
- Language swap functionality
- PDF download

### Out of Scope
- Audio processing page (future FYP feature)
- Firebase Storage internals
- Gemini API internals / billing
- Mobile responsiveness

---

## 3. Test Types & Tools

| Test Type | Tool | Count |
|---|---|---|
| Unit | Jest + React Testing Library | 15 |
| Integration | Pytest + requests | 5 |
| API | Pytest + requests | 5 |
| E2E | Playwright | 5 |
| Negative / Edge-case | Pytest + requests | 10 |

---

## 4. Risk Areas
- RAM constraints may cause Flask model to crash mid-load (HIGH)
- `pan_Arab` Punjabi token may not map correctly in quantized model (HIGH)
- Firebase quota limits on Spark plan (MEDIUM)
- Gemini API rate limits (MEDIUM)
- RTL text rendering in browser (LOW)

---

## 5. Entry Criteria
- Flask backend running on `http://127.0.0.1:5000`
- Frontend running on `http://localhost:3000`
- Firebase project configured with valid API keys
- `.env` file containing `GEMINI_API_KEY`

## 6. Exit Criteria
- All 40 tests executed (pass or documented fail)
- Coverage report generated
- Defect log completed for all failures
- QA report written

---

## 7. Environment
- OS: Windows 11
- Python: 3.x (venv)
- Node.js: latest LTS
- Browser: Chromium (Playwright default)
- Model: M2M100 fine-tuned checkpoint (quantized, float16)
