/**
 * UNIT TESTS — Translation Logic & Language Utilities
 * Tool: Jest
 * Run: npx jest tests/unit/translation.test.ts
 *
 * AI USAGE LOG:
 * Prompt: "Generate Jest unit tests for translation logic based on types.ts and Translate.tsx"
 * Changes made: Added specific language codes matching our FLORES-200 trained model,
 *               adjusted swap logic test to match actual component state behavior.
 */

import { LANGUAGES, LanguageOption } from '../../types';

// ─── Helper functions extracted from Translate.tsx for unit testing ───────────

function validateTranslationInput(text: string, sourceLang: string, targetLang: string): string | null {
  if (!text.trim()) return 'Enter some text to translate.';
  if (sourceLang === targetLang) return 'Source and target languages must be different.';
  return null;
}

function swapLanguages(
  sourceLang: string, targetLang: string, text: string, translated: string
): { sourceLang: string; targetLang: string; text: string; translated: string } {
  return {
    sourceLang: targetLang,
    targetLang: sourceLang,
    text: translated,
    translated: text,
  };
}

function buildTranslationTitle(text: string, sourceLang: string, targetLang: string): string {
  const titleSnippet = text.slice(0, 30) + (text.length > 30 ? '...' : '');
  return `Translation (${sourceLang}->${targetLang}): ${titleSnippet}`;
}

function getLanguageOption(code: string): LanguageOption | undefined {
  return LANGUAGES.find(l => l.code === code);
}

function isRTLLanguage(code: string): boolean {
  const lang = LANGUAGES.find(l => l.code === code);
  return lang?.isRTL ?? false;
}

const FLORES_MAP: Record<string, string> = {
  en: 'eng_Latn',
  ur: 'urd_Arab',
  pa: 'pan_Arab',
};

function mapToFloresCode(code: string): string {
  return FLORES_MAP[code] ?? code;
}

// ─── TESTS ────────────────────────────────────────────────────────────────────

describe('validateTranslationInput', () => {
  // TEST 1
  test('returns error when text is empty', () => {
    expect(validateTranslationInput('', 'en', 'ur')).toBe('Enter some text to translate.');
  });

  // TEST 2
  test('returns error when text is only whitespace', () => {
    expect(validateTranslationInput('   ', 'en', 'ur')).toBe('Enter some text to translate.');
  });

  // TEST 3
  test('returns error when source and target language are the same', () => {
    expect(validateTranslationInput('Hello', 'en', 'en')).toBe('Source and target languages must be different.');
  });

  // TEST 4
  test('returns null for valid English to Urdu input', () => {
    expect(validateTranslationInput('Hello world', 'en', 'ur')).toBeNull();
  });

  // TEST 5
  test('returns null for valid Urdu to English input', () => {
    expect(validateTranslationInput('آپ کیسے ہیں', 'ur', 'en')).toBeNull();
  });
});

describe('swapLanguages', () => {
  // TEST 6
  test('swaps source and target language codes', () => {
    const result = swapLanguages('en', 'ur', 'Hello', 'ہیلو');
    expect(result.sourceLang).toBe('ur');
    expect(result.targetLang).toBe('en');
  });

  // TEST 7
  test('swaps text and translated content', () => {
    const result = swapLanguages('en', 'ur', 'Hello', 'ہیلو');
    expect(result.text).toBe('ہیلو');
    expect(result.translated).toBe('Hello');
  });

  // TEST 8
  test('handles empty translated text on swap', () => {
    const result = swapLanguages('en', 'pa', 'Hello', '');
    expect(result.text).toBe('');
    expect(result.translated).toBe('Hello');
  });
});

describe('buildTranslationTitle', () => {
  // TEST 9
  test('builds correct title for short text', () => {
    const title = buildTranslationTitle('Hello', 'en', 'ur');
    expect(title).toBe('Translation (en->ur): Hello');
  });

  // TEST 10
  test('truncates long text to 30 characters with ellipsis', () => {
    const longText = 'This is a very long sentence that should be truncated properly';
    const title = buildTranslationTitle(longText, 'en', 'ur');
    expect(title).toContain('...');
    expect(title.length).toBeLessThan(longText.length + 30);
  });
});

describe('LANGUAGES config (types.ts)', () => {
  // TEST 11
  test('contains exactly 3 languages', () => {
    expect(LANGUAGES).toHaveLength(3);
  });

  // TEST 12
  test('English is not RTL', () => {
    expect(isRTLLanguage('en')).toBe(false);
  });

  // TEST 13
  test('Urdu is RTL', () => {
    expect(isRTLLanguage('ur')).toBe(true);
  });

  // TEST 14
  test('Punjabi (Shahmukhi) is RTL', () => {
    expect(isRTLLanguage('pa')).toBe(true);
  });
});

describe('mapToFloresCode', () => {
  // TEST 15
  test('maps en to eng_Latn', () => {
    expect(mapToFloresCode('en')).toBe('eng_Latn');
  });

  test('maps ur to urd_Arab', () => {
    expect(mapToFloresCode('ur')).toBe('urd_Arab');
  });

  test('maps pa to pan_Arab', () => {
    expect(mapToFloresCode('pa')).toBe('pan_Arab');
  });

  test('returns unknown code unchanged', () => {
    expect(mapToFloresCode('xyz')).toBe('xyz');
  });
});
