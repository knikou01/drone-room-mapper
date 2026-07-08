"""Integration smoke test: confirms .env loading and the Gemini API key
work end-to-end. Makes one real call to Google's API -- not meant to run
automatically in CI, and skips cleanly if GEMINI_API_KEY isn't configured
so it doesn't break the suite for anyone without a .env set up.

Run explicitly with:
    pytest tests/test_gemini_key.py -v -s
"""

import os

import pytest
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

pytestmark = pytest.mark.skipif(
    not GEMINI_API_KEY,
    reason="GEMINI_API_KEY not set -- copy .env.example to .env and fill it in to run this test",
)


def test_gemini_api_key_is_valid():
    from google import genai

    client = genai.Client()
    response = client.models.generate_content(
        model="gemini-3.5-flash",
        contents="Reply with exactly one word: OK",
    )
    print(f"Gemini response: {response.text!r}")

    assert response.text is not None
    assert len(response.text.strip()) > 0
