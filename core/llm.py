import time
from google import genai
from google.genai import types
from core.config import GEMINI_API_KEY, GEMINI_MODEL

_client = None


def _get_client():
    global _client
    if _client is None:
        if not GEMINI_API_KEY:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. Add it to your .env file."
            )
        _client = genai.Client(api_key=GEMINI_API_KEY)
    return _client


def call_llm(
    prompt: str,
    max_tokens: int = 2048,
    model: str | None = None,
    reasoning_effort: str = "low",
    max_retries: int = 3,
) -> str:
    client = _get_client()
    use_model = model or GEMINI_MODEL

    for attempt in range(max_retries + 1):
        try:
            response = client.models.generate_content(
                model=use_model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    max_output_tokens=max_tokens,
                    temperature=0.7,
                ),
            )
            content = response.text
            if not content:
                raise RuntimeError(
                    "Gemini returned empty content. "
                    "Try increasing max_tokens."
                )
            return content
        except Exception as e:
            if attempt == max_retries:
                raise RuntimeError(
                    f"Gemini failed after {max_retries + 1} attempts. Error: {e}"
                )
            time.sleep(5)