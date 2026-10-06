import re
import time

from openai import OpenAI, RateLimitError

from core.config import GROQ_API_KEY, GROQ_MODEL, GROQ_BASE_URL

_client = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        if not GROQ_API_KEY:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Add it to your .env file."
            )
        _client = OpenAI(base_url=GROQ_BASE_URL, api_key=GROQ_API_KEY)
    return _client


def _seconds_to_wait(error: RateLimitError) -> float:
    try:
        header_val = error.response.headers.get("retry-after")
        if header_val:
            return float(header_val)
    except Exception:
        pass
    match = re.search(r"try again in ([\d.]+)s", str(error))
    if match:
        return float(match.group(1))
    return 20.0


def call_llm(
    prompt: str,
    max_tokens: int = 2048,
    model: str | None = None,
    reasoning_effort: str = "low",
    max_retries: int = 3,
) -> str:
    client = _get_client()
    response = None

    for attempt in range(max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model or GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                reasoning_effort=reasoning_effort,
            )
            break
        except RateLimitError as e:
            if attempt == max_retries:
                raise RuntimeError(
                    f"Groq rate limit hit {max_retries + 1} times; giving up. Error: {e}"
                )
            time.sleep(_seconds_to_wait(e))

    choice = response.choices[0]
    content = choice.message.content
    finish_reason = getattr(choice, "finish_reason", "unknown")

    if not content:
        raise RuntimeError(
            f"Groq returned empty content (finish_reason={finish_reason}). "
            f"Try increasing max_tokens or lowering reasoning_effort."
        )

    if finish_reason == "length":
        raise RuntimeError(
            f"Groq response truncated (finish_reason=length) after "
            f"{max_tokens} max_tokens. Raise max_tokens for this call."
        )

    return content