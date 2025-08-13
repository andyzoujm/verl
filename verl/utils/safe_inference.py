import random
import asyncio
from openai import AsyncOpenAI, APIError, RateLimitError, InternalServerError, BadRequestError

async def safe_chat(
    client: AsyncOpenAI,
    model_name: str,
    messages: list[dict],
    *,
    max_tokens: int,
    temperature: float = 0.7,
    max_retries: int = 2,
    backoff_base: float = 1.0,
    backoff_max: float = 20.0,
) -> str:
    # return "I will not answer that question."
    for attempt in range(max_retries):
        try:
            if model_name.startswith("gpt-5"):
                resp = await client.chat.completions.create(
                    model=model_name,
                    messages=messages
                )
            else:
                resp = await client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
            return resp.choices[0].message.content.strip()
        except BadRequestError:
            return "I'm sorry, but I can't help with that. (Input Filter)"
        except (RateLimitError, InternalServerError, APIError, asyncio.TimeoutError) as e:
            if attempt == max_retries - 1:
                print(f"[safe_chat] Giving up after {max_retries} retries – {e!s}")
                return ""
            wait = min(backoff_base * 2 ** attempt + random.random(), backoff_max)
            print(f"[safe_chat] {e!s} – retrying in {wait:.1f}s … ({attempt + 1}/{max_retries})")
            await asyncio.sleep(wait)
        except Exception as e:
            print(f"[safe_chat] Unexpected error: {type(e).__name__}: {e}")
            return ""
    return ""
