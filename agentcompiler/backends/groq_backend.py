"""
agentcompiler/backends/groq_backend.py
=======================================
Real LLM backend using Groq's API.

Replaces asyncio.sleep() simulation with actual API calls.
Uses Groq's async client — fully non-blocking, so asyncio.gather()
genuinely overlaps multiple in-flight requests.

Install:
    pip install groq

Usage:
    export GROQ_API_KEY="your_key_here"
"""

from __future__ import annotations

import os
import asyncio
from groq import AsyncGroq

# ── Singleton client (one per process) ───────────────────────────────────────
_client: AsyncGroq | None = None

def get_client() -> AsyncGroq:
    global _client
    if _client is None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY environment variable not set.\n"
                "Run: export GROQ_API_KEY='your_key_here'"
            )
        _client = AsyncGroq(api_key=api_key)
    return _client


# ── Core call ─────────────────────────────────────────────────────────────────

async def groq_call(
    prompt: str,
    model: str = "llama-3.3-70b-versatile",
    max_tokens: int = 200,
    temperature: float = 0.0,
    retries: int = 3,
) -> str:
    """
    Make a single async Groq API call.
    Retries on rate-limit (429) with exponential backoff.
    """
    client = get_client()
    for attempt in range(retries):
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if "429" in str(e) and attempt < retries - 1:
                wait = 2 ** attempt
                print(f"  [Groq] Rate limit hit, waiting {wait}s...")
                await asyncio.sleep(wait)
            else:
                raise
    return ""
