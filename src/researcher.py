"""
researcher.py - Phase 3 (step 1)

Gathers real web info about product A and product B via DuckDuckGo (DDGS),
then asks the LLM to compress the raw snippets into a short brief of key
points per product. Only that compact brief is passed to the script writer.

Run on Kaggle with "Internet: ON".
"""

import asyncio
from pathlib import Path

from ddgs import DDGS

from llm import generate_json

BASE_DIR = Path(__file__).resolve().parent.parent
RESEARCHER_PROMPT = (BASE_DIR / "prompts" / "researcher.txt").read_text(encoding="utf-8")


def search_topic(query: str, max_results: int = 5, region: str = "vi-vn") -> str:
    """Sync DuckDuckGo search → newline-joined snippets (title + body)."""
    try:
        with DDGS() as ddgs:
            results = list(
                ddgs.text(query, region=region, safesearch="off", max_results=max_results)
            )
    except Exception as e:  # DDGS is flaky; never block the pipeline
        print(f"[researcher] search failed for {query!r}: {e}")
        return ""
    lines = [
        f"- {r.get('title', '')}: {r.get('body', '')}" for r in results if r.get("body")
    ]
    return "\n".join(lines) if lines else f"(không có kết quả cho {query})"


def research_products(topic_a: str, topic_b: str, max_results: int = 5) -> dict:
    """
    Returns {"topic_a": [...], "topic_b": [...]} — short key points per product,
    computed by the LLM from real search snippets.
    """
    print(f"[researcher] Searching web for: {topic_a} | {topic_b} …")
    search_a = search_topic(topic_a, max_results)
    search_b = search_topic(topic_b, max_results)

    if not search_a and not search_b:
        raise RuntimeError("Web search returned nothing — check Kaggle Internet is ON")

    system = RESEARCHER_PROMPT.format(search_a=search_a, search_b=search_b)
    brief = generate_json(system, "Tóm tắt 2 sản phẩm trên thành các điểm nổi bật.",
                          temperature=0.3, max_tokens=1024)
    return {"topic_a": brief["topic_a"], "topic_b": brief["topic_b"]}


async def research_products_async(topic_a: str, topic_b: str, max_results: int = 5) -> dict:
    """Async wrapper so the blocking search/LLM runs off the event loop."""
    return await asyncio.to_thread(research_products, topic_a, topic_b, max_results)