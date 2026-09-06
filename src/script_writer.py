"""
script_writer.py - Phase 3 (step 2)

Takes the research brief from researcher.py and uses the LLM to write a
10-line Vietnamese script for a crime A-vs-B comparison video, ending with a
humorous CTA for an optional product — funny, sarcastic, trait-driven — that
becomes the video's narration. Returns a list of exactly 10 strings.
"""

import asyncio
from pathlib import Path

from llm import generate_json

BASE_DIR = Path(__file__).resolve().parent.parent
SCRIPT_PROMPT = (BASE_DIR / "prompts" / "script_writer.txt").read_text(encoding="utf-8")


def write_script(
    brief: dict,
    style: str = "hài hước, châm biếm",
    model_path: str = None,
    name_a: str = None,
    name_b: str = None,
    intro_a: str = None,
    intro_b: str = None,
    note: str = "",
    product_name: str = None,
) -> list:
    """Generate 10 TikTok-style Vietnamese lines for the A-vs-B video.

    Lines 1 and 3 start verbatim with "Đây là <name>. <intro>" when intro text is
    provided; the rest are written by the LLM.
    """
    user = (
        f"PHONG CÁCH: {style}\n"
        f"TOPIC_A = {brief['topic_a']}\n"
        f"TOPIC_B = {brief['topic_b']}"
    )
    if name_a is not None:
        user += f"\nNAME_A = {name_a}"
    if name_b is not None:
        user += f"\nNAME_B = {name_b}"
    if intro_a is not None:
        user += f"\nINTRO_A = {intro_a}"
    if intro_b is not None:
        user += f"\nINTRO_B = {intro_b}"
    if note:
        user += f"\nNOTE = {note}"
    if product_name:
        user += f"\nPRODUCT_NAME = {product_name}"

    lines = generate_json(SCRIPT_PROMPT, user, temperature=0.8, max_tokens=1536,
                          model_path=model_path)
    if not isinstance(lines, list) or len(lines) != 10:
        # Tolerant retry: split on newlines if the model ignored the JSON shape
        if isinstance(lines, str):
            lines = [ln.strip() for ln in lines.splitlines() if ln.strip()]
        if len(lines) != 10:
            raise ValueError(f"Script writer returned {len(lines)} lines, expected 10")
    script = []
    for i, ln in enumerate(lines):
        script.append(str(ln).strip().strip('"'))
    return script


async def write_script_async(
    brief: dict,
    style: str = "hài hước, châm biếm",
    model_path: str = None,
    name_a: str = None,
    name_b: str = None,
    intro_a: str = None,
    intro_b: str = None,
    note: str = "",
    product_name: str = None,
) -> list:
    return await asyncio.to_thread(
        write_script, brief, style, model_path, name_a, name_b, intro_a, intro_b, note, product_name
    )