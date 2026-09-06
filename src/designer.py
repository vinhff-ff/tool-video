"""
designer.py - Phase 3 (step 3)

Maps the 6 script lines to a Scene JSON using the FIXED visual template.
Per spec, the designer AI only references existing animation/character names
and never writes new animation code — so the mapping is deterministic and the
creativity lives in the script itself (script_writer.py).
"""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DESIGNER_PROMPT = (BASE_DIR / "prompts" / "designer.txt").read_text(encoding="utf-8")

# 6-scene fixed template matching templates/animation.js
VISUAL_TEMPLATE = [
    {"image": "A",    "character": "pointLeftUp", "animation": "showA"},
    {"image": "B",    "character": "pointRight",  "animation": "showB"},
    {"image": "both", "character": "confused",    "animation": "compare"},
    {"image": "A",    "character": "pointLeft",   "animation": "showA"},
    {"image": "B",    "character": "pointRight",  "animation": "showB"},
    {"image": "both", "character": "cart",        "animation": "compare"},
]

DEFAULT_DURATION = 4  # placeholder only; TTS measures & overwrites real durations


def design_scenes(script_lines: list) -> dict:
    """Attach the fixed visual directives to each script line → Scene JSON."""
    if len(script_lines) != len(VISUAL_TEMPLATE):
        raise ValueError(
            f"Script has {len(script_lines)} lines, expected {len(VISUAL_TEMPLATE)}"
        )
    scenes = []
    for line, visual in zip(script_lines, VISUAL_TEMPLATE):
        scenes.append({**visual, "text": line, "duration": DEFAULT_DURATION})
    return {"duration": DEFAULT_DURATION * len(scenes), "scenes": scenes}