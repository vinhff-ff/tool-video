"""
designer.py - Phase 3 (step 3)

Maps the script lines (10-15, provided by the external AI) to a Scene JSON
using a deterministic FIXED visual template. The mapping only references
existing animation/character names and never writes new animation code, so the
visuals stay consistent regardless of line count.
"""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# Deterministic pattern per line index (verified against templates/animation.js
# and validator.py). Scene 0/1 open with A/B; the last line is the closing
# "compare" scene; middle lines alternate A/B. No cart character.
def _visual_for(index: int, total: int) -> dict:
    if index == 0:
        return {"image": "A",    "character": "pointLeftUp", "animation": "showA"}
    if index == 1:
        return {"image": "B",    "character": "pointRight",  "animation": "showB"}
    if index == total - 1:
        return {"image": "both", "character": "confused",    "animation": "compare"}
    if index % 2 == 0:
        return {"image": "A",    "character": "pointLeft",   "animation": "showA"}
    return {"image": "B",    "character": "pointRight",  "animation": "showB"}


DEFAULT_DURATION = 4  # placeholder only; TTS measures & overwrites real durations

MIN_LINES = 10
MAX_LINES = 15


def design_scenes(script_lines: list) -> dict:
    """Attach the fixed visual directives to each script line → Scene JSON.

    Accepts 10-15 lines (the range the external AI is instructed to produce).
    """
    if not (MIN_LINES <= len(script_lines) <= MAX_LINES):
        raise ValueError(
            f"Script has {len(script_lines)} lines, expected between "
            f"{MIN_LINES} and {MAX_LINES}"
        )
    scenes = []
    total = len(script_lines)
    for index, line in enumerate(script_lines):
        visuals = _visual_for(index, total)
        scenes.append({**visuals, "text": line, "duration": DEFAULT_DURATION})
    return {"duration": DEFAULT_DURATION * len(scenes), "scenes": scenes}