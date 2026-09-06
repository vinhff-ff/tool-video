"""
validator.py - Phase 3 (step 4)

Checks a Scene JSON against the fixed animation library before it reaches
the renderer. Raises a clear error on anything the browser layer can't play.
"""

VALID_ANIMATIONS = {
    "intro", "outro", "fadeIn", "fadeOut", "slideLeft", "slideRight",
    "zoomIn", "zoomOut", "showA", "showB", "compare",
    "showConfused", "showCart",
}
VALID_IMAGES = {"A", "B", "both"}
VALID_CHARACTERS = {"pointLeftUp", "pointLeft", "pointRight", "center", "confused", "cart"}


def validate_scene_json(scene_json: dict) -> dict:
    """Returns scene_json untouched when valid, otherwise raises ValueError."""
    if "scenes" not in scene_json or not isinstance(scene_json["scenes"], list):
        raise ValueError("Scene JSON must contain a non-empty 'scenes' list")
    if len(scene_json["scenes"]) == 0:
        raise ValueError("Scene JSON 'scenes' list is empty")

    for i, scene in enumerate(scene_json["scenes"]):
        text = scene.get("text")
        if not text or not isinstance(text, str):
            raise ValueError(f"Scene {i}: missing non-empty 'text'")

        if scene.get("image") not in VALID_IMAGES:
            raise ValueError(
                f"Scene {i}: invalid image {scene.get('image')!r} "
                f"(allowed: {sorted(VALID_IMAGES)})"
            )

        if scene.get("character") not in VALID_CHARACTERS:
            raise ValueError(
                f"Scene {i}: invalid character {scene.get('character')!r} "
                f"(allowed: {sorted(VALID_CHARACTERS)})"
            )

        if scene.get("animation") not in VALID_ANIMATIONS:
            raise ValueError(
                f"Scene {i}: invalid animation {scene.get('animation')!r} "
                f"(allowed: {sorted(VALID_ANIMATIONS)})"
            )

        duration = scene.get("duration")
        if not isinstance(duration, (int, float)) or duration <= 0:
            raise ValueError(f"Scene {i}: invalid duration {duration!r}")

    return scene_json