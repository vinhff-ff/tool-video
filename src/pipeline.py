"""
pipeline.py

Phase 1 (generate_video_phase1): Scene JSON (hand-typed durations, no AI)
-> render -> record (silent) -> ffmpeg -> final.mp4 (no audio).

Phase 2 (generate_video_phase2): TTS synthesizes each scene's audio first,
REAL measured durations overwrite the scene JSON's durations, then
render -> record (silent) -> mux with the narration track -> final.mp4
(with audio). Audio is always the source of truth for timing, never a guess.

Phase 3 (generate_video_phase3): FULLY AUTOMATIC — feed two product names
(topic_a, topic_b), and the pipeline itself:
  1. researcher.py  – DDGS web search + LLM brief of key points per product
  2. script_writer.py – LLM writes a 10-line Vietnamese script (funny/sarcastic)
  3. designer.py    – maps the script to Scene JSON (fixed visual template)
  4. validator.py   – checks animations/characters/images against the library
  then runs the same TTS→render→record→mux tail as phase 2.
The Lean model runs LOCALLY on Kaggle's GPU via llama.cpp (Qwen2.5-7B GGUF,
auto-downloaded on first use).

TTS engine is pluggable:
    engine="edge"   – Microsoft Edge neural TTS (free, no GPU, no API key)
                       default voice: vi-VN-HoaiMyNeural
    engine="vieneu"  – VieNeu-TTS v3 Turbo on Kaggle GPU (48 kHz,
                       20 preset voices + instant voice cloning)
                       default voice: "Adam" (Southern male)

Run from the project root in a Kaggle notebook cell (top-level await is
supported directly in Jupyter/Kaggle cells):

    import sys
    sys.path.insert(0, "src")
    from pipeline import generate_video_phase2, generate_video_phase3

    assets = {
        "background": "assets/background.jpg",
        "character": "assets/character.png",
        "character_confused": "assets/character_confused.png",  # optional
        "character_cart": "assets/character_cart.png",          # optional
        "image_a": "assets/A.jpg",
        "image_b": "assets/B.jpg",
    }

    # Phase 2 (from a scene JSON file, e.g. hand-edited)
    result = await generate_video_phase2(
        scene_json_path="generated/scripts/example_scene.json",
        assets=assets, run_id="test_run_001",
    )

    # Phase 3 (LLM auto-generates everything)
    result = await generate_video_phase3(
        topic_a="Chanel nước hoa", topic_b="Dior nước hoa",
        assets=assets, run_id="test_run_003",
    )
"""

import json
import subprocess
from pathlib import Path

from renderer import render_html
from recorder import record_html

BASE_DIR = Path(__file__).resolve().parent.parent


def load_scene_json(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def convert_to_mp4(webm_path: Path, run_id: str) -> Path:
    mp4_path = webm_path.with_suffix(".mp4")
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", str(webm_path),
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "23",
            "-pix_fmt", "yuv420p",
            str(mp4_path),
        ],
        check=True,
        capture_output=True,
    )
    return mp4_path


def mux_audio_video(silent_video_path: Path, narration_path: Path, run_id: str) -> Path:
    """Combine the silent recorded video with the concatenated TTS narration track."""
    final_path = silent_video_path.parent / f"{run_id}_final.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", str(silent_video_path),
            "-i", str(narration_path),
            "-c:v", "copy",
            "-c:a", "aac",
            "-shortest",
            str(final_path),
        ],
        check=True,
        capture_output=True,
    )
    return final_path


async def generate_video_phase1(scene_json_path: str, assets: dict, run_id: str) -> Path:
    """No TTS - scene durations come straight from the hand-typed JSON."""
    scene_json = load_scene_json(scene_json_path)

    html_path = render_html(scene_json, assets, run_id)
    print(f"[pipeline] HTML rendered: {html_path}")

    webm_path = await record_html(html_path, run_id)
    print(f"[pipeline] Video recorded: {webm_path}")

    mp4_path = convert_to_mp4(webm_path, run_id)
    print(f"[pipeline] Final MP4 (no audio): {mp4_path}")

    return mp4_path


async def _tts_and_mux(scene_json: dict, assets: dict, run_id: str,
                       engine: str, voice: str, ref_audio: str) -> Path:
    """Shared tail of phases 2/3: TTS → real durations → render → record → mux."""
    from tts import synthesize_all_scenes, concat_audio

    tts_result = await synthesize_all_scenes(
        scene_json, run_id,
        engine=engine,
        voice=voice,
        ref_audio=ref_audio,
    )
    scene_json = tts_result["scene_json"]  # durations are now real, measured values
    print(f"[pipeline] TTS ({engine}) generated for {len(tts_result['audio_paths'])} scenes")

    narration_path = concat_audio(tts_result["audio_paths"], run_id)
    print(f"[pipeline] Narration track: {narration_path}")

    html_path = render_html(scene_json, assets, run_id)
    print(f"[pipeline] HTML rendered: {html_path}")

    webm_path = await record_html(html_path, run_id)
    print(f"[pipeline] Video (silent) recorded: {webm_path}")

    silent_mp4_path = convert_to_mp4(webm_path, run_id)

    final_path = mux_audio_video(silent_mp4_path, narration_path, run_id)
    print(f"[pipeline] Final MP4 (with audio): {final_path}")

    return final_path


async def generate_video_phase2(
    scene_json_path: str,
    assets: dict,
    run_id: str,
    engine: str = "edge",
    voice: str = None,
    ref_audio: str = None,
) -> Path:
    """From a scene JSON file: TTS -> real durations -> render -> record -> mux audio.

    engine:    "edge" (default) or "vieneu".
    voice:     voice name for the chosen engine (see tts.py docstring).
    ref_audio: (vieneu only) path to a reference .wav clip for voice cloning.
    """
    scene_json = load_scene_json(scene_json_path)
    return await _tts_and_mux(scene_json, assets, run_id, engine, voice, ref_audio)


async def generate_video_phase3(
    topic_a: str,
    topic_b: str,
    assets: dict,
    run_id: str,
    engine: str = "edge",
    voice: str = None,
    ref_audio: str = None,
    style: str = "hài hước, châm biếm",
    model_path: str = None,
    name_a: str = None,
    name_b: str = None,
    intro_a: str = None,
    intro_b: str = None,
    note: str = "",
    research: bool = True,
    product_name: str = None,
) -> Path:
    """Fully automatic: web research → LLM script → designer → validator → video.

    topic_a/topic_b: two products to compare (any short description, e.g. names).
    name_a/name_b:   display names used as "Đây là <name>..." in lines 1–2.
    intro_a/intro_b: verbatim intro text appended to lines 1–2 (no need for
                     research when provided — the user is authoring the opener).
    note:            optional free-text instruction the LLM must follow for the
                     rest of the script (tone, points to hit, jokes, etc).
    research:        set False to skip the DDGS web search (used by the Telegram
                     bot which supplies intro text instead).
    style:           tone for the LLM script (default: funny/sarcastic).
    model_path:      path to a GGUF model; None → auto-download Qwen2.5-7B-Instruct.
    engine/voice/ref_audio: same TTS options as phase 2.
    """
    from researcher import research_products_async
    from script_writer import write_script_async
    from designer import design_scenes
    from validator import validate_scene_json

    if research and intro_a is None and intro_b is None:
        brief = await research_products_async(topic_a, topic_b)
        print(f"[pipeline] Research done: A={len(brief['topic_a'])} pts, B={len(brief['topic_b'])} pts")
    else:
        brief = {"topic_a": str(topic_a), "topic_b": str(topic_b)}
        print("[pipeline] Skipped research (user-supplied intro text).")

    script = await write_script_async(
        brief,
        style=style,
        model_path=model_path,
        name_a=name_a or topic_a,
        name_b=name_b or topic_b,
        intro_a=intro_a,
        intro_b=intro_b,
        note=note,
        product_name=product_name,
    )
    print(f"[pipeline] Script generated ({len(script)} lines).")

    scene_json = design_scenes(script)
    validate_scene_json(scene_json)
    print(f"[pipeline] Scene JSON validated ({len(scene_json['scenes'])} scenes).")

    # Save the auto-generated scene JSON for inspection/reuse
    scenes_dir = BASE_DIR / "generated" / "scripts"
    scenes_dir.mkdir(parents=True, exist_ok=True)
    out_path = scenes_dir / f"{run_id}.json"
    out_path.write_text(
        json.dumps(scene_json, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[pipeline] Scene JSON saved: {out_path}")

    return await _tts_and_mux(scene_json, assets, run_id, engine, voice, ref_audio)


if __name__ == "__main__":
    import asyncio

    assets = {
        "background": str(BASE_DIR / "assets" / "background.jpg"),
        "character": str(BASE_DIR / "assets" / "character.png"),
        "character_confused": str(BASE_DIR / "assets" / "character_confused.png"),
        "character_cart": str(BASE_DIR / "assets" / "character_cart.png"),
        "image_a": str(BASE_DIR / "assets" / "A.jpg"),
        "image_b": str(BASE_DIR / "assets" / "B.jpg"),
    }

    # --- edge-tts (default) ---------------------------------------------------
    result = asyncio.run(
        generate_video_phase2(
            scene_json_path=str(BASE_DIR / "generated" / "scripts" / "example_scene.json"),
            assets=assets,
            run_id="test_edge",
            engine="edge",
        )
    )
    print("DONE (edge):", result)

    # --- VieNeu-TTS v3 Turbo – preset voice -----------------------------------
    result = asyncio.run(
        generate_video_phase2(
            scene_json_path=str(BASE_DIR / "generated" / "scripts" / "example_scene.json"),
            assets=assets,
            run_id="test_vieneu_preset",
            engine="vieneu",
            voice="Phạm Tuyên",
        )
    )
    print("DONE (vieneu preset):", result)

    # --- VieNeu-TTS v3 Turbo – voice cloning ----------------------------------
    ref_clip = str(BASE_DIR / "assets" / "my_voice_sample.wav")
    result = asyncio.run(
        generate_video_phase2(
            scene_json_path=str(BASE_DIR / "generated" / "scripts" / "example_scene.json"),
            assets=assets,
            run_id="test_vieneu_clone",
            engine="vieneu",
            ref_audio=ref_clip,
        )
    )
    print("DONE (vieneu clone):", result)

    # --- Phase 3: LLM tự sinh toàn bộ -------------------------------------------------
    result = asyncio.run(
        generate_video_phase3(
            topic_a="Chanel nước hoa",
            topic_b="Dior nước hoa",
            assets=assets,
            run_id="test_phase3",
        )
    )
    print("DONE (phase3 LLM):", result)