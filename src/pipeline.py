"""
pipeline.py

Phase 1 (generate_video_phase1): Scene JSON (hand-typed durations, no AI)
-> render -> record (silent) -> ffmpeg -> final.mp4 (no audio).

Phase 2 (generate_video_phase2): TTS synthesizes each scene's audio first,
REAL measured durations overwrite the scene JSON's durations, then
render -> record (silent) -> mux with the narration track -> final.mp4
(with audio). Audio is always the source of truth for timing, never a guess.

Phase 3 (generate_video_phase3): takes the dialogue lines (10-15) provided by
an EXTERNAL AI (the Telegram bot receives them as a JSON array), and:
  1. designer.py  – maps the lines to Scene JSON (fixed visual template, no cart)
  2. validator.py – checks animations/characters/images against the library
  then runs the same TTS→render→record→mux tail as phase 2.
No LLM/GGUF model is needed anywhere in the pipeline.

TTS engine is pluggable:
    engine="edge"   – Microsoft Edge neural TTS (free, no GPU, no API key)
                       default voice: vi-VN-HoaiMyNeural
    engine="vieneu"  – VieNeu-TTS v3 Turbo on Kaggle GPU (48 kHz,
                       custom voice via ref_audio cloning)
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
        "image_a": "assets/A.jpg",
        "image_b": "assets/B.jpg",
    }

    # Phase 2 (from a scene JSON file, e.g. hand-edited)
    result = await generate_video_phase2(
        scene_json_path="generated/scripts/example_scene.json",
        assets=assets, run_id="test_run_001",
    )

    # Phase 3 (external AI supplies the dialogue lines)
    result = await generate_video_phase3(
        script_lines=["Đây là A.", "Đây là B.", "..."],
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
    script_lines: list,
    assets: dict,
    run_id: str,
    engine: str = "vieneu",
    voice: str = None,
    ref_audio: str = None,
) -> Path:
    """From external-AI dialogue lines (10-15) → designer → validator → video.

    script_lines: list of Vietnamese narration strings (10-15), produced by the
                  external AI and passed through by the Telegram bot.
    engine/voice/ref_audio: same TTS options as phase 2 (vieneu + custom
                  ref_audio clone for the fixed reused voice).
    """
    from designer import design_scenes
    from validator import validate_scene_json

    if not (10 <= len(script_lines) <= 15):
        raise ValueError(
            f"script_lines must be 10-15 lines, got {len(script_lines)}"
        )
    if not all(isinstance(ln, str) and ln.strip() for ln in script_lines):
        raise ValueError("Every line in script_lines must be a non-empty string")

    scene_json = design_scenes(script_lines)
    validate_scene_json(scene_json)
    print(f"[pipeline] Scene JSON validated ({len(scene_json['scenes'])} scenes).")

    # Save the scene JSON for inspection/reuse
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
        "image_a": str(BASE_DIR / "assets" / "A.jpg"),
        "image_b": str(BASE_DIR / "assets" / "B.jpg"),
    }

    result = asyncio.run(
        generate_video_phase2(
            scene_json_path=str(BASE_DIR / "generated" / "scripts" / "example_scene.json"),
            assets=assets,
            run_id="test_phase2",
            engine="edge",
        )
    )
    print("DONE (phase2 edge):", result)

    result = asyncio.run(
        generate_video_phase3(
            script_lines=[
                "Đây là kẻ lừa đảo A.",
                "Đây là tay buôn lậu B.",
                "Vậy hai kẻ này có tội khác nhau thế nào.",
                "A dùng chiêu bài cũ, tinh vi và lặng lẽ.",
                "B thì phô trương, liều mạng như một màn sân khấu.",
                "Án dành cho A là 5 năm sau song sắt.",
                "Án dành cho B là 12 năm sau chấn song.",
                "Cả hai đều mượn hào quang để che đi bóng tối.",
                "Lá bài nào rồi cũng phải lật ngửa.",
                "Hãy sống đúng pháp luật, đừng học theo họ.",
            ],
            assets=assets,
            run_id="test_phase3",
            engine="edge",
        )
    )
    print("DONE (phase3):", result)