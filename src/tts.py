"""
tts.py - Phase 2

Turns each scene's text into a Vietnamese voice clip using a pluggable engine:

- engine="edge": edge-tts (free, no API key, no local GPU). Voice is an
  edge-tts voice name such as "vi-VN-HoaiMyNeural".
- engine="vieneu": VieNeu-TTS v2/v3 Turbo (on-device, 48 kHz, preset voices +
  instant voice cloning). Runs on Kaggle's free GPU. Voice can be:
    * a preset voice name (e.g. "Adam", "Phạm Tuyên", "Minh Đức"), or
    * a path to a reference .wav clip for instant voice cloning, or
    * None to use the engine default.

Per spec: the LLM never estimates spoken duration. Each scene's audio is
generated first, its REAL duration is measured with ffprobe, and that
measured value overwrites the scene's "duration" field - audio is the
single source of truth for timing.
"""

import subprocess
from pathlib import Path

import edge_tts

BASE_DIR = Path(__file__).resolve().parent.parent
GENERATED_AUDIO_DIR = BASE_DIR / "generated" / "audio"

# --- edge-tts -----------------------------------------------------------------
# Free Vietnamese neural voices available via edge-tts:
#   vi-VN-HoaiMyNeural  (female)
#   vi-VN-NamMinhNeural (male)
EDGE_DEFAULT_VOICE = "vi-VN-HoaiMyNeural"

# --- VieNeu-TTS ----------------------------------------------------------------
# 20 built-in preset voices (North/Central/South). See
#     tts.list_preset_voices()
# for the full list. "Adam" is the v3 Turbo default (Southern, male).
VIENEU_DEFAULT_VOICE = "Adam"

MENU = {
    "edge": "Microsoft Edge neural TTS (free, internet required, no GPU)",
    "vieneu": "VieNeu-TTS v3 Turbo on-device (48 kHz, preset voices + voice cloning, Kaggle GPU)",
}


def get_audio_duration(path: Path) -> float:
    """Exact duration in seconds via ffprobe (never estimated)."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        capture_output=True, text=True, check=True,
    )
    return float(result.stdout.strip())


class TTSBackend:
    """Engine-agnostic facade. Instantiated once per run to cache model load."""

    def __init__(self, engine: str, voice=None, ref_audio=None):
        self.engine = engine
        self.voice = voice
        self.ref_audio = ref_audio
        self._vieneu = None

    # -- edge-tts ---------------------------------------------------------------
    def _prepare_edge(self):
        self.voice = self.voice or EDGE_DEFAULT_VOICE

    async def _infer_edge(self, text, out_path):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        communicate = edge_tts.Communicate(text, self.voice)
        await communicate.save(str(out_path))

    # -- vieneu ------------------------------------------------------------------
    def _prepare_vieneu(self):
        if self._vieneu is None:
            from vieneu import Vieneu
            self._vieneu = Vieneu()  # default v3 Turbo; auto-uses GPU on Kaggle
            if self.ref_audio:
                registered = str(Path(self.ref_audio).resolve())
                # enrollment (denoise + speaker profile) once, reuse by name
                self.voice = f"__ref_{Path(registered).stem}"
                self._vieneu.add_voice(self.voice, registered)
            elif not self.voice:
                self.voice = VIENEU_DEFAULT_VOICE

    def _infer_vieneu(self, text, out_path):
        audio = self._vieneu.infer(text, voice=self.voice)
        self._vieneu.save(audio, str(out_path))

    # -- dispatch ---------------------------------------------------------------
    def prepare(self):
        if self.engine == "edge":
            self._prepare_edge()
        elif self.engine == "vieneu":
            self._prepare_vieneu()
        else:
            raise ValueError(f"Unknown TTS engine: {self.engine!r}")

    async def synthesize(self, text: str, out_path: Path):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if self.engine == "edge":
            await self._infer_edge(text, out_path)
        elif self.engine == "vieneu":
            self._infer_vieneu(text, out_path)
        else:
            raise ValueError(f"Unknown TTS engine: {self.engine!r}")
        return get_audio_duration(out_path)


async def synthesize_all_scenes(scene_json: dict, run_id: str,
                                engine: str = "edge",
                                voice: str = None,
                                ref_audio: str = None) -> dict:
    """
    Generates one audio clip per scene (in generated/audio/<run_id>/scene_NN.wav),
    overwrites each scene's 'duration' with the real measured value, and returns
    the updated scene_json plus the ordered list of audio paths.

    engine:     "edge" (default) or "vieneu".
    voice:      voice name for the chosen engine (see module docstring).
    ref_audio:  (vieneu only) path to a reference .wav clip for voice cloning.
    """
    backend = TTSBackend(engine=engine, voice=voice, ref_audio=ref_audio)
    backend.prepare()

    out_ext = "wav" if engine == "vieneu" else "mp3"

    run_audio_dir = GENERATED_AUDIO_DIR / run_id
    audio_paths = []

    for i, scene in enumerate(scene_json["scenes"]):
        out_path = run_audio_dir / f"scene_{i:02d}.{out_ext}"
        duration = await backend.synthesize(scene["text"], out_path)
        scene["duration"] = round(duration, 3)
        audio_paths.append(out_path)

    return {"scene_json": scene_json, "audio_paths": audio_paths}


def concat_audio(audio_paths: list, run_id: str) -> Path:
    """Concatenate per-scene clips (in scene order) into one narration track.

    Always outputs WAV to avoid codec mismatch between engines.
    """
    run_audio_dir = GENERATED_AUDIO_DIR / run_id
    concat_list_path = run_audio_dir / "concat_list.txt"
    concat_list_path.write_text(
        "\n".join(f"file '{p.resolve()}'" for p in audio_paths),
        encoding="utf-8",
    )
    final_audio_path = run_audio_dir / "narration.wav"
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0",
            "-i", str(concat_list_path),
            "-ar", "44100",   # standard sample rate for cross-engine consistency
            "-ac", "1",
            str(final_audio_path),
        ],
        check=True, capture_output=True,
    )
    return final_audio_path