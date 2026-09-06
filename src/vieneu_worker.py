"""
vieneu_worker.py — chạy TRONG venv riêng (tts_env), có torch bản cũ tương thích GPU Kaggle.

Không import trực tiếp trong kernel chính (torch ở đó bị Kaggle "đóng băng" ở bản mới,
không tương thích T4/P100). Script này nhận 1 file JSON mô tả job, load model 1 lần,
sinh toàn bộ audio, rồi thoát.

Input JSON (--job):
{
    "voice": "Adam" | null,
    "ref_audio": "/path/to/ref.wav" | null,
    "items": [{"text": "...", "out_path": "/abs/path/scene_00.wav"}, ...]
}

Output: in ra "WORKER_OK" nếu thành công, hoặc "WORKER_ERROR: <msg>" ra stderr rồi exit(1).
"""
import argparse
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", required=True, help="Đường dẫn file JSON mô tả job")
    args = parser.parse_args()

    job = json.loads(Path(args.job).read_text(encoding="utf-8"))
    voice = job.get("voice")
    ref_audio = job.get("ref_audio")
    items = job["items"]

    from vieneu import Vieneu
    vieneu = Vieneu()  # load model 1 lần, dùng cho toàn bộ scene

    if ref_audio:
        registered = str(Path(ref_audio).resolve())
        voice = f"__ref_{Path(registered).stem}"
        vieneu.add_voice(voice, registered)
    elif not voice:
        voice = "Adam"

    for item in items:
        out_path = Path(item["out_path"])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        audio = vieneu.infer(item["text"], voice=voice)
        vieneu.save(audio, str(out_path))

    print("WORKER_OK")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"WORKER_ERROR: {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)