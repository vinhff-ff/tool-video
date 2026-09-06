"""
llm.py - Phase 3

Thin wrapper around llama-cpp-python for running an instruct GGUF model
(Qwen2.5-7B-Instruct Q4_K_M by default) locally on Kaggle's free GPU.

The model is lazy-loaded on first use and cached for the rest of the run.
LLM replies are parsed tolerantly for JSON (strips markdown fences, finds the
first balanced {…} or […] block), so stray wording from the 7B model won't
break downstream parsing.
"""

import json
import re
from pathlib import Path
import ast 
from huggingface_hub import hf_hub_download, list_repo_files

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_REPO = "Qwen/Qwen2.5-7B-Instruct-GGUF"
DEFAULT_QUANT = "q4_k_m"  # quant substring; None => any *.gguf
DEFAULT_MODEL_DIR = BASE_DIR / "models"

_MODEL = None  # cache the loaded Llama instance


def _pick_quant_files(files: list, quant: str) -> list:
    """Filter a repo file list down to the requested quant's GGUF file(s).

    Works for both layouts:
    - single merged file:      `foo-q4_k_m.gguf`
    - split quant:             `foo-q4_k_m-00001-of-00002.gguf`,
                               `foo-q4_k_m-00002-of-00002.gguf`
    Falls back to split parts when no merged file exists, sorted so the
    `-00001-of-…` part comes first.
    """
    gguf = [f for f in files if f.lower().endswith(".gguf")]
    if quant:
        gguf = [f for f in gguf if quant.lower() in f.lower()]
    if not gguf:
        raise ValueError(
            f"No GGUF file matching quant={quant!r} in repo."
            f" Available files: {[f for f in files]}"
        )
    singles = sorted([f for f in gguf if "-of-" not in f.lower()])
    if singles:
        return singles
    return sorted(gguf)


def ensure_model(
    repo: str = DEFAULT_REPO,
    quant: str = DEFAULT_QUANT,
    local_dir=None,
) -> str:
    """Download every part of the requested quant GGUF if missing.

    Returns the path to pass to llama.cpp:
    - the single merged file when one exists, otherwise
    - the first part (`-00001-of-…`) of a split quant.

    llama.cpp / llama-cpp-python auto-join the remaining parts as long as they
    sit in the same directory with the same prefix (which `local_dir` ensures).
    """
    local_dir = Path(local_dir or DEFAULT_MODEL_DIR)
    local_dir.mkdir(parents=True, exist_ok=True)

    print(f"[llm] Listing files in {repo}/main …")
    names = _pick_quant_files(list_repo_files(repo_id=repo), quant)

    first = None
    for name in names:
        local_file = local_dir / name
        if not local_file.exists():
            print(f"[llm] Downloading {repo}/{name} …")
            hf_hub_download(repo_id=repo, filename=name, local_dir=str(local_dir))
        if first is None:
            first = local_file
    return str(first)


def get_model(
    model_path: str = None,
    n_ctx: int = 8192,
    n_gpu_layers: int = -1,
    quant: str = DEFAULT_QUANT,
):
    """Lazy singleton: load the Llama model once, reuse for all calls."""
    global _MODEL
    if _MODEL is None:
        path = str(Path(model_path).resolve()) if model_path else ensure_model(quant=quant)
        from llama_cpp import Llama

        print(f"[llm] Loading model {path} …")
        try:
            _MODEL = Llama(
                model_path=path,
                n_ctx=n_ctx,
                n_gpu_layers=n_gpu_layers,  # -1 = all layers on GPU (Kaggle T4/P100)
                verbose=False,
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to load {path}.\n"
                f"If this llama-cpp-python build cannot auto-merge split GGUF files, "
                f"merge them manually first:\n"
                f"  llama-gguf-split --merge {path} {path}.merged.gguf\n"
                f"Then pass model_path pointing to the merged file.\n"
                f"Original error: {e}"
            ) from e
    return _MODEL


def _extract_json(text: str):
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    def _try_parse(s: str):
        # 1. Thử JSON chuẩn trước
        try:
            return json.loads(s)
        except Exception:
            pass
        # 2. Fallback: parse như Python literal — bắt được case model
        #    trộn lẫn single/double quote (vẫn là cú pháp Python hợp lệ)
        try:
            result = ast.literal_eval(s)
            if isinstance(result, (list, dict)):
                return result
        except Exception:
            pass
        return None

    result = _try_parse(text)
    if result is not None:
        return result

    for open_ch, close_ch in (("{", "}"), ("[", "]")):
        start = text.find(open_ch)
        if start == -1:
            continue
        depth = 0
        for i in range(start, len(text)):
            if text[i] == open_ch:
                depth += 1
            elif text[i] == close_ch:
                depth -= 1
                if depth == 0:
                    result = _try_parse(text[start:i + 1])
                    if result is not None:
                        return result
                    break

    raise ValueError(f"Cannot parse JSON from LLM output:\n{text[:500]}")


def generate_json(
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.7,
    max_tokens: int = 2048,
    model_path: str = None,
    quant: str = DEFAULT_QUANT,
):
    """Chat completion that returns parsed JSON (tolerant parsing)."""
    llm = get_model(model_path=model_path, quant=quant)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    out = llm.create_chat_completion(
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    reply = out["choices"][0]["message"]["content"]
    return _extract_json(reply)