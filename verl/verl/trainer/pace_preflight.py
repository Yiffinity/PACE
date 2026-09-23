"""Fail-fast checks for PACE model and tokenizer contracts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


class PreflightError(RuntimeError):
    pass


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise PreflightError(f"required model metadata is missing: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PreflightError(f"cannot read valid JSON from {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise PreflightError(f"expected an object in {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def _weight_manifest(root: Path) -> dict[str, Any]:
    weights = sorted(root.glob("*.safetensors"))
    if not weights:
        raise PreflightError(f"{root}: no safetensors weight files found")
    index = root / "model.safetensors.index.json"
    return {
        "index_sha256": _sha256(index) if index.is_file() else None,
        "shards": [
            {"name": path.name, "size": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}
            for path in weights
        ],
    }



def inspect_checkpoint(path: str | Path) -> dict[str, Any]:
    root = Path(path).expanduser().resolve()
    config = _read_json(root / "config.json")
    processor = _read_json(root / "preprocessor_config.json")
    architecture = config.get("architectures")
    vision = config.get("vision_config")
    image_token_id = config.get("image_token_id")
    vocab_size = config.get("text_config", {}).get("vocab_size", config.get("vocab_size"))
    if not isinstance(architecture, list) or not architecture:
        raise PreflightError(f"{root}: config.architectures is missing")
    if not isinstance(vision, Mapping) or image_token_id is None:
        raise PreflightError(f"{root}: model is not image-capable (vision_config/image_token_id missing)")
    if not processor.get("processor_class") or not processor.get("image_processor_type"):
        raise PreflightError(f"{root}: multimodal processor metadata is missing")
    if not isinstance(vocab_size, int) or vocab_size <= 0:
        raise PreflightError(f"{root}: invalid vocabulary size {vocab_size!r}")
    tokenizer_files = ("tokenizer.json", "vocab.json", "merges.txt", "tokenizer_config.json")
    input_contract_files = ("preprocessor_config.json", "chat_template.jinja")
    missing = [name for name in tokenizer_files + input_contract_files if not (root / name).is_file()]
    if missing:
        raise PreflightError(f"{root}: tokenizer files missing: {missing}")
    return {
        "path": str(root),
        "architecture": architecture[0],
        "model_type": config.get("model_type"),
        "processor_class": processor["processor_class"],
        "image_processor_type": processor["image_processor_type"],
        "image_token_id": image_token_id,
        "vocab_size": vocab_size,
        "tokenizer_hashes": {name: _sha256(root / name) for name in tokenizer_files},
        "input_contract_hashes": {name: _sha256(root / name) for name in input_contract_files},
        "weight_manifest": _weight_manifest(root),
    }


def validate_model_pair(teacher_path: str | Path, student_path: str | Path) -> dict[str, Any]:
    teacher = inspect_checkpoint(teacher_path)
    student = inspect_checkpoint(student_path)
    if teacher["vocab_size"] != student["vocab_size"]:
        raise PreflightError(
            f"token-level KL forbidden: teacher vocab={teacher['vocab_size']} != student vocab={student['vocab_size']}"
        )
    if teacher["tokenizer_hashes"] != student["tokenizer_hashes"]:
        raise PreflightError("token-level KL forbidden: teacher and student tokenizer files are not identical")
    if teacher["input_contract_hashes"] != student["input_contract_hashes"]:
        raise PreflightError("teacher and student processor/chat-template inputs are not identical")
    if teacher["image_token_id"] != student["image_token_id"]:
        raise PreflightError("teacher and student use different image token ids")
    return {"compatible": True, "teacher": teacher, "student": student}


def assert_runtime_multimodal(model: Any, processor: Any, *, role: str) -> None:
    config = getattr(model, "config", None)
    if config is None or getattr(config, "vision_config", None) is None or getattr(config, "image_token_id", None) is None:
        raise PreflightError(f"{role} runtime model cannot accept images")
    if processor is None or getattr(processor, "image_processor", None) is None:
        raise PreflightError(f"{role} runtime processor cannot process images")
