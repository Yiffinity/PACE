"""Dataset normalization and VeRL conversion for multimodal sarcasm detection."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from verl.trainer.pace_prompts import verl_blind_messages
from verl.trainer.pace_schema import Reasoning, SarcasmSample, SchemaError, validate_reasoning, validate_sample


FIELD_ALIASES = {
    "sample_id": ("sample_id", "id", "guid", "uid"),
    "image_path": ("image_path", "image", "img_path", "photo_path"),
    "text": ("text", "post_text", "caption", "tweet", "sentence"),
    "label": ("label", "gold_label", "target", "class"),
}

LABEL_ALIASES = {
    "sarcastic": "sarcastic",
    "sarcasm": "sarcastic",
    "ironic": "sarcastic",
    "1": "sarcastic",
    1: "sarcastic",
    True: "sarcastic",
    "non-sarcastic": "non-sarcastic",
    "non_sarcastic": "non-sarcastic",
    "nonsarcastic": "non-sarcastic",
    "not_sarcastic": "non-sarcastic",
    "0": "non-sarcastic",
    0: "non-sarcastic",
    False: "non-sarcastic",
}


def _first(record: Mapping[str, Any], names: tuple[str, ...]) -> Any:
    for name in names:
        if name in record and record[name] is not None:
            return record[name]
    return None


def normalize_label(value: Any) -> str:
    key = value.strip().lower() if isinstance(value, str) else value
    try:
        return LABEL_ALIASES[key]
    except (KeyError, TypeError) as exc:
        raise SchemaError(f"unsupported sarcasm label {value!r}") from exc


def normalize_record(record: Mapping[str, Any], *, base_dir: str | Path, require_image: bool = True) -> SarcasmSample:
    normalized: dict[str, Any] = {name: _first(record, aliases) for name, aliases in FIELD_ALIASES.items()}
    if normalized["sample_id"] is None:
        stable = json.dumps(record, ensure_ascii=True, sort_keys=True, default=str).encode("utf-8")
        normalized["sample_id"] = f"generated:{hashlib.sha256(stable).hexdigest()[:24]}"
    normalized["label"] = normalize_label(normalized["label"])
    image_path = normalized["image_path"]
    if isinstance(image_path, Mapping):
        image_path = image_path.get("path") or image_path.get("image")
    if not isinstance(image_path, str):
        raise SchemaError(f"sample {normalized['sample_id']!r}: image path is missing")
    image = Path(image_path).expanduser()
    if not image.is_absolute():
        image = Path(base_dir) / image
    normalized["image_path"] = str(image.resolve())
    normalized["source"] = record.get("source") or record.get("dataset")
    normalized["language"] = record.get("language") or record.get("lang")
    normalized["extraction_metadata"] = {
        key: record[key]
        for key in ("text_label", "image_label", "img_label")
        if key in record and record[key] is not None
    }

    reference = record.get("reference_reasoning")
    if reference is None and isinstance(record.get("analysis"), Mapping) and record.get("reference_valid") is True:
        reference = record["analysis"]
    if reference is not None:
        parsed = validate_reasoning(reference)
        normalized["reference_reasoning"] = parsed.to_dict()
    return validate_sample(normalized, require_image=require_image)


def to_verl_record(sample: SarcasmSample, *, split: str, index: int) -> dict[str, Any]:
    """Return only trainer-facing fields; references and extraction metadata never enter VeRL."""
    return {
        "data_source": "pace_msd",
        "prompt": verl_blind_messages(sample.text),
        "images": [{"image": sample.image_path}],
        "ability": "multimodal_sarcasm_detection",
        "reward_model": {"style": "rule", "ground_truth": sample.label},
        "extra_info": {
            "split": split,
            "index": index,
            "sample_id": sample.sample_id,
            "gold_label": sample.label,
            "source": sample.source,
            "language": sample.language,
        },
    }
