"""Experience-free multimodal sarcasm evaluation and classification metrics."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from verl.trainer.pace_plus_config import project_path
from verl.trainer.pace_plus_data import normalize_record
from verl.trainer.pace_plus_modeling import MultimodalGenerator
from verl.trainer.pace_plus_metrics import classification_metrics
from verl.trainer.pace_plus_pool import load_active_experience_text
from verl.trainer.pace_plus_preflight import inspect_checkpoint
from verl.trainer.pace_plus_prompts import BLIND_REASONING_SYSTEM_PROMPT, blind_messages, experience_conditioned_messages
from verl.trainer.pace_plus_schema import (
    generate_json_with_retries,
    load_records_by_id,
    parse_reasoning,
    read_jsonl,
    upsert_jsonl,
)
def _content_fingerprint(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _evaluation_fingerprint(
    *,
    model_fingerprint: str,
    mode: str,
    generation: Mapping[str, Any],
    experience: str,
    sample: Any,
) -> str:
    value = {
        "model": model_fingerprint,
        "mode": mode,
        "generation": dict(generation),
        "prompt": BLIND_REASONING_SYSTEM_PROMPT,
        "experience": experience,
        "sample_id": sample.sample_id,
        "text": sample.text,
        "image_sha256": _content_fingerprint(sample.image_path),
    }
    return hashlib.sha256(json.dumps(value, ensure_ascii=True, sort_keys=True).encode("utf-8")).hexdigest()






def run_evaluation(
    config: Mapping[str, Any], checkpoint: str | Path, data_path: str | Path, *, limit: int | None = None
) -> dict[str, Any]:
    checkpoint = Path(checkpoint).expanduser().resolve()
    inspect_checkpoint(checkpoint)
    mode = config.get("ablation", {}).get("inference_experience_mode", "none")
    if mode not in {"none", "full_pool"}:
        raise ValueError(f"unsupported ablation.inference_experience_mode: {mode!r}")
    experience = ""
    if mode == "full_pool":
        experience = load_active_experience_text(
            project_path(config, config.get("experience_pool", {}).get("path", "experience_pool/experience_pool.json"))
        )
    source = Path(data_path).expanduser().resolve()
    samples = []
    seen: set[str] = set()
    for line_number, record in read_jsonl(source):
        sample = normalize_record(record, base_dir=source.parent, require_image=True)
        if sample.sample_id in seen:
            raise ValueError(f"{source}:{line_number}: duplicate sample_id {sample.sample_id!r}")
        seen.add(sample.sample_id)
        samples.append(sample)
        if limit is not None and len(samples) >= limit:
            break
    if not samples:
        raise ValueError(f"no valid samples found in {source}")
    output_dir = project_path(config, config.get("evaluation", {}).get("output_path", "evaluations"))
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "predictions.jsonl"
    existing = load_records_by_id(results_path, allow_missing=True)
    generation = config.get("evaluation", {}).get("generation", {})
    json_config = generation.get("json", {})
    json_max_attempts = int(json_config.get("max_attempts", 3))
    failure_log_path = project_path(config, json_config.get("failure_log_path", "logs/evaluation_json_failures.jsonl"))
    with MultimodalGenerator(
        checkpoint,
        role="evaluation_student",
        device_map=generation.get("device_map", "auto"),
        max_new_tokens=int(generation.get("max_new_tokens", 768)),
        deterministic=bool(generation.get("deterministic", True)),
        chat_template_kwargs=generation.get("chat_template_kwargs", {}),
    ) as student:
        for sample in samples:
            fingerprint = _evaluation_fingerprint(
                model_fingerprint=student.fingerprint,
                mode=mode,
                generation=generation,
                experience=experience,
                sample=sample,
            )
            cached = existing.get(sample.sample_id)
            if cached and cached.get("fingerprint") == fingerprint and cached.get("json_valid") is True and not cached.get("error"):
                continue
            messages = blind_messages(sample.text)
            if experience:
                messages = experience_conditioned_messages(messages, experience)
            parsed, raw, error, attempts = generate_json_with_retries(
                student,
                messages,
                image_path=sample.image_path,
                parser=parse_reasoning,
                max_attempts=json_max_attempts,
                failure_log_path=failure_log_path,
                context={
                    "stage": "evaluation",
                    "artifact": "student_reasoning",
                    "sample_id": sample.sample_id,
                    "model_path": str(checkpoint),
                },
            )
            record = {
                "sample_id": sample.sample_id,
                "gold_label": sample.label,
                "analysis": parsed.to_dict() if parsed else None,
                "raw_output": raw,
                "error": error,
                "attempts": attempts,
                "json_valid": parsed is not None,
                "fingerprint": fingerprint,
                "inference_experience_mode": mode,
            }
            upsert_jsonl(results_path, record)
            existing[sample.sample_id] = record
    gold: list[str] = []
    predicted: list[str] = []
    invalid = 0
    for sample in samples:
        record = existing.get(sample.sample_id)
        if not record or not isinstance(record.get("analysis"), dict):
            invalid += 1
            continue
        gold.append(sample.label)
        predicted.append(record["analysis"]["label"])
    metrics = classification_metrics(gold, predicted, invalid=invalid)
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=True, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return metrics
