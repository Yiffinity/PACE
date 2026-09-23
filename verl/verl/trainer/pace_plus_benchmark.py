"""Resumable teacher/student benchmark evaluation on the four local test sets."""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import multiprocessing
import os
import signal
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from verl.trainer.pace_plus_config import project_path
from verl.trainer.pace_plus_data import normalize_record
from verl.trainer.pace_plus_metrics import classification_metrics
from verl.trainer.pace_plus_modeling import VllmMultimodalGenerator
from verl.trainer.pace_plus_pool import load_active_experience_text
from verl.trainer.pace_plus_preflight import inspect_checkpoint
from verl.trainer.pace_plus_prompts import (
    BLIND_REASONING_SYSTEM_PROMPT,
    blind_messages,
    experience_conditioned_messages,
)
from verl.trainer.pace_plus_schema import (
    REASONING_JSON_SCHEMA,
    SarcasmSample,
    append_jsonl,
    parse_reasoning,
    read_jsonl,
)


DATASET_ORDER = ("mmsd2", "sarcnet", "docmsu", "redeval")
ROLE_ORDER = ("teacher", "student")
ALIASES = {
    "mmsd2": "mmsd2",
    "mmsd2.0": "mmsd2",
    "mmsd-v2": "mmsd2",
    "sarcnet": "sarcnet",
    "docmsu": "docmsu",
    "redeval": "redeval",
}
SUMMARY_METRICS = (
    "accuracy",
    "valid_accuracy",
    "precision",
    "recall",
    "f1",
    "specificity",
    "balanced_accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "weighted_f1",
    "micro_f1",
    "prediction_coverage",
    "matthews_correlation_coefficient_valid",
    "cohen_kappa_valid",
)

_PAUSE_REQUESTED = False
_ACTIVE_PAUSE_FILE: Path | None = None


class BenchmarkError(RuntimeError):
    pass


@dataclass(frozen=True)
class BenchmarkDataset:
    name: str
    samples: list[SarcasmSample]
    sources: list[str]


@dataclass
class _DatasetState:
    dataset: BenchmarkDataset
    output_dir: Path
    output_path: Path
    journal_path: Path
    records: dict[str, dict[str, Any]]
    fingerprints: dict[str, str]
    pending: list[tuple[SarcasmSample, str]]


def _stable_hash(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=True, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_teacher_experience_context(
    config: Mapping[str, Any],
    value: str | Path,
    *,
    expected_active: int | None = None,
) -> tuple[str, dict[str, Any]]:
    source = project_path(config, value)
    experience = load_active_experience_text(source)
    if not experience.strip():
        raise BenchmarkError(f"teacher experience context is empty: {source}")
    active_count: int | None = None
    if source.suffix.casefold() == ".json":
        try:
            payload = json.loads(source.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise BenchmarkError(f"invalid teacher experience pool JSON: {source}: {exc}") from exc
        entries = payload.get("experiences") if isinstance(payload, Mapping) else None
        if isinstance(entries, list):
            active_count = sum(
                isinstance(item, Mapping) and item.get("active") is True for item in entries
            )
    if expected_active is not None:
        if expected_active < 1:
            raise BenchmarkError("expected active experience count must be positive")
        if active_count is None:
            raise BenchmarkError(
                "expected active experience count can only be checked for a JSON experience pool"
            )
        if active_count != expected_active:
            raise BenchmarkError(
                f"teacher experience pool has {active_count} active entries; expected {expected_active}"
            )
    metadata = {
        "path": str(source),
        "active_count": active_count,
        "sha256": hashlib.sha256(experience.encode("utf-8")).hexdigest(),
        "characters": len(experience),
        "words": len(experience.split()),
    }
    return experience, metadata


def _json_array(path: Path) -> list[dict[str, Any]]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise BenchmarkError(f"{path}: invalid JSON: {exc}") from exc
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise BenchmarkError(f"{path}: expected an array of JSON objects")
    return value


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [record for _, record in read_jsonl(path)]


def _dataset_config(config: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    section = config.get("benchmark_evaluation", {}).get("datasets", {}).get(name, {})
    if not isinstance(section, Mapping):
        raise BenchmarkError(f"benchmark_evaluation.datasets.{name} must be a mapping")
    return section


def _path(config: Mapping[str, Any], value: str) -> Path:
    resolved = project_path(config, value)
    if not resolved.is_file():
        raise BenchmarkError(f"benchmark test split does not exist: {resolved}")
    return resolved


def _directory(config: Mapping[str, Any], value: str) -> Path:
    resolved = project_path(config, value)
    if not resolved.is_dir():
        raise BenchmarkError(f"benchmark image directory does not exist: {resolved}")
    return resolved


def _normalized_sample(
    raw: Mapping[str, Any],
    *,
    sample_id: str,
    image_path: Path,
    dataset: str,
    language: str | None,
) -> SarcasmSample:
    record = {
        **dict(raw),
        "sample_id": sample_id,
        "image_path": str(image_path),
        "source": dataset,
        "language": language,
    }
    return normalize_record(record, base_dir=image_path.parent, require_image=True)


def load_benchmark_dataset(
    config: Mapping[str, Any],
    name: str,
    *,
    limit: int | None = None,
) -> BenchmarkDataset:
    canonical = ALIASES.get(name.casefold())
    if canonical is None:
        raise BenchmarkError(f"unsupported benchmark dataset {name!r}; choose from {list(DATASET_ORDER)}")
    section = _dataset_config(config, canonical)
    samples: list[SarcasmSample] = []
    sources: list[str] = []

    if canonical == "mmsd2":
        source = _path(
            config,
            str(section.get("test_path", "datasets/MMSD2.0/data/text_json_final/test.json")),
        )
        image_dir = _directory(
            config,
            str(section.get("image_dir", "datasets/MMSD2.0/data/dataset_image")),
        )
        sources.append(str(source))
        for raw in _json_array(source):
            image_id = str(raw.get("image_id"))
            samples.append(
                _normalized_sample(
                    raw,
                    sample_id=f"mmsd2:{image_id}",
                    image_path=image_dir / f"{image_id}.jpg",
                    dataset=canonical,
                    language="en",
                )
            )
    elif canonical == "redeval":
        source = _path(
            config,
            str(section.get("test_path", "datasets/RedEval/reddit_test.json")),
        )
        image_dir = _directory(
            config,
            str(section.get("image_dir", "datasets/RedEval/images")),
        )
        sources.append(str(source))
        for raw in _json_array(source):
            image_id = str(raw.get("image_id"))
            samples.append(
                _normalized_sample(
                    raw,
                    sample_id=f"redeval:{image_id}",
                    image_path=image_dir / f"{image_id}.jpg",
                    dataset=canonical,
                    language="en",
                )
            )
    elif canonical == "docmsu":
        source = _path(
            config,
            str(section.get("test_path", "datasets/DocMSU/data/test.jsonl")),
        )
        sources.append(str(source))
        for raw in _jsonl(source):
            original_id = str(raw.get("sample_id"))
            image_path = Path(str(raw.get("image_path", ""))).expanduser()
            if not image_path.is_absolute():
                image_path = source.parent / image_path
            samples.append(
                _normalized_sample(
                    raw,
                    sample_id=f"docmsu:{original_id}",
                    image_path=image_path.resolve(),
                    dataset=canonical,
                    language=str(raw.get("language") or "en"),
                )
            )
    else:
        configured = section.get(
            "test_paths",
            {
                "en": "datasets/sarcnet/data/en/test.jsonl",
                "zh": "datasets/sarcnet/data/zh/test.jsonl",
            },
        )
        if not isinstance(configured, Mapping) or not configured:
            raise BenchmarkError("benchmark_evaluation.datasets.sarcnet.test_paths must be a mapping")
        for language in ("en", "zh"):
            if language not in configured:
                continue
            source = _path(config, str(configured[language]))
            sources.append(str(source))
            for raw in _jsonl(source):
                original_id = str(raw.get("sample_id"))
                image_path = Path(str(raw.get("image_path", ""))).expanduser()
                if not image_path.is_absolute():
                    image_path = source.parent / image_path
                samples.append(
                    _normalized_sample(
                        raw,
                        sample_id=f"sarcnet:{language}:{original_id}",
                        image_path=image_path.resolve(),
                        dataset=canonical,
                        language=language,
                    )
                )

    if limit is not None:
        if limit < 1:
            raise BenchmarkError("benchmark limit must be positive")
        samples = samples[:limit]
    if not samples:
        raise BenchmarkError(f"no samples found for benchmark dataset {canonical}")
    seen: set[str] = set()
    for sample in samples:
        if sample.sample_id in seen:
            raise BenchmarkError(f"{canonical}: duplicate sample_id {sample.sample_id!r}")
        seen.add(sample.sample_id)
    return BenchmarkDataset(name=canonical, samples=samples, sources=sources)


def load_benchmark_datasets(
    config: Mapping[str, Any],
    names: Sequence[str] | None = None,
    *,
    limit: int | None = None,
) -> list[BenchmarkDataset]:
    requested = list(names or DATASET_ORDER)
    canonical: list[str] = []
    for name in requested:
        resolved = ALIASES.get(name.casefold())
        if resolved is None:
            raise BenchmarkError(f"unsupported benchmark dataset {name!r}")
        if resolved not in canonical:
            canonical.append(resolved)
    return [load_benchmark_dataset(config, name, limit=limit) for name in canonical]


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=True, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _atomic_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _load_prediction_records(output_path: Path) -> tuple[dict[str, dict[str, Any]], Path]:
    records: dict[str, dict[str, Any]] = {}
    if output_path.exists():
        for line_number, record in read_jsonl(output_path):
            sample_id = record.get("sample_id")
            if not isinstance(sample_id, str) or not sample_id:
                raise BenchmarkError(f"{output_path}:{line_number}: missing sample_id")
            if sample_id in records:
                raise BenchmarkError(f"{output_path}:{line_number}: duplicate sample_id {sample_id!r}")
            records[sample_id] = record
    journal_path = output_path.with_name(f".{output_path.name}.pending")
    if journal_path.exists():
        journal_records: list[dict[str, Any]] = []
        recovered_tail = False
        with journal_path.open("r", encoding="utf-8") as handle:
            lines = handle.readlines()
        for line_number, line in enumerate(lines, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                if line_number == len(lines):
                    recovered_tail = True
                    break
                raise BenchmarkError(
                    f"{journal_path}:{line_number}: invalid JSON: {exc}"
                ) from exc
            if not isinstance(record, dict):
                raise BenchmarkError(f"{journal_path}:{line_number}: expected a JSON object")
            sample_id = record.get("sample_id")
            if not isinstance(sample_id, str) or not sample_id:
                raise BenchmarkError(f"{journal_path}:{line_number}: missing sample_id")
            journal_records.append(record)
            records[sample_id] = record
        if recovered_tail:
            _atomic_jsonl(journal_path, journal_records)
    return records, journal_path


def _append_prediction_records(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _compact_predictions(state: _DatasetState) -> None:
    current_ids = [sample.sample_id for sample in state.dataset.samples]
    current = [state.records[sample_id] for sample_id in current_ids if sample_id in state.records]
    extra = [
        record for sample_id, record in state.records.items() if sample_id not in set(current_ids)
    ]
    _atomic_jsonl(state.output_path, [*current, *extra])
    state.journal_path.unlink(missing_ok=True)


def _generation_fingerprint(generation: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "backend",
        "deterministic",
        "max_new_tokens",
        "thinking_token_budget",
        "max_model_len",
        "chat_template_kwargs",
    )
    return {
        **{key: generation.get(key) for key in keys},
        "json_max_attempts": generation.get("json", {}).get("max_attempts", 3),
        "reasoning_schema": REASONING_JSON_SCHEMA,
    }


def _sample_fingerprint(base: str, dataset: str, sample: SarcasmSample) -> str:
    image = Path(sample.image_path)
    stat = image.stat()
    return _stable_hash(
        {
            "base": base,
            "dataset": dataset,
            "sample_id": sample.sample_id,
            "text": sample.text,
            "gold_label": sample.label,
            "image_path": str(image),
            "image_size": stat.st_size,
            "image_mtime_ns": stat.st_mtime_ns,
        }
    )


def _valid_analysis(record: Mapping[str, Any] | None) -> bool:
    if not record or record.get("json_valid") is not True or record.get("error"):
        return False
    try:
        parse_reasoning(json.dumps(record.get("analysis"), ensure_ascii=True))
    except Exception:
        return False
    return True


def _completed_cache(
    record: Mapping[str, Any] | None,
    fingerprint: str,
    *,
    retry_invalid: bool,
) -> bool:
    if not record or record.get("fingerprint") != fingerprint:
        return False
    if _valid_analysis(record):
        return True
    return not retry_invalid and record.get("json_valid") is False


def _plan_dataset(
    *,
    dataset: BenchmarkDataset,
    output_root: Path,
    role: str,
    base_fingerprint: str,
    retry_invalid: bool,
    compatible_base_fingerprints: Sequence[str] = (),
) -> _DatasetState:
    output_dir = output_root / role / dataset.name
    output_path = output_dir / "predictions.jsonl"
    records, journal_path = _load_prediction_records(output_path)
    fingerprints = {
        sample.sample_id: _sample_fingerprint(base_fingerprint, dataset.name, sample)
        for sample in dataset.samples
    }
    pending: list[tuple[SarcasmSample, str]] = []
    migrated: list[dict[str, Any]] = []
    for sample in dataset.samples:
        fingerprint = fingerprints[sample.sample_id]
        cached = records.get(sample.sample_id)
        if _completed_cache(cached, fingerprint, retry_invalid=retry_invalid):
            continue
        migrated_cache = False
        if _valid_analysis(cached):
            for compatible_base in compatible_base_fingerprints:
                compatible_fingerprint = _sample_fingerprint(
                    compatible_base,
                    dataset.name,
                    sample,
                )
                if cached.get("fingerprint") == compatible_fingerprint:
                    updated = {
                        **cached,
                        "fingerprint": fingerprint,
                        "cache_migrated_from": compatible_fingerprint,
                    }
                    records[sample.sample_id] = updated
                    migrated.append(updated)
                    migrated_cache = True
                    break
        if not migrated_cache:
            pending.append((sample, fingerprint))
    if migrated:
        _append_prediction_records(journal_path, migrated)
    return _DatasetState(
        dataset=dataset,
        output_dir=output_dir,
        output_path=output_path,
        journal_path=journal_path,
        records=records,
        fingerprints=fingerprints,
        pending=pending,
    )


def _generate_with_isolation(
    generator: Any,
    requests: list[tuple[Sequence[Mapping[str, Any]], str]],
) -> tuple[list[str | None], list[str | None]]:
    outputs: list[str | None] = [None] * len(requests)
    errors: list[str | None] = [None] * len(requests)

    def generate(indices: list[int]) -> None:
        try:
            generated = generator.generate_batch([requests[index] for index in indices])
        except ValueError as exc:
            if len(indices) == 1:
                errors[indices[0]] = f"generation failed: {type(exc).__name__}: {exc}"
                return
            midpoint = len(indices) // 2
            generate(indices[:midpoint])
            generate(indices[midpoint:])
            return
        if len(generated) != len(indices):
            raise BenchmarkError(
                f"batch generator returned {len(generated)} outputs for {len(indices)} requests"
            )
        for index, output in zip(indices, generated, strict=True):
            outputs[index] = output

    if requests:
        generate(list(range(len(requests))))
    return outputs, errors


def _generate_reasoning_batch(
    *,
    generator: Any,
    batch: list[tuple[SarcasmSample, str]],
    role: str,
    dataset: str,
    model_path: Path,
    failure_log_path: Path,
    max_attempts: int,
    experience: str = "",
) -> list[dict[str, Any]]:
    experience_sha256 = (
        hashlib.sha256(experience.encode("utf-8")).hexdigest() if experience else None
    )
    states = [
        {
            "sample": sample,
            "fingerprint": fingerprint,
            "messages": experience_conditioned_messages(
                blind_messages(sample.text), experience
            ),
            "raw": None,
            "error": None,
            "attempt_count": 0,
            "attempts": [],
            "parsed": None,
        }
        for sample, fingerprint in batch
    ]
    remaining = list(range(len(states)))
    for attempt in range(1, max_attempts + 1):
        requests = [
            (states[index]["messages"], states[index]["sample"].image_path)
            for index in remaining
        ]
        outputs, generation_errors = _generate_with_isolation(generator, requests)
        retry: list[int] = []
        for index, raw, generation_error in zip(
            remaining, outputs, generation_errors, strict=True
        ):
            state = states[index]
            state["raw"] = raw
            state["attempt_count"] = attempt
            if generation_error is not None:
                error = generation_error
            else:
                try:
                    state["parsed"] = parse_reasoning(raw)
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
                else:
                    state["error"] = None
                    continue
            state["error"] = error
            state["attempts"].append(
                {"attempt": attempt, "error": error, "raw_output": raw}
            )
            prompt_too_long = "maximum model length" in error.casefold()
            if attempt < max_attempts and not prompt_too_long:
                state["messages"] = [
                    *state["messages"],
                    {
                        "role": "user",
                        "content": (
                            "Your previous response did not satisfy the required four-field JSON "
                            "contract. Return exactly visual_evidence, textual_evidence, explanation, "
                            "and label, with no Markdown, prose, or extra fields. Never copy a double "
                            "quote character into a narrative value; replace quoted text with single "
                            "quotes or paraphrase it while preserving its meaning. "
                            f"Validation error: {error}"
                        ),
                    },
                ]
                retry.append(index)
        remaining = retry
        if not remaining:
            break

    records: list[dict[str, Any]] = []
    for state in states:
        sample = state["sample"]
        parsed = state["parsed"]
        if parsed is None:
            append_jsonl(
                failure_log_path,
                {
                    "stage": "benchmark_evaluation",
                    "dataset": dataset,
                    "role": role,
                    "sample_id": sample.sample_id,
                    "model_path": str(model_path),
                    "max_attempts": max_attempts,
                    "attempts": state["attempts"],
                    "error": state["error"] or "invalid JSON output",
                },
            )
        records.append(
            {
                "sample_id": sample.sample_id,
                "dataset": dataset,
                "role": role,
                "text": sample.text,
                "image_path": sample.image_path,
                "language": sample.language,
                "gold_label": sample.label,
                "analysis": parsed.to_dict() if parsed else None,
                "raw_output": state["raw"],
                "error": None if parsed else state["error"],
                "attempts": state["attempt_count"],
                "json_valid": parsed is not None,
                "model_path": str(model_path),
                "fingerprint": state["fingerprint"],
                "teacher_experience_sha256": experience_sha256,
                "reasoning_fields": [
                    "visual_evidence",
                    "textual_evidence",
                    "explanation",
                    "label",
                ],
            }
        )
    return records


def _prediction_vectors(
    samples: Sequence[SarcasmSample],
    records: Mapping[str, Mapping[str, Any]],
    fingerprints: Mapping[str, str],
) -> tuple[list[str], list[str], list[str], int]:
    gold: list[str] = []
    predicted: list[str] = []
    invalid_gold: list[str] = []
    completed = 0
    for sample in samples:
        record = records.get(sample.sample_id)
        if not record or record.get("fingerprint") != fingerprints[sample.sample_id]:
            continue
        completed += 1
        if _valid_analysis(record):
            analysis = parse_reasoning(json.dumps(record["analysis"], ensure_ascii=True))
            gold.append(sample.label)
            predicted.append(analysis.label)
        else:
            invalid_gold.append(sample.label)
    return gold, predicted, invalid_gold, completed


def _dataset_metrics(state: _DatasetState, *, status: str) -> dict[str, Any]:
    gold, predicted, invalid_gold, completed = _prediction_vectors(
        state.dataset.samples,
        state.records,
        state.fingerprints,
    )
    result = classification_metrics(
        gold,
        predicted,
        invalid=len(invalid_gold),
        invalid_gold=invalid_gold,
    )
    result.update(
        {
            "dataset": state.dataset.name,
            "status": status,
            "dataset_samples": len(state.dataset.samples),
            "completed_samples": completed,
            "remaining_samples": len(state.dataset.samples) - completed,
            "completion_rate": completed / len(state.dataset.samples),
            "sources": state.dataset.sources,
        }
    )
    languages = sorted(
        {sample.language for sample in state.dataset.samples if sample.language}
    )
    if len(languages) > 1:
        result["subgroups"] = {}
        for language in languages:
            subgroup = [
                sample for sample in state.dataset.samples if sample.language == language
            ]
            sub_gold, sub_predicted, sub_invalid, sub_completed = _prediction_vectors(
                subgroup,
                state.records,
                state.fingerprints,
            )
            submetrics = classification_metrics(
                sub_gold,
                sub_predicted,
                invalid=len(sub_invalid),
                invalid_gold=sub_invalid,
            )
            submetrics.update(
                {
                    "dataset_samples": len(subgroup),
                    "completed_samples": sub_completed,
                    "completion_rate": sub_completed / len(subgroup),
                }
            )
            result["subgroups"][language] = submetrics
    return result


def _write_progress(state: _DatasetState, *, role: str, status: str) -> dict[str, Any]:
    metrics = _dataset_metrics(state, status=status)
    _atomic_json(state.output_dir / "metrics.json", metrics)
    progress = {
        "role": role,
        "dataset": state.dataset.name,
        "status": status,
        "total": metrics["dataset_samples"],
        "completed": metrics["completed_samples"],
        "remaining": metrics["remaining_samples"],
        "valid": metrics["valid_predictions"],
        "invalid": metrics["invalid_json"],
    }
    _atomic_json(state.output_dir / "progress.json", progress)
    return metrics


def _chunks(values: list[Any], size: int) -> Iterable[list[Any]]:
    for offset in range(0, len(values), size):
        yield values[offset : offset + size]


def _pause_requested(pause_file: Path) -> bool:
    return _PAUSE_REQUESTED or pause_file.exists()


def _handle_pause_signal(_signum: int, _frame: Any) -> None:
    global _PAUSE_REQUESTED
    _PAUSE_REQUESTED = True
    if _ACTIVE_PAUSE_FILE is not None:
        _ACTIVE_PAUSE_FILE.parent.mkdir(parents=True, exist_ok=True)
        _ACTIVE_PAUSE_FILE.touch(exist_ok=True)


def _evaluate_state(
    *,
    state: _DatasetState,
    generator: Any | None,
    role: str,
    model_path: Path,
    generation: Mapping[str, Any],
    pause_file: Path,
    progress_position: int,
    experience: str = "",
) -> dict[str, Any]:
    from tqdm.auto import tqdm

    if not state.pending:
        _compact_predictions(state)
        return _write_progress(state, role=role, status="complete")
    if generator is None or _pause_requested(pause_file):
        return _write_progress(state, role=role, status="paused")

    batch_size = int(generation.get("batch_size", 32))
    if batch_size < 1:
        raise BenchmarkError("benchmark_evaluation.generation.batch_size must be positive")
    max_attempts = int(generation.get("json", {}).get("max_attempts", 3))
    if not 1 <= max_attempts <= 3:
        raise BenchmarkError("benchmark JSON max_attempts must be between 1 and 3")
    completed_initial = len(state.dataset.samples) - len(state.pending)
    valid_initial = sum(
        _valid_analysis(state.records.get(sample.sample_id))
        and state.records[sample.sample_id].get("fingerprint") == state.fingerprints[sample.sample_id]
        for sample in state.dataset.samples
    )
    invalid_initial = completed_initial - valid_initial
    progress = tqdm(
        total=len(state.dataset.samples),
        initial=completed_initial,
        desc=f"{role}/{state.dataset.name}",
        unit="sample",
        dynamic_ncols=True,
        mininterval=1.0,
        position=progress_position,
        disable=not bool(generation.get("show_progress", True)),
    )
    progress.set_postfix(
        valid=valid_initial,
        invalid=invalid_initial,
        batch=batch_size,
        inflight=min(batch_size, int(generation.get("max_num_seqs", 16))),
        refresh=False,
    )
    valid_count = valid_initial
    invalid_count = invalid_initial
    try:
        for batch in _chunks(state.pending, batch_size):
            records = _generate_reasoning_batch(
                generator=generator,
                batch=batch,
                role=role,
                dataset=state.dataset.name,
                model_path=model_path,
                failure_log_path=state.output_dir / "failures.jsonl",
                max_attempts=max_attempts,
                experience=experience,
            )
            _append_prediction_records(state.journal_path, records)
            for record in records:
                state.records[record["sample_id"]] = record
                if record["json_valid"]:
                    valid_count += 1
                else:
                    invalid_count += 1
            progress.update(len(records))
            progress.set_postfix(
                valid=valid_count,
                invalid=invalid_count,
                batch=batch_size,
                inflight=min(batch_size, int(generation.get("max_num_seqs", 16))),
                refresh=False,
            )
            _write_progress(state, role=role, status="running")
            if _pause_requested(pause_file):
                break
    finally:
        progress.close()

    metrics = _dataset_metrics(state, status="running")
    complete = metrics["completed_samples"] == metrics["dataset_samples"]
    if not complete:
        status = "paused"
    elif metrics["invalid_json"]:
        status = "invalid_outputs"
    else:
        status = "complete"
    if complete:
        _compact_predictions(state)
    return _write_progress(state, role=role, status=status)


def _pooled_metrics(states: Sequence[_DatasetState]) -> dict[str, Any]:
    all_gold: list[str] = []
    all_predicted: list[str] = []
    all_invalid: list[str] = []
    completed = 0
    total = 0
    for state in states:
        gold, predicted, invalid_gold, state_completed = _prediction_vectors(
            state.dataset.samples,
            state.records,
            state.fingerprints,
        )
        all_gold.extend(gold)
        all_predicted.extend(predicted)
        all_invalid.extend(invalid_gold)
        completed += state_completed
        total += len(state.dataset.samples)
    result = classification_metrics(
        all_gold,
        all_predicted,
        invalid=len(all_invalid),
        invalid_gold=all_invalid,
    )
    result.update(
        {
            "dataset_samples": total,
            "completed_samples": completed,
            "remaining_samples": total - completed,
            "completion_rate": completed / total if total else 0.0,
        }
    )
    return result


def _role_worker(
    config: Mapping[str, Any],
    role: str,
    device: str,
    dataset_names: Sequence[str],
    limit: int | None,
    retry_invalid: bool,
    progress_position: int,
    teacher_experience: str = "",
    shard_index: int = 0,
    shard_count: int = 1,
) -> dict[str, Any]:
    global _PAUSE_REQUESTED, _ACTIVE_PAUSE_FILE
    _PAUSE_REQUESTED = False
    os.environ["CUDA_VISIBLE_DEVICES"] = device
    signal.signal(signal.SIGINT, _handle_pause_signal)

    benchmark = config.get("benchmark_evaluation", {})
    generation = benchmark.get("generation", {})
    if not isinstance(generation, Mapping):
        raise BenchmarkError("benchmark_evaluation.generation must be a mapping")
    if str(generation.get("backend", "vllm")).casefold() != "vllm":
        raise BenchmarkError("the four-step benchmark requires the batched vLLM backend")
    if generation.get("chat_template_kwargs", {}).get("enable_thinking") is not True:
        raise BenchmarkError("benchmark four-step reasoning requires enable_thinking=true")
    thinking_budget = generation.get("thinking_token_budget")
    if thinking_budget is None or int(thinking_budget) < 1:
        raise BenchmarkError("benchmark four-step reasoning requires a positive thinking_token_budget")

    model_path = Path(str(config.get("models", {}).get(role, {}).get("path", ""))).expanduser().resolve()
    metadata = inspect_checkpoint(model_path)
    experience = teacher_experience if role == "teacher" else ""

    def generation_base(config_value: Mapping[str, Any]) -> str:
        return _stable_hash(
            {
                "role": role,
                "model": metadata,
                "prompt": BLIND_REASONING_SYSTEM_PROMPT,
                "teacher_experience_sha256": (
                    hashlib.sha256(experience.encode("utf-8")).hexdigest()
                    if experience
                    else None
                ),
                "generation": _generation_fingerprint(config_value),
            }
        )

    model_base = generation_base(generation)
    compatible_bases: list[str] = []
    for overrides in generation.get("cache_compatible_generation_overrides", ()):
        if not isinstance(overrides, Mapping):
            raise BenchmarkError(
                "cache_compatible_generation_overrides entries must be mappings"
            )
        compatible_generation = dict(generation)
        compatible_generation.update(overrides)
        compatible_bases.append(generation_base(compatible_generation))
    output_root = project_path(
        config,
        str(benchmark.get("output_path", "evaluations/benchmarks")),
    )
    output_root.mkdir(parents=True, exist_ok=True)
    pause_file = project_path(
        config,
        str(benchmark.get("pause_file", "evaluations/benchmarks/.pause")),
    )
    _ACTIVE_PAUSE_FILE = pause_file
    datasets = load_benchmark_datasets(config, dataset_names, limit=limit)
    if shard_count < 1 or not 0 <= shard_index < shard_count:
        raise BenchmarkError(f"invalid benchmark shard {shard_index} of {shard_count}")
    if shard_count > 1:
        datasets = [
            BenchmarkDataset(
                name=dataset.name,
                samples=dataset.samples[shard_index::shard_count],
                sources=dataset.sources,
            )
            for dataset in datasets
        ]
        if any(not dataset.samples for dataset in datasets):
            raise BenchmarkError(
                f"benchmark shard {shard_index} of {shard_count} contains an empty dataset"
            )
    states = [
        _plan_dataset(
            dataset=dataset,
            output_root=output_root,
            role=role,
            base_fingerprint=model_base,
            retry_invalid=retry_invalid,
            compatible_base_fingerprints=compatible_bases,
        )
        for dataset in datasets
    ]

    from tqdm.auto import tqdm

    generator: Any | None = None
    has_pending = any(state.pending for state in states)
    reports: dict[str, Any] = {}
    role_position = progress_position * 2
    sample_position = role_position + 1
    role_progress = tqdm(
        total=len(states) + int(has_pending),
        desc=f"{role}@GPU{device}",
        unit="stage",
        dynamic_ncols=True,
        mininterval=1.0,
        position=role_position,
        leave=True,
        disable=not bool(generation.get("show_progress", True)),
    )
    role_progress.set_postfix_str("planning complete", refresh=True)
    try:
        if has_pending and not _pause_requested(pause_file):
            role_progress.set_postfix_str(f"loading {model_path.name}", refresh=True)
            generator = VllmMultimodalGenerator(
                model_path,
                role=f"benchmark_{role}",
                max_new_tokens=int(generation.get("max_new_tokens", 4096)),
                deterministic=bool(generation.get("deterministic", True)),
                chat_template_kwargs=generation.get("chat_template_kwargs", {}),
                max_model_len=int(generation.get("max_model_len", 32768)),
                max_num_seqs=int(generation.get("max_num_seqs", 16)),
                gpu_memory_utilization=float(generation.get("gpu_memory_utilization", 0.75)),
                thinking_token_budget=int(thinking_budget),
                json_schema=REASONING_JSON_SCHEMA,
                enforce_eager=bool(generation.get("enforce_eager", False)),
            )
            role_progress.update(1)
        if generator is None:
            for state in states:
                role_progress.set_postfix_str(state.dataset.name, refresh=True)
                reports[state.dataset.name] = _evaluate_state(
                    state=state,
                    generator=None,
                    role=role,
                    model_path=model_path,
                    generation=generation,
                    pause_file=pause_file,
                    progress_position=sample_position,
                    experience=experience,
                )
                role_progress.update(1)
        else:
            with generator:
                for state in states:
                    role_progress.set_postfix_str(state.dataset.name, refresh=True)
                    reports[state.dataset.name] = _evaluate_state(
                        state=state,
                        generator=generator,
                        role=role,
                        model_path=model_path,
                        generation=generation,
                        pause_file=pause_file,
                        progress_position=sample_position,
                        experience=experience,
                    )
                    role_progress.update(1)
                    if _pause_requested(pause_file):
                        remaining_states = states[len(reports) :]
                        for remaining in remaining_states:
                            reports[remaining.dataset.name] = _write_progress(
                                remaining,
                                role=role,
                                status="paused",
                            )
                        role_progress.update(len(remaining_states))
                        break
            generator = None
    finally:
        if generator is not None:
            generator.close()
        role_progress.set_postfix_str(
            "paused" if _pause_requested(pause_file) else "done",
            refresh=False,
        )
        role_progress.close()

    statuses = {report["status"] for report in reports.values()}
    if statuses == {"complete"}:
        status = "complete"
    elif "paused" in statuses:
        status = "paused"
    else:
        status = "invalid_outputs"
    pooled = _pooled_metrics(states)
    result = {
        "role": role,
        "status": status,
        "device": device,
        "shard_index": shard_index,
        "shard_count": shard_count,
        "model_path": str(model_path),
        "datasets": reports,
        "pooled": pooled,
        "invalid_outputs": pooled["invalid_json"],
    }
    _atomic_json(output_root / role / "summary.json", result)
    return result



def _merge_role_shards(
    config: Mapping[str, Any],
    *,
    role: str,
    devices: Sequence[str],
    dataset_names: Sequence[str],
    limit: int | None,
    output_root: Path,
    shard_roots: Sequence[Path],
) -> dict[str, Any]:
    datasets = load_benchmark_datasets(config, dataset_names, limit=limit)
    reports: dict[str, Any] = {}
    states: list[_DatasetState] = []
    for dataset in datasets:
        requested_ids = {sample.sample_id for sample in dataset.samples}
        records: dict[str, dict[str, Any]] = {}
        failures: list[dict[str, Any]] = []
        for shard_root in shard_roots:
            shard_output = shard_root / role / dataset.name / "predictions.jsonl"
            shard_records, _ = _load_prediction_records(shard_output)
            for sample_id, record in shard_records.items():
                if sample_id not in requested_ids:
                    continue
                if sample_id in records:
                    raise BenchmarkError(
                        f"duplicate prediction {sample_id!r} while merging data-parallel shards"
                    )
                records[sample_id] = record
            failure_path = shard_root / role / dataset.name / "failures.jsonl"
            if failure_path.exists():
                failures.extend(record for _, record in read_jsonl(failure_path))

        output_dir = output_root / role / dataset.name
        output_path = output_dir / "predictions.jsonl"
        ordered_records = [
            records[sample.sample_id]
            for sample in dataset.samples
            if sample.sample_id in records
        ]
        _atomic_jsonl(output_path, ordered_records)
        journal_path = output_path.with_name(f".{output_path.name}.pending")
        journal_path.unlink(missing_ok=True)
        if failures:
            _atomic_jsonl(output_dir / "failures.jsonl", failures)
        else:
            (output_dir / "failures.jsonl").unlink(missing_ok=True)
        fingerprints = {
            sample.sample_id: str(records.get(sample.sample_id, {}).get("fingerprint", ""))
            for sample in dataset.samples
        }
        state = _DatasetState(
            dataset=dataset,
            output_dir=output_dir,
            output_path=output_path,
            journal_path=journal_path,
            records=records,
            fingerprints=fingerprints,
            pending=[
                (sample, fingerprints[sample.sample_id])
                for sample in dataset.samples
                if sample.sample_id not in records
            ],
        )
        provisional = _dataset_metrics(state, status="running")
        if provisional["completed_samples"] < provisional["dataset_samples"]:
            status = "paused"
        elif provisional["invalid_json"]:
            status = "invalid_outputs"
        else:
            status = "complete"
        reports[dataset.name] = _write_progress(state, role=role, status=status)
        states.append(state)

    statuses = {report["status"] for report in reports.values()}
    if statuses == {"complete"}:
        status = "complete"
    elif "paused" in statuses:
        status = "paused"
    else:
        status = "invalid_outputs"
    pooled = _pooled_metrics(states)
    result = {
        "role": role,
        "status": status,
        "devices": list(devices),
        "data_parallel_shards": len(shard_roots),
        "model_path": str(
            Path(str(config.get("models", {}).get(role, {}).get("path", "")))
            .expanduser()
            .resolve()
        ),
        "datasets": reports,
        "pooled": pooled,
        "invalid_outputs": pooled["invalid_json"],
    }
    _atomic_json(output_root / role / "summary.json", result)
    return result


def _dataset_macro(reports: Mapping[str, Mapping[str, Any]]) -> dict[str, float]:
    values: dict[str, float] = {}
    for key in SUMMARY_METRICS:
        available = [
            float(report[key])
            for report in reports.values()
            if report.get("completed_samples") and key in report
        ]
        values[key] = sum(available) / len(available) if available else 0.0
    return values


def _write_csv(path: Path, role_results: Mapping[str, Mapping[str, Any]]) -> None:
    fields = [
        "role",
        "dataset",
        "status",
        "dataset_samples",
        "completed_samples",
        *SUMMARY_METRICS,
    ]
    rows: list[dict[str, Any]] = []
    for role in ROLE_ORDER:
        if role not in role_results:
            continue
        result = role_results[role]
        for dataset in DATASET_ORDER:
            if dataset not in result["datasets"]:
                continue
            report = result["datasets"][dataset]
            rows.append(
                {
                    "role": role,
                    "dataset": dataset,
                    "status": report["status"],
                    "dataset_samples": report["dataset_samples"],
                    "completed_samples": report["completed_samples"],
                    **{key: report.get(key) for key in SUMMARY_METRICS},
                }
            )
        pooled = result["pooled"]
        rows.append(
            {
                "role": role,
                "dataset": "all_pooled",
                "status": result["status"],
                "dataset_samples": pooled["dataset_samples"],
                "completed_samples": pooled["completed_samples"],
                **{key: pooled.get(key) for key in SUMMARY_METRICS},
            }
        )
        rows.append(
            {
                "role": role,
                "dataset": "dataset_macro",
                "status": result["status"],
                "dataset_samples": "",
                "completed_samples": "",
                **_dataset_macro(result["datasets"]),
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _write_primary_csv(path: Path, role_results: Mapping[str, Mapping[str, Any]]) -> None:
    fields = (
        "role",
        "dataset",
        "dataset_samples",
        "accuracy",
        "precision",
        "recall",
        "macro_f1",
    )
    rows = []
    for role in ROLE_ORDER:
        result = role_results.get(role)
        if result is None:
            continue
        reports = result.get("datasets", {})
        for dataset in DATASET_ORDER:
            report = reports.get(dataset)
            if report is None:
                continue
            rows.append(
                {
                    "role": role,
                    "dataset": dataset,
                    "dataset_samples": report["dataset_samples"],
                    "accuracy": report["accuracy"],
                    "precision": report["precision"],
                    "recall": report["recall"],
                    "macro_f1": report["macro_f1"],
                }
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _final_summary(
    output_root: Path,
    role_results: Mapping[str, Mapping[str, Any]],
    dataset_sizes: Mapping[str, int],
    teacher_experience: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    enriched = {role: dict(result) for role, result in role_results.items()}
    for result in enriched.values():
        result["dataset_macro"] = _dataset_macro(result["datasets"])
    comparison: dict[str, Any] = {}
    if "teacher" in enriched and "student" in enriched:
        for dataset in [*DATASET_ORDER, "all_pooled", "dataset_macro"]:
            if dataset == "all_pooled":
                teacher_metrics = enriched["teacher"]["pooled"]
                student_metrics = enriched["student"]["pooled"]
            elif dataset == "dataset_macro":
                teacher_metrics = enriched["teacher"]["dataset_macro"]
                student_metrics = enriched["student"]["dataset_macro"]
            elif (
                dataset in enriched["teacher"]["datasets"]
                and dataset in enriched["student"]["datasets"]
            ):
                teacher_metrics = enriched["teacher"]["datasets"][dataset]
                student_metrics = enriched["student"]["datasets"][dataset]
            else:
                continue
            comparison[dataset] = {
                key: float(teacher_metrics[key]) - float(student_metrics[key])
                for key in SUMMARY_METRICS
                if key in teacher_metrics and key in student_metrics
            }
    role_statuses = {result["status"] for result in enriched.values()}
    if enriched and role_statuses == {"complete"}:
        status = "complete"
    elif "paused" in role_statuses:
        status = "paused"
    else:
        status = "invalid_outputs"
    invalid_outputs = sum(
        int(result["pooled"]["invalid_json"]) for result in enriched.values()
    )
    summary = {
        "status": status,
        "metrics_final": status == "complete" and invalid_outputs == 0,
        "invalid_outputs": invalid_outputs,
        "reasoning_contract": [
            "visual_evidence",
            "textual_evidence",
            "explanation",
            "label",
        ],
        "thinking_mode": True,
        "teacher_experience": dict(teacher_experience) if teacher_experience else None,
        "dataset_sizes": dict(dataset_sizes),
        "roles": enriched,
        "teacher_minus_student": comparison,
        "resume": {
            "automatic": True,
            "journal_granularity": "completed_batch",
            "pause_file": str(output_root / ".pause"),
        },
    }
    _atomic_json(output_root / "summary.json", summary)
    _write_csv(output_root / "summary.csv", enriched)
    _write_primary_csv(output_root / "primary_metrics.csv", enriched)
    return summary


def run_benchmark(
    config: Mapping[str, Any],
    *,
    devices: Sequence[str],
    roles: Sequence[str] = ROLE_ORDER,
    datasets: Sequence[str] = DATASET_ORDER,
    limit: int | None = None,
    retry_invalid: bool = False,
    teacher_experience_pool: str | Path | None = None,
    expected_active_experiences: int | None = None,
) -> dict[str, Any]:
    global _PAUSE_REQUESTED, _ACTIVE_PAUSE_FILE
    benchmark = config.get("benchmark_evaluation", {})
    retry_invalid = bool(
        retry_invalid or benchmark.get("retry_invalid_by_default", True)
    )
    selected_roles = list(dict.fromkeys(roles))
    if not selected_roles or any(role not in ROLE_ORDER for role in selected_roles):
        raise BenchmarkError(f"roles must be chosen from {list(ROLE_ORDER)}")
    teacher_experience = ""
    teacher_experience_metadata: dict[str, Any] | None = None
    if teacher_experience_pool is not None:
        if selected_roles != ["teacher"]:
            raise BenchmarkError(
                "teacher experience benchmark must select only --roles teacher"
            )
        teacher_experience, teacher_experience_metadata = _load_teacher_experience_context(
            config,
            teacher_experience_pool,
            expected_active=expected_active_experiences,
        )
    elif expected_active_experiences is not None:
        raise BenchmarkError(
            "--expected-active-experiences requires --teacher-experience-pool"
        )

    selected_devices = [str(device).strip() for device in devices if str(device).strip()]
    if not selected_devices:
        raise BenchmarkError("at least one CUDA device is required")
    loaded = load_benchmark_datasets(config, datasets, limit=limit)
    dataset_names = [dataset.name for dataset in loaded]
    dataset_sizes = {dataset.name: len(dataset.samples) for dataset in loaded}

    output_root = project_path(
        config,
        str(benchmark.get("output_path", "evaluations/benchmarks")),
    )
    pause_file = project_path(
        config,
        str(benchmark.get("pause_file", "evaluations/benchmarks/.pause")),
    )
    output_root.mkdir(parents=True, exist_ok=True)
    pause_file.unlink(missing_ok=True)
    _PAUSE_REQUESTED = False
    _ACTIVE_PAUSE_FILE = pause_file

    role_results: dict[str, Mapping[str, Any]] = {}
    unique_devices = list(dict.fromkeys(selected_devices))
    data_parallel = (
        bool(teacher_experience)
        and selected_roles == ["teacher"]
        and len(unique_devices) > 1
    )
    parallel = len(selected_roles) > 1 and len(unique_devices) >= len(selected_roles)
    if data_parallel:
        role = "teacher"
        shard_count = min(len(unique_devices), min(dataset_sizes.values()))
        shard_devices = unique_devices[:shard_count]
        shard_roots = [
            output_root / ".shards" / role / f"shard-{index:02d}-of-{shard_count:02d}"
            for index in range(shard_count)
        ]
        context = multiprocessing.get_context("spawn")
        executor = ProcessPoolExecutor(max_workers=shard_count, mp_context=context)
        previous_handler = signal.signal(signal.SIGINT, _handle_pause_signal)
        try:
            futures = {}
            for index, (device, shard_root) in enumerate(
                zip(shard_devices, shard_roots, strict=True)
            ):
                shard_config = copy.deepcopy(config)
                shard_config["benchmark_evaluation"]["output_path"] = str(shard_root)
                future = executor.submit(
                    _role_worker,
                    shard_config,
                    role,
                    device,
                    dataset_names,
                    limit,
                    retry_invalid,
                    index,
                    teacher_experience,
                    index,
                    shard_count,
                )
                futures[future] = index
            for future in as_completed(futures):
                future.result()
        finally:
            executor.shutdown(wait=True, cancel_futures=False)
            signal.signal(signal.SIGINT, previous_handler)
        role_results[role] = _merge_role_shards(
            config,
            role=role,
            devices=shard_devices,
            dataset_names=dataset_names,
            limit=limit,
            output_root=output_root,
            shard_roots=shard_roots,
        )
    elif parallel:
        context = multiprocessing.get_context("spawn")
        executor = ProcessPoolExecutor(max_workers=len(selected_roles), mp_context=context)
        previous_handler = signal.signal(signal.SIGINT, _handle_pause_signal)
        try:
            futures = {
                executor.submit(
                    _role_worker,
                    config,
                    role,
                    selected_devices[index],
                    dataset_names,
                    limit,
                    retry_invalid,
                    index,
                    teacher_experience if role == "teacher" else "",
                ): role
                for index, role in enumerate(selected_roles)
            }
            for future in as_completed(futures):
                role = futures[future]
                role_results[role] = future.result()
        finally:
            executor.shutdown(wait=True, cancel_futures=False)
            signal.signal(signal.SIGINT, previous_handler)
    else:
        for index, role in enumerate(selected_roles):
            device = selected_devices[index % len(selected_devices)]
            role_results[role] = _role_worker(
                config,
                role,
                device,
                dataset_names,
                limit,
                retry_invalid,
                0,
                teacher_experience if role == "teacher" else "",
            )
            if role_results[role]["status"] == "paused":
                break

    return _final_summary(
        output_root,
        role_results,
        dataset_sizes,
        teacher_experience=teacher_experience_metadata,
    )
