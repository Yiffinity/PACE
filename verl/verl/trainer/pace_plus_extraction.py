"""Resumable comparative experience extraction for PACE_PLUS Stage A."""

from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from verl.trainer.pace_plus_config import get_required, project_path
from verl.trainer.pace_plus_data import normalize_record
from verl.trainer.pace_plus_modeling import MultimodalGenerator, VllmMultimodalGenerator
from verl.trainer.pace_plus_preflight import validate_model_pair
from verl.trainer.pace_plus_prompts import (
    BLIND_REASONING_SYSTEM_PROMPT,
    TEACHER_LABEL_CORRECTION_SYSTEM_PROMPT,
    blind_messages,
    teacher_label_correction_messages,
)
from verl.trainer.pace_plus_reflection import (
    LEGACY_REFLECTION_CACHE_IDENTITIES,
    PROMPT_VERSION,
    ReflectionExperience,
    ReflectionPromptDocument,
    build_comparative_reflection_messages,
    build_single_source_reflection_messages,
    embedded_reflection_prompt_document,
    parse_reflection,
    repair_reflection_sentence_format,
)
from verl.trainer.pace_plus_schema import (
    Reasoning,
    SarcasmSample,
    SchemaError,
    append_jsonl,
    generate_json_with_retries,
    load_records_by_id,
    parse_reasoning,
    read_jsonl,
    upsert_jsonl,
    validate_reasoning,
)


class ExtractionError(RuntimeError):
    pass


def _hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=True, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def _file_hash(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sample_input_fingerprint(sample: SarcasmSample) -> dict[str, str]:
    return {
        "sample_id": sample.sample_id,
        "text_sha256": hashlib.sha256(sample.text.encode("utf-8")).hexdigest(),
        "image_sha256": _file_hash(sample.image_path),
        "metadata_sha256": _hash(dict(sample.extraction_metadata)),
        "source": sample.source or "",
        "language": sample.language or "",
    }



def _atomic_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n")
        os.replace(temp_name, path)
    except BaseException:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def _load_samples(data_path: str | Path, *, limit: int | None) -> list[SarcasmSample]:
    source = Path(data_path).expanduser().resolve()
    samples: list[SarcasmSample] = []
    seen: set[str] = set()
    for line_number, record in read_jsonl(source):
        try:
            sample = normalize_record(record, base_dir=source.parent, require_image=True)
        except SchemaError as exc:
            raise ExtractionError(f"{source}:{line_number}: {exc}") from exc
        if sample.sample_id in seen:
            raise ExtractionError(f"{source}:{line_number}: duplicate sample_id {sample.sample_id!r}")
        seen.add(sample.sample_id)
        samples.append(sample)
        if limit is not None and len(samples) >= limit:
            break
    if not samples:
        raise ExtractionError(f"no valid samples found in {source}")
    return samples


def _reasoning_from_record(record: Mapping[str, Any], *, role: str, gold_label: str | None = None) -> Reasoning:
    value = record.get("analysis", record.get("reference_reasoning", record))
    parsed = validate_reasoning(value)
    if gold_label is not None and parsed.label != gold_label:
        raise ExtractionError(f"{role} reasoning label {parsed.label!r} does not match gold label {gold_label!r}")
    return parsed


def _parse_reasoning_for_sample(
    raw: str,
    sample: SarcasmSample,
    *,
    enforce_gold_label: bool,
) -> Reasoning:
    parsed = parse_reasoning(raw)
    if enforce_gold_label and parsed.label != sample.label:
        raise SchemaError(
            f"reasoning label {parsed.label!r} does not match gold label {sample.label!r}"
        )
    return parsed

def _load_prompt_document(config: Mapping[str, Any]) -> ReflectionPromptDocument:
    del config
    return embedded_reflection_prompt_document()

def _valid_cached(
    record: Mapping[str, Any] | None,
    fingerprint: str,
    *,
    reasoning: bool = False,
    reflection: bool = False,
) -> bool:
    if not record or record.get("fingerprint") != fingerprint or record.get("error"):
        return False
    try:
        if reasoning:
            validate_reasoning(record.get("analysis"))
        if reflection:
            parse_reflection(json.dumps({"experiences": record.get("experiences")}))
        return True
    except SchemaError:
        return False


def _reasoning_journal_path(output_path: Path) -> Path:
    return output_path.with_name(f".{output_path.name}.pending")


def _load_reasoning_records(output_path: Path) -> tuple[dict[str, dict[str, Any]], Path]:
    existing = load_records_by_id(output_path, allow_missing=True)
    journal_path = _reasoning_journal_path(output_path)
    if journal_path.exists():
        for line_number, record in read_jsonl(journal_path):
            sample_id = record.get("sample_id")
            if not isinstance(sample_id, str) or not sample_id:
                raise ExtractionError(f"{journal_path}:{line_number}: missing sample_id")
            existing[sample_id] = record
    return existing, journal_path


def _append_reasoning_records(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n")
        handle.flush()


def _compact_reasoning_records(
    output_path: Path,
    journal_path: Path,
    records: Mapping[str, Mapping[str, Any]],
) -> None:
    _atomic_jsonl(output_path, records.values())
    journal_path.unlink(missing_ok=True)


def _chunks(values: list[Any], size: int) -> Iterable[list[Any]]:
    for offset in range(0, len(values), size):
        yield values[offset : offset + size]


def _generate_batch_with_isolation(
    generator: Any,
    requests: list[tuple[list[dict[str, Any]], str]],
) -> tuple[list[str | None], list[str | None]]:
    """Isolate request-local ValueErrors without penalizing the rest of a batch."""
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
            raise ExtractionError(
                f"batch generator returned {len(generated)} outputs for {len(indices)} requests"
            )
        for index, output in zip(indices, generated, strict=True):
            outputs[index] = output

    if requests:
        generate(list(range(len(requests))))
    return outputs, errors


def _reasoning_generation_fingerprint(config: Mapping[str, Any]) -> dict[str, Any]:
    content_keys = (
        "backend",
        "deterministic",
        "max_new_tokens",
        "thinking_token_budget",
        "max_model_len",
        "chat_template_kwargs",
        "_reasoning_json_schema",
    )
    return {
        **{key: config.get(key) for key in content_keys},
        "json_max_attempts": config.get("json", {}).get("max_attempts", 3),
    }


def _generate_reasoning_batch(
    *,
    generator: Any,
    batch: list[tuple[SarcasmSample, str]],
    role: str,
    model_path: str,
    failure_log_path: Path,
    max_attempts: int,
    messages_builder: Callable[[SarcasmSample], list[dict[str, Any]]] = lambda sample: blind_messages(sample.text),
    enforce_gold_label: bool = False,
) -> list[dict[str, Any]]:
    states: list[dict[str, Any]] = [
        {
            "sample": sample,
            "fingerprint": fingerprint,
            "messages": messages_builder(sample),
            "raw": None,
            "error": None,
            "attempts": [],
            "parsed": None,
        }
        for sample, fingerprint in batch
    ]
    remaining = list(range(len(states)))
    for attempt in range(1, int(max_attempts) + 1):
        requests = [
            (states[index]["messages"], states[index]["sample"].image_path)
            for index in remaining
        ]
        try:
            outputs, generation_errors = _generate_batch_with_isolation(
                generator, requests
            )
        except Exception as exc:
            outputs = [None] * len(remaining)
            generation_errors = [
                f"generation failed: {type(exc).__name__}: {exc}"
            ] * len(remaining)

        retry: list[int] = []
        for index, raw, generation_error in zip(
            remaining, outputs, generation_errors, strict=True
        ):
            state = states[index]
            state["raw"] = raw
            if generation_error is not None:
                error = generation_error
            else:
                try:
                    state["parsed"] = _parse_reasoning_for_sample(
                        raw,
                        state["sample"],
                        enforce_gold_label=enforce_gold_label,
                    )
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
                else:
                    state["error"] = None
                    state["attempt_count"] = attempt
                    continue
            state["error"] = error
            state["attempts"].append(
                {"attempt": attempt, "error": error, "raw_output": raw}
            )
            state["attempt_count"] = attempt
            prompt_too_long = (
                generation_error is not None and "maximum model length" in generation_error
            )
            if attempt < int(max_attempts) and not prompt_too_long:
                state["messages"] = [
                    *state["messages"],
                    {
                        "role": "user",
                        "content": (
                            "Your previous response did not satisfy the required JSON contract. "
                            "Return exactly one valid JSON object, with no Markdown, prose, or extra fields. "
                            "Use double-quoted JSON strings and do not backslash-escape apostrophes. "
                            "Keep each evidence field under 40 words and the explanation under 80 words. "
                            f"Validation error: {error}"
                        ),
                    },
                ]
                retry.append(index)
        remaining = retry
        if not remaining:
            break

    resolved_model = str(Path(model_path).resolve())
    records: list[dict[str, Any]] = []
    for state in states:
        sample = state["sample"]
        parsed = state["parsed"]
        if parsed is None:
            append_jsonl(
                failure_log_path,
                {
                    "stage": "experience_extraction",
                    "artifact": f"{role}_reasoning",
                    "sample_id": sample.sample_id,
                    "model_path": resolved_model,
                    "max_attempts": int(max_attempts),
                    "attempts": state["attempts"],
                    "error": state["error"] or "invalid JSON output",
                },
            )
        records.append(
            {
                "sample_id": sample.sample_id,
                "role": role,
                "analysis": parsed.to_dict() if parsed else None,
                "raw_output": state["raw"],
                "error": None if parsed else state["error"],
                "attempts": state["attempt_count"],
                "json_valid": parsed is not None,
                "model_path": resolved_model,
                "fingerprint": state["fingerprint"],
                "origin": "generated",
            }
        )
    return records


REFLECTION_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "experiences": {
            "type": "array",
            "minItems": 1,
            "maxItems": 3,
            "items": {
                "type": "object",
                "properties": {
                    "target_models": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 1,
                        "items": {"enum": ["mllm_b"]},
                    },
                    "content": {"type": "string", "minLength": 1},
                },
                "required": ["target_models", "content"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["experiences"],
    "additionalProperties": False,
}


def _reflection_messages(
    mode: str,
    document: ReflectionPromptDocument | None,
    sample: SarcasmSample,
    teacher: Reasoning,
    initial_teacher: Reasoning,
    student: Reasoning,
    teacher_origin: str,
) -> list[dict[str, Any]]:
    if mode == "comparative":
        if document is None:
            raise ExtractionError("comparative reflection prompt was not loaded")
        return build_comparative_reflection_messages(
            document,
            sample,
            teacher,
            student,
            teacher_origin,
            initial_teacher=initial_teacher,
        )
    if mode == "teacher_only":
        return build_single_source_reflection_messages(sample, teacher, target_model="mllm_b")
    return build_single_source_reflection_messages(sample, student, target_model="mllm_b")


def _parse_reflection_for_teacher_origin(
    raw: str,
    teacher_origin: str,
) -> tuple[ReflectionExperience, ...]:
    del teacher_origin
    try:
        return parse_reflection(raw)
    except SchemaError as exc:
        if "exactly two or three English sentences" not in str(exc):
            raise
        repaired = repair_reflection_sentence_format(raw)
        if repaired == raw:
            raise
        return parse_reflection(repaired)


def _legacy_reflection_schema() -> dict[str, Any]:
    schema = json.loads(json.dumps(REFLECTION_JSON_SCHEMA))
    schema["properties"]["experiences"].pop("minItems", None)
    return schema


def _reflection_generation_fingerprint(
    generation: Mapping[str, Any],
) -> dict[str, Any]:
    # Context capacity does not change a successful deterministic response.
    return {
        key: value
        for key, value in generation.items()
        if key != "reflection_max_model_len"
    }


def _reflection_max_model_len(generation: Mapping[str, Any]) -> int:
    return int(
        generation.get(
            "reflection_max_model_len",
            generation.get("max_model_len", 6144),
        )
    )


def _generate_reflection_batch(
    *,
    generator: Any,
    batch: list[tuple[SarcasmSample, Reasoning, Reasoning, Reasoning, str, str]],
    mode: str,
    document: ReflectionPromptDocument | None,
    failure_log_path: Path,
    max_attempts: int,
    model_path: str,
) -> list[dict[str, Any]]:
    states: list[dict[str, Any]] = []
    for pending_item in batch:
        sample, teacher, initial_teacher, student, teacher_origin, _ = pending_item
        states.append(
            {
                "pending_item": pending_item,
                "messages": _reflection_messages(
                    mode,
                    document,
                    sample,
                    teacher,
                    initial_teacher,
                    student,
                    teacher_origin,
                ),
                "raw": None,
                "error": None,
                "attempt_count": 0,
                "attempts": [],
                "parsed": None,
            }
        )

    remaining = list(range(len(states)))
    for attempt in range(1, int(max_attempts) + 1):
        requests = [
            (states[index]["messages"], states[index]["pending_item"][0].image_path)
            for index in remaining
        ]
        try:
            outputs, generation_errors = _generate_batch_with_isolation(
                generator, requests
            )
        except Exception as exc:
            error_text = str(exc).casefold()
            if "device-side assert" in error_text or "illegal memory access" in error_text:
                raise
            outputs = [None] * len(remaining)
            generation_errors = [
                f"generation failed: {type(exc).__name__}: {exc}"
            ] * len(remaining)

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
                    teacher_origin = state["pending_item"][4]
                    state["parsed"] = _parse_reflection_for_teacher_origin(
                        raw,
                        teacher_origin,
                    )
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
                else:
                    state["error"] = None
                    continue
            state["error"] = error
            state["attempts"].append({"attempt": attempt, "error": error, "raw_output": raw})
            prompt_too_long = (
                generation_error is not None and "maximum model length" in generation_error
            )
            if attempt < int(max_attempts) and not prompt_too_long:
                correction = (
                    "Return exactly one valid JSON object, with no Markdown, prose, thinking block, "
                    "or extra fields. "
                )
                if "one to three items" in error:
                    sample = state["pending_item"][0]
                    student = state["pending_item"][3]
                    correction += "Return exactly one source-agnostic experience; an empty array is forbidden. "
                    if teacher_origin in {"codex_accepted", "codex_revised"}:
                        correction += (
                            "Compare the initial incorrect teacher reasoning with the verified corrected "
                            "reasoning and state the transferable correction. "
                        )
                    elif student.label == sample.label:
                        correction += (
                            "Because both predictions are correct, state their strongest jointly supported "
                            "transferable decision rule and nearest exclusion boundary without inventing a gap. "
                        )
                    else:
                        correction += (
                            "Because the student prediction is incorrect, state the first unsupported inference "
                            "or missed decisive evidence and the verification step that prevents it. "
                        )
                elif "exactly two or three English sentences" in error:
                    correction += (
                        "Return exactly one experience item whose content is exactly two short plain-ASCII "
                        "English sentences, with each sentence ending in a period. Do not quote or repeat "
                        "source text, names, entities, slogans, or examples. "
                    )
                elif "must be in English" in error:
                    correction += (
                        "Return exactly one experience item whose content is exactly two short plain-ASCII "
                        "English sentences. Do not copy, transliterate, translate, or mention any quoted "
                        "non-English source phrase; describe only its general semantic function in English. "
                        "The content value must contain ASCII characters only. "
                    )
                elif "forbidden sample or supervision identifiers" in error:
                    correction += (
                        "Remove all names, entities, quoted sample phrases, dataset terms, model identifiers, "
                        "annotation terms, placeholders, coordinates, UUIDs, and other sample-specific details. "
                        "Do not use angle brackets or placeholder words such as user. Rewrite the rule from "
                        "observable evidence types and their relationship only. "
                    )
                elif "must end with sentence punctuation" in error:
                    correction += (
                        "Return exactly one experience item whose content is exactly two short plain-ASCII "
                        "English sentences, with each sentence ending in a period. Do not use quotation "
                        "marks or repeat source text. "
                    )
                else:
                    correction += "Satisfy every field and content constraint in the system prompt. "
                retry_assistant_content = (
                    raw or ""
                    if generation_error is not None
                    else "I will discard the invalid content and return a fresh, corrected JSON object."
                )
                state["messages"] = [
                    *state["messages"],
                    {"role": "assistant", "content": retry_assistant_content},
                    {
                        "role": "user",
                        "content": (
                            "Your previous response did not satisfy the required JSON contract. "
                            f"{correction}"
                            f"Validation error: {error}"
                        ),
                    },
                ]
                retry.append(index)
        remaining = retry
        if not remaining:
            break

    resolved_model = str(Path(model_path).resolve())
    for state in states:
        if state["parsed"] is not None:
            continue
        sample = state["pending_item"][0]
        append_jsonl(
            failure_log_path,
            {
                "stage": "experience_extraction",
                "artifact": "comparative_experience",
                "sample_id": sample.sample_id,
                "model_path": resolved_model,
                "max_attempts": int(max_attempts),
                "attempts": state["attempts"],
                "error": state["error"] or "invalid JSON output",
            },
        )
    return states


def _generate_reasonings(
    *,
    role: str,
    samples: list[SarcasmSample],
    model_path: str,
    output_path: Path,
    precomputed_path: str | None,
    generation_config: Mapping[str, Any],
    failure_log_path: Path,
    json_max_attempts: int,
    messages_builder: Callable[[SarcasmSample], list[dict[str, Any]]] | None = None,
    prompt_fingerprint: str = BLIND_REASONING_SYSTEM_PROMPT,
    enforce_gold_label: bool = False,
    fingerprint_context: Mapping[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    from tqdm.auto import tqdm
    messages_builder = messages_builder or (lambda sample: blind_messages(sample.text))

    existing, journal_path = _load_reasoning_records(output_path)
    precomputed = (
        load_records_by_id(precomputed_path, allow_missing=True)
        if precomputed_path
        else {}
    )
    model_role = role if role in {"teacher", "student"} else "teacher"
    model_meta = validate_model_pair(model_path, model_path)[model_role]

    def base_fingerprint(config_value: Mapping[str, Any]) -> str:
        return _hash(
            {
                "role": role,
                "model": model_meta,
                "prompt": prompt_fingerprint,
                "generation": _reasoning_generation_fingerprint(config_value),
            }
        )

    current_base_fingerprint = base_fingerprint(generation_config)
    compatible_base_fingerprints: list[str] = []
    compatible_overrides = generation_config.get(
        "cache_compatible_generation_overrides", ()
    )
    for overrides in compatible_overrides:
        if not isinstance(overrides, Mapping):
            raise ExtractionError(
                "cache_compatible_generation_overrides entries must be mappings"
            )
        compatible_config = dict(generation_config)
        compatible_config.update(overrides)
        compatible_base_fingerprints.append(base_fingerprint(compatible_config))
    pending: list[tuple[SarcasmSample, str]] = []
    precomputed_updates: list[dict[str, Any]] = []
    for sample in samples:
        source_record = precomputed.get(sample.sample_id)
        fingerprint_input = {
            "sample_input": _sample_input_fingerprint(sample),
            "precomputed": source_record,
            "context": (fingerprint_context or {}).get(sample.sample_id),
        }
        fingerprint = _hash({"base": current_base_fingerprint, **fingerprint_input})
        cached = existing.get(sample.sample_id)
        if _valid_cached(cached, fingerprint, reasoning=True):
            continue
        cache_migrated = False
        for compatible_base_fingerprint in compatible_base_fingerprints:
            compatible_fingerprint = _hash(
                {"base": compatible_base_fingerprint, **fingerprint_input}
            )
            if _valid_cached(cached, compatible_fingerprint, reasoning=True):
                migrated = {**cached, "fingerprint": fingerprint}
                existing[sample.sample_id] = migrated
                precomputed_updates.append(migrated)
                cache_migrated = True
                break
        if cache_migrated:
            continue
        if source_record is not None:
            try:
                analysis = _reasoning_from_record(source_record, role=role)
            except (SchemaError, ExtractionError) as exc:
                append_jsonl(
                    failure_log_path,
                    {
                        "stage": "experience_extraction",
                        "artifact": f"{role}_reasoning_precomputed",
                        "sample_id": sample.sample_id,
                        "error": f"{type(exc).__name__}: {exc}",
                        "raw_output": source_record.get("raw_content") or source_record.get("analysis"),
                        "attempts": 0,
                    },
                )
                pending.append((sample, fingerprint))
            else:
                record = {
                    "sample_id": sample.sample_id,
                    "role": role,
                    "analysis": analysis.to_dict(),
                    "raw_output": source_record.get("raw_content") or json.dumps(analysis.to_dict(), ensure_ascii=True),
                    "model_path": str(Path(model_path).resolve()),
                    "fingerprint": fingerprint,
                    "origin": "precomputed",
                    "json_valid": True,
                    "attempts": 0,
                }
                precomputed_updates.append(record)
                existing[sample.sample_id] = record
        else:
            pending.append((sample, fingerprint))
    if precomputed_updates:
        _append_reasoning_records(journal_path, precomputed_updates)

    progress = tqdm(
        total=len(samples),
        initial=len(samples) - len(pending),
        desc=f"{role.capitalize()} reasonings",
        unit="sample",
        dynamic_ncols=True,
        mininterval=1.0,
        position=int(generation_config.get("progress_position", 0)),
        disable=not bool(generation_config.get("show_progress", True)),
    )
    valid_total = len(samples) - len(pending)
    try:
        if pending:
            backend = str(generation_config.get("backend", "transformers")).lower()
            if backend == "vllm":
                json_schema = generation_config.get("_reasoning_json_schema")
                if not isinstance(json_schema, Mapping):
                    raise ExtractionError("vLLM reasoning generation requires a resolved JSON schema")
                generator: Any = VllmMultimodalGenerator(
                    model_path,
                    role=role,
                    max_new_tokens=int(generation_config.get("max_new_tokens", 1536)),
                    deterministic=bool(generation_config.get("deterministic", True)),
                    chat_template_kwargs=generation_config.get("chat_template_kwargs", {}),
                    max_model_len=int(generation_config.get("max_model_len", 6144)),
                    max_num_seqs=int(generation_config.get("max_num_seqs", 8)),
                    gpu_memory_utilization=float(generation_config.get("gpu_memory_utilization", 0.75)),
                    thinking_token_budget=int(generation_config.get("thinking_token_budget", 1024)),
                    json_schema=json_schema,
                    enforce_eager=bool(generation_config.get("enforce_eager", True)),
                )
                batch_size = int(generation_config.get("batch_size", 32))
            elif backend == "transformers":
                generator = MultimodalGenerator(
                    model_path,
                    role=role,
                    device_map=generation_config.get("device_map", "auto"),
                    max_new_tokens=int(generation_config.get("max_new_tokens", 768)),
                    deterministic=bool(generation_config.get("deterministic", True)),
                    chat_template_kwargs=generation_config.get("chat_template_kwargs", {}),
                )
                batch_size = 1
            else:
                raise ExtractionError(f"unsupported reasoning generation backend: {backend!r}")
            if batch_size < 1:
                raise ExtractionError("experience_extraction.generation.batch_size must be positive")

            with generator:
                for batch in _chunks(pending, batch_size):
                    if hasattr(generator, "generate_batch"):
                        records = _generate_reasoning_batch(
                            generator=generator,
                            batch=batch,
                            role=role,
                            model_path=model_path,
                            failure_log_path=failure_log_path,
                            max_attempts=json_max_attempts,
                            messages_builder=messages_builder,
                            enforce_gold_label=enforce_gold_label,
                        )
                    else:
                        records = []
                        for sample, fingerprint in batch:
                            parsed, raw, error, attempts = generate_json_with_retries(
                                generator,
                                messages_builder(sample),
                                image_path=sample.image_path,
                                parser=lambda raw, current=sample: _parse_reasoning_for_sample(
                                    raw,
                                    current,
                                    enforce_gold_label=enforce_gold_label,
                                ),
                                max_attempts=json_max_attempts,
                                failure_log_path=failure_log_path,
                                context={
                                    "stage": "experience_extraction",
                                    "artifact": f"{role}_reasoning",
                                    "sample_id": sample.sample_id,
                                    "model_path": str(Path(model_path).resolve()),
                                },
                            )
                            records.append(
                                {
                                    "sample_id": sample.sample_id,
                                    "role": role,
                                    "analysis": parsed.to_dict() if parsed else None,
                                    "raw_output": raw,
                                    "error": error,
                                    "attempts": attempts,
                                    "json_valid": parsed is not None,
                                    "model_path": str(Path(model_path).resolve()),
                                    "fingerprint": fingerprint,
                                    "origin": "generated",
                                }
                            )
                    _append_reasoning_records(journal_path, records)
                    for record in records:
                        existing[record["sample_id"]] = record
                    valid_total += sum(record["json_valid"] for record in records)
                    progress.set_postfix(
                        valid=f"{valid_total}/{progress.n + len(records)}",
                        refresh=False,
                    )
                    progress.update(len(records))
    finally:
        progress.close()
    _compact_reasoning_records(output_path, journal_path, existing)
    return existing

def _prepare_verified_teacher_reasonings(
    config: Mapping[str, Any],
    samples: list[SarcasmSample],
    teacher_records: Mapping[str, Mapping[str, Any]],
    *,
    output_dir: Path,
    generation: Mapping[str, Any],
    failure_log_path: Path,
    json_max_attempts: int,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    teacher_path = str(get_required(config, "models.teacher.path"))
    correction_path = output_dir / "teacher_label_conditioned_reasonings.jsonl"
    review_queue_path = output_dir / "teacher_correction_review_queue.jsonl"
    reviews_path = output_dir / "teacher_correction_reviews.jsonl"
    verified_path = output_dir / "teacher_reasonings_for_reflection.jsonl"

    initial: dict[str, Reasoning] = {}
    invalid_initial = 0
    wrong_samples: list[SarcasmSample] = []
    for sample in samples:
        try:
            reasoning = _reasoning_from_record(
                teacher_records[sample.sample_id],
                role="teacher",
            )
        except (KeyError, SchemaError, ExtractionError):
            invalid_initial += 1
            continue
        initial[sample.sample_id] = reasoning
        if reasoning.label != sample.label:
            wrong_samples.append(sample)

    correction_context = {
        sample.sample_id: {
            "gold_label": sample.label,
            "initial_teacher_reasoning": initial[sample.sample_id].to_dict(),
        }
        for sample in wrong_samples
    }
    corrections = _generate_reasonings(
        role="teacher_correction",
        samples=wrong_samples,
        model_path=teacher_path,
        output_path=correction_path,
        precomputed_path=None,
        generation_config={**dict(generation), "progress_position": 0},
        failure_log_path=failure_log_path,
        json_max_attempts=json_max_attempts,
        messages_builder=lambda sample: teacher_label_correction_messages(
            sample.text,
            sample.label,
            initial[sample.sample_id].to_dict(),
        ),
        prompt_fingerprint=TEACHER_LABEL_CORRECTION_SYSTEM_PROMPT,
        enforce_gold_label=True,
        fingerprint_context=correction_context,
    )
    reviews = load_records_by_id(reviews_path, allow_missing=True)

    verified: dict[str, dict[str, Any]] = {}
    queue: list[dict[str, Any]] = []
    correction_invalid = 0
    review_pending = 0
    review_revised = 0
    review_accepted = 0
    for sample in samples:
        blind = initial.get(sample.sample_id)
        if blind is None:
            continue
        if blind.label == sample.label:
            verified[sample.sample_id] = {
                "sample_id": sample.sample_id,
                "gold_label": sample.label,
                "analysis": blind.to_dict(),
                "origin": "blind_correct",
            }
            continue

        try:
            corrected = _reasoning_from_record(
                corrections[sample.sample_id],
                role="teacher_correction",
                gold_label=sample.label,
            )
        except (KeyError, SchemaError, ExtractionError):
            correction_invalid += 1
            continue
        correction_record = corrections[sample.sample_id]
        correction_fingerprint = correction_record.get("fingerprint")
        if not isinstance(correction_fingerprint, str) or not correction_fingerprint:
            correction_invalid += 1
            continue
        review = reviews.get(sample.sample_id)
        review_status = "PENDING"
        selected_reasoning: Reasoning | None = None
        selected_origin: str | None = None
        if review is not None:
            required = {
                "sample_id",
                "correction_fingerprint",
                "decision",
                "revised_reasoning",
                "rationale",
            }
            if set(review) != required:
                raise ExtractionError(
                    f"Codex review for {sample.sample_id!r} must contain exactly "
                    f"{sorted(required)}"
                )
            if review.get("correction_fingerprint") != correction_fingerprint:
                review_status = "STALE"
                review_pending += 1
            else:
                decision = review.get("decision")
                rationale = review.get("rationale")
                if decision not in {"ACCEPT", "REVISE"}:
                    raise ExtractionError(
                        f"Codex review for {sample.sample_id!r} has invalid decision {decision!r}"
                    )
                if not isinstance(rationale, str) or not rationale.strip():
                    raise ExtractionError(
                        f"Codex review for {sample.sample_id!r} requires a rationale"
                    )
                revised = review.get("revised_reasoning")
                if decision == "ACCEPT":
                    if revised is not None:
                        raise ExtractionError(
                            f"ACCEPT review for {sample.sample_id!r} requires revised_reasoning=null"
                        )
                    selected_reasoning = corrected
                    selected_origin = "codex_accepted"
                    review_accepted += 1
                else:
                    selected_reasoning = _reasoning_from_record(
                        {"analysis": revised},
                        role="codex_revised_teacher",
                        gold_label=sample.label,
                    )
                    selected_origin = "codex_revised"
                    review_revised += 1
                review_status = decision
        else:
            review_pending += 1

        queue.append(
            {
                "sample_id": sample.sample_id,
                "image_path": sample.image_path,
                "text": sample.text,
                "gold_label": sample.label,
                "initial_teacher_reasoning": blind.to_dict(),
                "corrected_teacher_reasoning": corrected.to_dict(),
                "correction_fingerprint": correction_fingerprint,
                "review_status": review_status,
            }
        )
        if selected_reasoning is not None and selected_origin is not None:
            verified[sample.sample_id] = {
                "sample_id": sample.sample_id,
                "gold_label": sample.label,
                "analysis": selected_reasoning.to_dict(),
                "origin": selected_origin,
            }

    review_queue_ready = invalid_initial == 0 and correction_invalid == 0
    _atomic_jsonl(review_queue_path, queue if review_queue_ready else ())
    _atomic_jsonl(
        verified_path,
        (verified[sample.sample_id] for sample in samples if sample.sample_id in verified),
    )
    return verified, {
        "teacher_initial_invalid": invalid_initial,
        "teacher_initial_incorrect": len(wrong_samples),
        "teacher_correction_invalid": correction_invalid,
        "teacher_review_pending": review_pending,
        "teacher_review_accepted": review_accepted,
        "teacher_review_revised": review_revised,
        "teacher_review_queue_ready": review_queue_ready,
        "teacher_verified": len(verified),
        "teacher_review_queue": str(review_queue_path),
        "teacher_reviews": str(reviews_path),
        "teacher_verified_output": str(verified_path),
    }



def _materialize_references(
    samples: list[SarcasmSample], *, source_path: str | None, output_path: Path
) -> dict[str, dict[str, Any]]:
    source = load_records_by_id(source_path, allow_missing=True) if source_path else {}
    existing = load_records_by_id(output_path, allow_missing=True)
    materialized: dict[str, dict[str, Any]] = {}
    for sample in samples:
        if sample.sample_id in source:
            analysis = _reasoning_from_record(source[sample.sample_id], role="reference", gold_label=sample.label)
            origin = "reference_artifact"
        elif sample.reference_reasoning is not None:
            analysis = sample.reference_reasoning
            origin = "sample_fallback"
        else:
            raise ExtractionError(
                f"sample {sample.sample_id!r} has no ground-truth reference reasoning; "
                "reference generation is intentionally not allowed to fall back to a label-only target"
            )
        fingerprint = _hash(
            {
                "sample_id": sample.sample_id,
                "gold_label": sample.label,
                "analysis": analysis.to_dict(),
            }
        )
        cached = existing.get(sample.sample_id)
        if _valid_cached(cached, fingerprint, reasoning=True):
            materialized[sample.sample_id] = {
                key: cached[key]
                for key in ("sample_id", "gold_label", "analysis", "origin")
                if key in cached
            }
            continue
        if analysis.label != sample.label:
            raise ExtractionError(f"sample {sample.sample_id!r}: reference label does not match gold")
        materialized[sample.sample_id] = {
            "sample_id": sample.sample_id,
            "gold_label": sample.label,
            "analysis": analysis.to_dict(),
            "origin": origin,
        }
    requested_ids = {sample.sample_id for sample in samples}
    preserved: list[dict[str, Any]] = []
    for sample_id, record in existing.items():
        if sample_id in requested_ids:
            continue
        try:
            analysis = _reasoning_from_record(record, role="reference")
        except (SchemaError, ExtractionError):
            continue
        preserved.append(
            {
                "sample_id": sample_id,
                "gold_label": record.get("gold_label", analysis.label),
                "analysis": analysis.to_dict(),
                "origin": record.get("origin", "reference_artifact"),
            }
        )
    ordered = [materialized[sample.sample_id] for sample in samples] + preserved
    _atomic_jsonl(output_path, ordered)
    return materialized


def _reasoning_worker(kwargs: Mapping[str, Any], visible_device: str) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = visible_device
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    _generate_reasonings(**dict(kwargs))


def _prepare_reasoning_artifacts(
    config: Mapping[str, Any], data_path: str | Path, *, limit: int | None
) -> tuple[list[SarcasmSample], dict[str, dict[str, Any]], dict[str, dict[str, Any]], dict[str, dict[str, Any]], dict[str, Any], Path]:
    teacher_path = str(get_required(config, "models.teacher.path"))
    student_path = str(get_required(config, "models.student.path"))
    validate_model_pair(teacher_path, student_path)
    samples = _load_samples(data_path, limit=limit)
    extraction_config = config.get("experience_extraction", {})
    output_dir = project_path(
        config, extraction_config.get("output_path", "experience_extraction")
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    generation = dict(extraction_config.get("generation", {}))
    json_config = generation.get("json", {})
    json_max_attempts = int(json_config.get("max_attempts", 3))
    failure_log_path = project_path(config, json_config.get("failure_log_path", "logs/json_failures.jsonl"))

    backend = str(generation.get("backend", "transformers")).lower()
    if backend == "vllm":
        schema_value = generation.get(
            "json_schema_path",
            config.get("opcd", {}).get("json_schema_path", "configs/pace_plus_reasoning.schema.json"),
        )
        schema_path = project_path(config, schema_value)
        try:
            schema = json.loads(schema_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ExtractionError(f"reasoning JSON schema is invalid: {schema_path}: {exc}") from exc
        if not isinstance(schema, Mapping):
            raise ExtractionError(f"reasoning JSON schema must be an object: {schema_path}")
        generation["_reasoning_json_schema"] = dict(schema)

    teacher_output = output_dir / "teacher_reasonings.jsonl"
    student_output = output_dir / "student_reasonings.jsonl"
    teacher_kwargs = {
        "role": "teacher",
        "samples": samples,
        "model_path": teacher_path,
        "output_path": teacher_output,
        "precomputed_path": extraction_config.get("precomputed_teacher_reasoning_path"),
        "generation_config": {**generation, "progress_position": 0},
        "failure_log_path": failure_log_path,
        "json_max_attempts": json_max_attempts,
    }
    student_kwargs = {
        "role": "student",
        "samples": samples,
        "model_path": student_path,
        "output_path": student_output,
        "precomputed_path": extraction_config.get("precomputed_student_reasoning_path"),
        "generation_config": {**generation, "progress_position": 1},
        "failure_log_path": failure_log_path,
        "json_max_attempts": json_max_attempts,
    }

    parallel_roles = backend == "vllm" and bool(generation.get("parallel_roles", False))
    if parallel_roles:
        role_devices = generation.get("role_devices", {})
        teacher_device = str(role_devices.get("teacher", "0"))
        student_device = str(role_devices.get("student", "1"))
        if teacher_device == student_device:
            raise ExtractionError("teacher and student vLLM workers must use different GPUs")
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=2, mp_context=context) as executor:
            teacher_future = executor.submit(_reasoning_worker, teacher_kwargs, teacher_device)
            student_future = executor.submit(_reasoning_worker, student_kwargs, student_device)
            teacher_future.result()
            student_future.result()
        teacher_records = load_records_by_id(teacher_output)
        student_records = load_records_by_id(student_output)
    else:
        teacher_records = _generate_reasonings(**teacher_kwargs)
        student_records = _generate_reasonings(**student_kwargs)

    verified_teachers, review_status = _prepare_verified_teacher_reasonings(
        config,
        samples,
        teacher_records,
        output_dir=output_dir,
        generation=generation,
        failure_log_path=failure_log_path,
        json_max_attempts=json_max_attempts,
    )
    return samples, teacher_records, student_records, verified_teachers, review_status, output_dir


def run_reasoning_generation(
    config: Mapping[str, Any], data_path: str | Path, *, limit: int | None = None
) -> dict[str, int | str]:
    (
        samples,
        teacher_records,
        student_records,
        _verified_teachers,
        review_status,
        output_dir,
    ) = _prepare_reasoning_artifacts(config, data_path, limit=limit)
    return {
        "samples": len(samples),
        "teacher_valid": sum(_record_reasoning_valid(teacher_records.get(sample.sample_id)) for sample in samples),
        "student_valid": sum(_record_reasoning_valid(student_records.get(sample.sample_id)) for sample in samples),
        **review_status,
        "teacher_output": str(output_dir / "teacher_reasonings.jsonl"),
        "student_output": str(output_dir / "student_reasonings.jsonl"),
    }


def run_extraction(config: Mapping[str, Any], data_path: str | Path, *, limit: int | None = None) -> dict[str, int]:
    teacher_path = str(get_required(config, "models.teacher.path"))
    (
        samples,
        teacher_records,
        student_records,
        verified_teachers,
        review_status,
        output_dir,
    ) = _prepare_reasoning_artifacts(config, data_path, limit=limit)
    if review_status["teacher_verified"] != len(samples):
        raise ExtractionError(
            "comparative extraction requires every teacher reasoning to be ready and "
            "Codex-reviewed when label-conditioned correction was needed: "
            f"verified={review_status['teacher_verified']}/{len(samples)}, "
            f"pending_review={review_status['teacher_review_pending']}, "
            f"invalid_initial={review_status['teacher_initial_invalid']}, "
            f"invalid_correction={review_status['teacher_correction_invalid']}. "
            f"Review queue: {review_status['teacher_review_queue']}; write reviews to "
            f"{review_status['teacher_reviews']}."
        )
    generation = config.get("experience_extraction", {}).get("generation", {})
    json_config = generation.get("json", {})
    json_max_attempts = int(json_config.get("max_attempts", 3))
    failure_log_path = project_path(config, json_config.get("failure_log_path", "logs/json_failures.jsonl"))

    mode = config.get("ablation", {}).get("experience_source", "comparative")
    if mode not in {"none", "teacher_only", "student_only", "comparative"}:
        raise ExtractionError(f"unsupported ablation.experience_source: {mode!r}")
    comparative_path = output_dir / "comparative_experiences.jsonl"
    comparative_existing = load_records_by_id(comparative_path, allow_missing=True)
    document = _load_prompt_document(config) if mode == "comparative" else None
    base_fingerprint = _hash(
        {
            "prompt_version": PROMPT_VERSION,
            "prompt_sha256": document.sha256 if document else None,
            "reflection_schema": REFLECTION_JSON_SCHEMA,
            "mode": mode,
            "extractor": teacher_path,
            "generation": _reflection_generation_fingerprint(generation),
        }
    )
    legacy_base_fingerprints = tuple(
        _hash(
            {
                "prompt_version": prompt_version,
                "prompt_sha256": prompt_sha256,
                "reflection_schema": _legacy_reflection_schema(),
                "mode": mode,
                "extractor": teacher_path,
                "generation": _reflection_generation_fingerprint(generation),
            }
        )
        for prompt_version, prompt_sha256 in LEGACY_REFLECTION_CACHE_IDENTITIES
    )
    pending: list[
        tuple[SarcasmSample, Reasoning, Reasoning, Reasoning, str, str]
    ] = []
    invalid_triplets = 0
    recovered_cached_records = False
    for sample in samples:
        try:
            verified_teacher = verified_teachers[sample.sample_id]
            teacher = _reasoning_from_record(
                verified_teacher,
                role="verified_teacher",
                gold_label=sample.label,
            )
            initial_teacher = _reasoning_from_record(
                teacher_records[sample.sample_id],
                role="initial_teacher",
            )
            student = _reasoning_from_record(student_records[sample.sample_id], role="student")
            teacher_origin = str(verified_teacher["origin"])
        except (KeyError, SchemaError, ExtractionError):
            invalid_triplets += 1
            continue
        fingerprint_payload = {
            "sample_input": _sample_input_fingerprint(sample),
            "teacher": teacher.to_dict(),
            "initial_teacher": initial_teacher.to_dict(),
            "student": student.to_dict(),
            "teacher_origin": teacher_origin,
        }
        fingerprint = _hash({"base": base_fingerprint, **fingerprint_payload})
        cached = comparative_existing.get(sample.sample_id)
        if (
            cached
            and cached.get("fingerprint") == fingerprint
            and cached.get("error")
            and isinstance(cached.get("raw_output"), str)
        ):
            try:
                recovered_experiences = _parse_reflection_for_teacher_origin(
                    cached["raw_output"], teacher_origin
                )
            except SchemaError:
                pass
            else:
                recovered = _comparative_record(
                    sample,
                    teacher,
                    initial_teacher,
                    student,
                    teacher_origin,
                    recovered_experiences,
                    cached["raw_output"],
                    fingerprint,
                    mode,
                )
                recovered["error"] = None
                recovered["attempts"] = int(cached.get("attempts", 0))
                recovered["json_valid"] = True
                _save_comparative(
                    output_dir,
                    comparative_path,
                    recovered,
                    update_aggregate=False,
                )
                comparative_existing[sample.sample_id] = recovered
                recovered_cached_records = True
                continue
        if _valid_cached(cached, fingerprint, reflection=True):
            continue
        if mode == "comparative" and any(
            _valid_cached(
                cached,
                _hash({"base": legacy_base, **fingerprint_payload}),
                reflection=True,
            )
            for legacy_base in legacy_base_fingerprints
        ):
            continue
        pending.append(
            (sample, teacher, initial_teacher, student, teacher_origin, fingerprint)
        )

    if recovered_cached_records:
        _atomic_jsonl(comparative_path, comparative_existing.values())

    if mode == "none":
        for sample, teacher, initial_teacher, student, teacher_origin, fingerprint in pending:
            record = _comparative_record(
                sample,
                teacher,
                initial_teacher,
                student,
                teacher_origin,
                (),
                None,
                fingerprint,
                mode,
            )
            record["error"] = None
            record["attempts"] = 0
            record["json_valid"] = True
            _save_comparative(
                output_dir,
                comparative_path,
                record,
                update_aggregate=False,
            )
            comparative_existing[sample.sample_id] = record
        if pending:
            _atomic_jsonl(comparative_path, comparative_existing.values())
    elif pending:
        backend = str(generation.get("backend", "transformers")).lower()
        if backend == "vllm":
            extractor: Any = VllmMultimodalGenerator(
                teacher_path,
                role="experience_extractor",
                max_new_tokens=int(generation.get("reflection_max_new_tokens", 2048)),
                deterministic=bool(generation.get("deterministic", True)),
                chat_template_kwargs=generation.get("chat_template_kwargs", {}),
                max_model_len=_reflection_max_model_len(generation),
                max_num_seqs=int(generation.get("max_num_seqs", 8)),
                gpu_memory_utilization=float(generation.get("gpu_memory_utilization", 0.75)),
                thinking_token_budget=int(generation.get("reflection_thinking_token_budget", 1024)),
                json_schema=REFLECTION_JSON_SCHEMA,
                enforce_eager=bool(generation.get("enforce_eager", True)),
            )
            batch_size = int(generation.get("reflection_batch_size", 16))
        else:
            extractor = MultimodalGenerator(
                teacher_path,
                role="experience_extractor",
                device_map=generation.get("device_map", "auto"),
                max_new_tokens=int(generation.get("reflection_max_new_tokens", 2048)),
                deterministic=bool(generation.get("deterministic", True)),
                chat_template_kwargs=generation.get("chat_template_kwargs", {}),
            )
            batch_size = 1

        from tqdm.auto import tqdm

        progress = tqdm(
            total=len(samples),
            initial=len(samples) - len(pending),
            disable=not bool(generation.get("show_progress", True)),
            desc="Comparative reflections",
            unit="sample",
            dynamic_ncols=True,
        )
        with extractor, progress:
            for pending_batch in _chunks(pending, batch_size):
                if hasattr(extractor, "generate_batch"):
                    states = _generate_reflection_batch(
                        generator=extractor,
                        batch=pending_batch,
                        mode=mode,
                        document=document,
                        failure_log_path=failure_log_path,
                        max_attempts=json_max_attempts,
                        model_path=teacher_path,
                    )
                else:
                    states = []
                    for pending_item in pending_batch:
                        sample, teacher, initial_teacher, student, teacher_origin, _ = pending_item
                        messages = _reflection_messages(
                            mode,
                            document,
                            sample,
                            teacher,
                            initial_teacher,
                            student,
                            teacher_origin,
                        )
                        experiences, raw, error, attempts = generate_json_with_retries(
                            extractor,
                            messages,
                            image_path=sample.image_path,
                            parser=lambda raw: _parse_reflection_for_teacher_origin(
                                raw,
                                teacher_origin,
                            ),
                            max_attempts=json_max_attempts,
                            failure_log_path=failure_log_path,
                            context={
                                "stage": "experience_extraction",
                                "artifact": "comparative_experience",
                                "sample_id": sample.sample_id,
                                "model_path": str(Path(teacher_path).resolve()),
                            },
                        )
                        states.append(
                            {
                                "pending_item": pending_item,
                                "parsed": experiences,
                                "raw": raw,
                                "error": error,
                                "attempt_count": attempts,
                            }
                        )

                for state in states:
                    (
                        sample,
                        teacher,
                        initial_teacher,
                        student,
                        teacher_origin,
                        fingerprint,
                    ) = state["pending_item"]
                    record = _comparative_record(
                        sample,
                        teacher,
                        initial_teacher,
                        student,
                        teacher_origin,
                        state["parsed"] or (),
                        state["raw"],
                        fingerprint,
                        mode,
                    )
                    record["error"] = state["error"]
                    record["attempts"] = state["attempt_count"]
                    record["json_valid"] = state["error"] is None
                    _save_comparative(
                        output_dir,
                        comparative_path,
                        record,
                        update_aggregate=False,
                    )
                    comparative_existing[sample.sample_id] = record
                _atomic_jsonl(comparative_path, comparative_existing.values())
                progress.update(len(states))

    complete = [comparative_existing.get(sample.sample_id) for sample in samples]
    result = {
        "samples": len(samples),
        "teacher_valid": sum(_record_reasoning_valid(teacher_records.get(sample.sample_id)) for sample in samples),
        "student_valid": sum(_record_reasoning_valid(student_records.get(sample.sample_id)) for sample in samples),
        "teacher_verified": len(verified_teachers),
        "experience_non_null": sum(bool(record and record.get("experiences")) for record in complete),
        "experience_null": sum(bool(record and not record.get("experiences") and not record.get("error")) for record in complete),
        "experience_items": sum(len(record.get("experiences", [])) for record in complete if record),
        "experience_invalid": sum(bool(record and record.get("error")) for record in complete),
        "invalid_pairs": invalid_triplets,
    }
    _require_complete_extraction(result, require_non_empty=mode != "none")
    return result


def _require_complete_extraction(
    result: Mapping[str, int],
    *,
    require_non_empty: bool = True,
) -> None:
    experience_null = int(result.get("experience_null", 0))
    if (
        result["experience_invalid"]
        or result["invalid_pairs"]
        or (require_non_empty and experience_null)
    ):
        raise ExtractionError(
            "comparative extraction is incomplete: "
            f"experience_invalid={result['experience_invalid']}, "
            f"invalid_pairs={result['invalid_pairs']}, "
            f"experience_null={experience_null}. "
            "Rerun the same extract command to retry only invalid records."
        )


def _record_reasoning_valid(record: Mapping[str, Any] | None) -> bool:
    if not record:
        return False
    try:
        validate_reasoning(record.get("analysis"))
        return True
    except SchemaError:
        return False


def _comparative_record(
    sample: SarcasmSample,
    teacher: Reasoning,
    initial_teacher: Reasoning,
    student: Reasoning,
    teacher_origin: str,
    experiences: Iterable[ReflectionExperience],
    raw: str | None,
    fingerprint: str,
    mode: str,
) -> dict[str, Any]:
    experience_items = tuple(experiences)
    return {
        "sample_id": sample.sample_id,
        "gold_label": sample.label,
        "teacher_reasoning": teacher.to_dict(),
        "initial_teacher_reasoning": initial_teacher.to_dict(),
        "student_reasoning": student.to_dict(),
        "teacher_reasoning_origin": teacher_origin,
        "teacher_initial_correct": teacher_origin == "blind_correct",
        "student_correct": student.label == sample.label,
        "experiences": [item.to_dict() for item in experience_items],
        "raw_output": raw,
        "experience_source": mode,
        "prompt_version": PROMPT_VERSION,
        "fingerprint": fingerprint,
    }


def _save_comparative(
    output_dir: Path,
    aggregate_path: Path,
    record: Mapping[str, Any],
    *,
    update_aggregate: bool = True,
) -> None:
    record_path = output_dir / "comparative_records" / f"{_hash(record['sample_id'])[:24]}.json"
    record_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.write_text(json.dumps(record, ensure_ascii=True, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    if update_aggregate:
        upsert_jsonl(aggregate_path, record)
