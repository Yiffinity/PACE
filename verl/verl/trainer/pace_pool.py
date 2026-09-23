"""Shared helpers for PACE weighted experience refinement."""

from __future__ import annotations

import json
import unicodedata
import re
from pathlib import Path
from typing import Any, Mapping


from verl.trainer.pace_modeling import (
    MultimodalGenerator,
    VllmTextGenerator,
)
from verl.trainer.pace_prompts import (
    CANDIDATE_NORMALIZATION_SYSTEM_PROMPT,
)
from verl.trainer.pace_schema import (
    SchemaError,
    generate_json_with_retries,
    load_records_by_id,
    parse_json_object,
    read_jsonl,
)


class ConsolidationError(RuntimeError):
    pass


STRUCTURAL_ACTIONS = frozenset({"ADD", "MODIFY", "UPVOTE", "DOWNVOTE"})
NORMALIZATION_METHOD = "label_conditioned_sarcasm_boundary_v1"


_RATIONALE_SCHEMA = {"type": "string", "minLength": 1, "maxLength": 1000}
_COMPARISON_RATIONALE_SCHEMA = {
    "type": "string",
    "minLength": 1,
    "maxLength": 500,
}

_STAGE_JSON_SCHEMAS: dict[str, dict[str, Any]] = {
    "normalization": {
        "type": "object",
        "properties": {
            "valid": {"type": "boolean"},
            "mechanism": {"type": ["string", "null"]},
            "exclusion_boundary": {"type": ["string", "null"]},
            "rationale": _RATIONALE_SCHEMA,
        },
        "required": [
            "valid",
            "mechanism",
            "exclusion_boundary",
            "rationale",
        ],
        "additionalProperties": False,
    },
    "comparison": {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": sorted(STRUCTURAL_ACTIONS)},
            "target_id": {"type": ["string", "null"]},
            "rationale": _COMPARISON_RATIONALE_SCHEMA,
        },
        "required": ["action", "target_id", "rationale"],
        "additionalProperties": False,
    },
    "merge": {
        "type": "object",
        "properties": {
            "mechanism": {"type": "string", "minLength": 1},
            "exclusion_boundary": {"type": "string", "minLength": 1},
        },
        "required": ["mechanism", "exclusion_boundary"],
        "additionalProperties": False,
    },
}


def _validate_experience_text(
    value: Any,
    *,
    field: str,
    max_experience_words: int,
) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SchemaError(f"{field} must be a non-empty string")
    if max_experience_words <= 0:
        raise SchemaError("max_experience_words must be positive")
    text = value.strip()
    unnatural = re.search(r"[A-Za-z]{31,}", text)
    if unnatural is not None:
        raise SchemaError(
            f"{field} contains an unnaturally long alphabetic token: "
            f"{unnatural.group(0)[:40]!r}"
        )
    word_count = len(text.split())
    if word_count > max_experience_words:
        raise SchemaError(
            f"{field} has {word_count} words; maximum is {max_experience_words}"
        )
    return text


def _compose_experience(
    mechanism: str,
    exclusion_boundary: str,
    *,
    max_experience_words: int,
) -> str:
    mechanism = _validate_experience_text(
        mechanism,
        field="mechanism",
        max_experience_words=max_experience_words,
    )
    exclusion_boundary = _validate_experience_text(
        exclusion_boundary,
        field="exclusion_boundary",
        max_experience_words=max_experience_words,
    )
    if not mechanism.startswith("Classify as sarcastic when "):
        raise SchemaError(
            "mechanism must begin exactly with 'Classify as sarcastic when '"
        )
    if not exclusion_boundary.startswith("Do not apply when "):
        raise SchemaError(
            "exclusion_boundary must begin exactly with 'Do not apply when '"
        )
    experience = f"{mechanism} {exclusion_boundary}"
    word_count = len(experience.split())
    if word_count > max_experience_words:
        raise SchemaError(
            f"complete experience has {word_count} words; maximum is "
            f"{max_experience_words}"
        )
    return experience


def _parse_label_conditioned_normalization(
    raw: str,
    *,
    gold_label: str,
    max_experience_words: int,
) -> dict[str, Any]:
    if gold_label not in {"sarcastic", "non-sarcastic"}:
        raise SchemaError(f"invalid candidate gold label {gold_label!r}")
    value = parse_json_object(raw)
    required = {"valid", "mechanism", "exclusion_boundary", "rationale"}
    if set(value) != required:
        raise SchemaError(
            f"candidate normalization fields must be exactly {sorted(required)}"
        )
    valid = value["valid"]
    mechanism = value["mechanism"]
    exclusion_boundary = value["exclusion_boundary"]
    rationale = value["rationale"]
    if not isinstance(valid, bool):
        raise SchemaError("candidate normalization valid must be a boolean")
    if not isinstance(rationale, str) or not rationale.strip():
        raise SchemaError("candidate normalization rationale must be non-empty")
    if not valid:
        if mechanism is not None or exclusion_boundary is not None:
            raise SchemaError(
                "invalid candidate mechanism and exclusion_boundary must be null"
            )
    elif gold_label == "sarcastic":
        _compose_experience(
            mechanism,
            exclusion_boundary,
            max_experience_words=max_experience_words,
        )
    else:
        if mechanism is not None:
            raise SchemaError("non-sarcastic candidate mechanism must be null")
        exclusion_boundary = _validate_experience_text(
            exclusion_boundary,
            field="non-sarcastic exclusion evidence",
            max_experience_words=max_experience_words,
        )
        if not exclusion_boundary.startswith("Do not infer sarcasm when "):
            raise SchemaError(
                "non-sarcastic exclusion evidence must begin exactly with "
                "'Do not infer sarcasm when '"
            )
    return {
        "valid": valid,
        "mechanism": mechanism,
        "exclusion_boundary": exclusion_boundary,
        "rationale": rationale.strip(),
    }


def _parse_merge(
    raw: str,
    *,
    max_experience_words: int = 70,
    max_mechanism_words: int | None = None,
    max_boundary_words: int | None = None,
) -> dict[str, str]:
    value = parse_json_object(raw)
    required = {"mechanism", "exclusion_boundary"}
    if set(value) != required:
        raise SchemaError(f"experience merge fields must be exactly {sorted(required)}")
    if max_mechanism_words is not None:
        _validate_experience_text(
            value["mechanism"],
            field="mechanism",
            max_experience_words=max_mechanism_words,
        )
    if max_boundary_words is not None:
        _validate_experience_text(
            value["exclusion_boundary"],
            field="exclusion_boundary",
            max_experience_words=max_boundary_words,
        )
    experience = _compose_experience(
        value["mechanism"],
        value["exclusion_boundary"],
        max_experience_words=max_experience_words,
    )
    return {
        "mechanism": value["mechanism"].strip(),
        "exclusion_boundary": value["exclusion_boundary"].strip(),
        "experience": experience,
    }


def _parse_boundary_merge(
    raw: str,
    *,
    mechanism: str,
    max_experience_words: int,
) -> dict[str, str]:
    value = parse_json_object(raw)
    if set(value) != {"revised_exclusion_boundary"}:
        raise SchemaError(
            "boundary merge fields must be exactly ['revised_exclusion_boundary']"
        )
    maximum_boundary_words = max_experience_words - len(mechanism.strip().split())
    boundary = _validate_experience_text(
        value["revised_exclusion_boundary"],
        field="revised_exclusion_boundary",
        max_experience_words=maximum_boundary_words,
    )
    experience = _compose_experience(
        mechanism,
        boundary,
        max_experience_words=max_experience_words,
    )
    return {
        "mechanism": mechanism.strip(),
        "exclusion_boundary": boundary,
        "experience": experience,
    }


def _active_items(experiences: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [item for item in experiences if item.get("active") is True]


def _active_ids(experiences: list[dict[str, Any]]) -> set[str]:
    return {item["id"] for item in _active_items(experiences)}


def _normalize_experience(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(normalized.split())


def _deduplicate_generated_add(
    decision: Mapping[str, Any],
    experiences: list[dict[str, Any]],
) -> dict[str, Any]:
    result = dict(decision)
    if result.get("action") != "ADD":
        return result
    generated = result.get("experience")
    if not isinstance(generated, str):
        return result
    normalized = _normalize_experience(generated)
    duplicate = next(
        (
            item
            for item in _active_items(experiences)
            if _normalize_experience(item["text"]) == normalized
        ),
        None,
    )
    if duplicate is not None:
        return {
            "action": "UPVOTE",
            "target_id": duplicate["id"],
            "experience": None,
            "rationale": "Generated ADD experience exactly matches and therefore upvotes an active experience after normalization.",
        }
    return result


def _render_pool(experiences: list[dict[str, Any]]) -> str:
    return "\n".join(f"- {item['text']}" for item in _active_items(experiences))


def _load_pool_entries(source: Path) -> list[dict[str, Any]]:
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConsolidationError(f"invalid experience pool JSON: {source}: {exc}") from exc
    experiences = value.get("experiences") if isinstance(value, dict) else None
    if not isinstance(experiences, list):
        raise ConsolidationError(f"experience pool must contain an experiences list: {source}")
    parsed: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for item in experiences:
        if not isinstance(item, dict) or set(item) != {"id", "text", "active"}:
            raise ConsolidationError("experience pool entries must contain only id, text, and active")
        item_id, item_text, item_active = item["id"], item["text"], item["active"]
        if (
            not isinstance(item_id, str)
            or not item_id
            or not isinstance(item_text, str)
            or not item_text.strip()
            or not isinstance(item_active, bool)
        ):
            raise ConsolidationError("invalid experience pool entry")
        if item_id in seen_ids:
            raise ConsolidationError(f"duplicate experience pool id {item_id!r}")
        seen_ids.add(item_id)
        parsed.append({"id": item_id, "text": item_text.strip(), "active": item_active})
    return parsed


def load_active_experience_text(path: str | Path) -> str:
    source = Path(path)
    if not source.is_file():
        raise ConsolidationError(f"experience pool does not exist: {source}")
    if source.suffix.lower() == ".txt":
        return source.read_text(encoding="utf-8").strip()
    return _render_pool(_load_pool_entries(source))


def _load_candidates(input_path: Path) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    for _, record in read_jsonl(input_path):
        if record.get("error"):
            continue
        source_sample_id = record.get("sample_id")
        gold_label = record.get("gold_label")
        experiences = record.get("experiences")
        if (
            not isinstance(source_sample_id, str)
            or gold_label not in {"sarcastic", "non-sarcastic"}
            or not isinstance(experiences, list)
        ):
            raise ConsolidationError("comparative reflection artifact has an invalid schema")
        for item_index, item in enumerate(experiences):
            if not isinstance(item, Mapping):
                raise ConsolidationError("comparative reflection item must be an object")
            targets, content = item.get("target_models"), item.get("content")
            if not isinstance(targets, list) or not targets:
                raise ConsolidationError("experience target_models must be a non-empty list")
            if not isinstance(content, str) or not content.strip():
                raise ConsolidationError("experience must be a non-empty string")
            candidate_id = f"{source_sample_id}::item-{item_index:02d}"
            if candidate_id in seen:
                raise ConsolidationError(f"duplicate candidate id {candidate_id!r}")
            seen.add(candidate_id)
            candidates.append(
                {
                    "sample_id": candidate_id,
                    "source_sample_id": source_sample_id,
                    "source_item_index": item_index,
                    "experience": content.strip(),
                    "gold_label": gold_label,
                }
            )
    return candidates


def _stage_settings(
    generation: Mapping[str, Any],
    stage_name: str,
) -> tuple[int, dict[str, Any]]:
    stage = generation.get(stage_name, {})
    if not isinstance(stage, Mapping):
        raise ConsolidationError(f"experience_pool.generation.{stage_name} must be an object")
    max_new_tokens = int(stage.get("max_new_tokens", generation.get("max_new_tokens", 512)))
    if max_new_tokens <= 0:
        raise ConsolidationError(
            f"experience_pool.generation.{stage_name}.max_new_tokens must be positive"
        )
    chat_template_kwargs = stage.get(
        "chat_template_kwargs", generation.get("chat_template_kwargs", {})
    )
    if not isinstance(chat_template_kwargs, Mapping):
        raise ConsolidationError(
            f"experience_pool.generation.{stage_name}.chat_template_kwargs must be an object"
        )
    return max_new_tokens, dict(chat_template_kwargs)


def _build_consolidation_generator(
    model_path: str,
    generation: Mapping[str, Any],
    *,
    max_input_tokens: int,
) -> Any:
    max_new_tokens, chat_template_kwargs = _stage_settings(generation, "normalization")
    backend = str(generation.get("backend", "transformers")).casefold()
    common = {
        "role": "consolidation_model",
        "max_new_tokens": max_new_tokens,
        "deterministic": bool(generation.get("deterministic", True)),
        "chat_template_kwargs": chat_template_kwargs,
    }
    if backend == "transformers":
        return MultimodalGenerator(
            model_path,
            device_map=generation.get("device_map", "auto"),
            empty_cache_after_generate=bool(
                generation.get("empty_cache_after_generate", False)
            ),
            **common,
        )
    if backend != "vllm":
        raise ConsolidationError(
            f"unsupported experience_pool generation backend: {backend!r}"
        )

    max_model_len = int(generation.get("max_model_len", max_input_tokens + 4096))
    if max_model_len <= max_input_tokens:
        raise ConsolidationError(
            "experience_pool.generation.max_model_len must exceed max_input_tokens"
        )
    return VllmTextGenerator(
        model_path,
        max_model_len=max_model_len,
        max_num_seqs=int(generation.get("max_num_seqs", 1)),
        gpu_memory_utilization=float(
            generation.get("gpu_memory_utilization", 0.75)
        ),
        enable_prefix_caching=bool(
            generation.get("enable_prefix_caching", True)
        ),
        prefix_caching_hash_algo=str(
            generation.get("prefix_caching_hash_algo", "xxhash")
        ),
        mamba_cache_mode=str(generation.get("mamba_cache_mode", "align")),
        enforce_eager=bool(generation.get("enforce_eager", False)),
        **common,
    )


def _run_json_stage(
    generator: Any,
    *,
    stage_name: str,
    artifact: str,
    sample_id: str,
    messages: list[dict[str, str]],
    parser: Any,
    generation: Mapping[str, Any],
    max_input_tokens: int,
    json_max_attempts: int,
    failure_log_path: Path,
    teacher_path: str,
    json_schema: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], int]:
    max_new_tokens, chat_template_kwargs = _stage_settings(generation, stage_name)
    generator.max_new_tokens = max_new_tokens
    generator.chat_template_kwargs = chat_template_kwargs
    prompt_tokens = generator.count_input_tokens(messages)
    if prompt_tokens > max_input_tokens:
        raise ConsolidationError(
            f"{stage_name} prompt for {sample_id} has {prompt_tokens} input tokens, "
            f"exceeding max_input_tokens={max_input_tokens}; no action was written"
        )
    previous_schema = getattr(generator, "json_schema", None)
    if hasattr(generator, "json_schema"):
        generator.json_schema = json_schema or _STAGE_JSON_SCHEMAS[stage_name]
    try:
        result, _raw, error, attempts = generate_json_with_retries(
            generator,
            messages,
            image_path=None,
            parser=parser,
            max_attempts=json_max_attempts,
            failure_log_path=failure_log_path,
            context={
                "stage": "experience_consolidation",
                "artifact": artifact,
                "sample_id": sample_id,
                "model_path": teacher_path,
            },
        )
    finally:
        if hasattr(generator, "json_schema"):
            generator.json_schema = previous_schema
    if result is None:
        raise ConsolidationError(
            f"{stage_name} stopped at {sample_id} after {attempts} invalid attempts: "
            f"{error}; no action was written. See {failure_log_path}"
        )
    return result, {"result": result, "attempts": attempts}, prompt_tokens


def _run_candidate_normalization(
    generator: MultimodalGenerator,
    *,
    candidate: Mapping[str, Any],
    generation: Mapping[str, Any],
    max_input_tokens: int,
    max_experience_words: int,
    json_max_attempts: int,
    failure_log_path: Path,
    teacher_path: str,
) -> tuple[dict[str, Any], dict[str, Any], int]:
    messages = [
        {"role": "system", "content": CANDIDATE_NORMALIZATION_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": json.dumps(
                {
                    "candidate_experience": candidate["experience"],
                    "gold_label": candidate["gold_label"],
                },
                ensure_ascii=True,
                sort_keys=True,
            ),
        },
    ]
    return _run_json_stage(
        generator,
        stage_name="normalization",
        artifact="candidate_normalization",
        sample_id=candidate["sample_id"],
        messages=messages,
        parser=lambda value: _parse_label_conditioned_normalization(
            value,
            gold_label=candidate["gold_label"],
            max_experience_words=max_experience_words,
        ),
        generation=generation,
        max_input_tokens=max_input_tokens,
        json_max_attempts=json_max_attempts,
        failure_log_path=failure_log_path,
        teacher_path=teacher_path,
    )


_NORMALIZATION_WORKER_GENERATOR: MultimodalGenerator | None = None


def _initialize_normalization_worker(
    model_path: str,
    device_map: Any,
    max_new_tokens: int,
    deterministic: bool,
    chat_template_kwargs: Mapping[str, Any],
    empty_cache_after_generate: bool,
) -> None:
    global _NORMALIZATION_WORKER_GENERATOR
    _NORMALIZATION_WORKER_GENERATOR = MultimodalGenerator(
        model_path,
        role="consolidation_normalization_model",
        device_map=device_map,
        max_new_tokens=max_new_tokens,
        deterministic=deterministic,
        chat_template_kwargs=chat_template_kwargs,
        empty_cache_after_generate=empty_cache_after_generate,
    )


def _run_candidate_normalization_worker(
    *,
    candidate: Mapping[str, Any],
    generation: Mapping[str, Any],
    max_input_tokens: int,
    max_experience_words: int,
    json_max_attempts: int,
    failure_log_path: Path,
    teacher_path: str,
) -> tuple[dict[str, Any], dict[str, Any], int]:
    if _NORMALIZATION_WORKER_GENERATOR is None:
        raise ConsolidationError("normalization worker model was not initialized")
    return _run_candidate_normalization(
        _NORMALIZATION_WORKER_GENERATOR,
        candidate=candidate,
        generation=generation,
        max_input_tokens=max_input_tokens,
        max_experience_words=max_experience_words,
        json_max_attempts=json_max_attempts,
        failure_log_path=failure_log_path,
        teacher_path=teacher_path,
    )


def _close_normalization_worker() -> None:
    global _NORMALIZATION_WORKER_GENERATOR
    generator = _NORMALIZATION_WORKER_GENERATOR
    _NORMALIZATION_WORKER_GENERATOR = None
    if generator is not None:
        generator.close()


def _load_normalization_cache(
    cache_path: Path,
    candidates: list[dict[str, Any]],
    *,
    max_experience_words: int,
) -> dict[str, dict[str, Any]]:
    records = load_records_by_id(cache_path, allow_missing=True)
    cache_order = list(records)
    candidate_order = [item["sample_id"] for item in candidates]
    if cache_order[: len(candidate_order)] != candidate_order[: len(cache_order)]:
        raise ConsolidationError(
            "normalization cache and comparative source are reordered or incompatible"
        )
    parsed: dict[str, dict[str, Any]] = {}
    candidate_by_id = {item["sample_id"]: item for item in candidates}
    for sample_id, record in records.items():
        candidate = candidate_by_id.get(sample_id)
        if candidate is None:
            continue
        required = {"sample_id", "method", "gold_label", "normalization", "call"}
        if set(record) != required or record.get("method") != NORMALIZATION_METHOD:
            raise ConsolidationError(
                f"normalization cache {sample_id} uses an incompatible format; "
                "rerun refine once with --fresh"
            )
        gold_label = record.get("gold_label")
        if gold_label != candidate["gold_label"]:
            raise ConsolidationError(
                f"normalization cache {sample_id} gold_label does not match its source"
            )
        normalization = _parse_label_conditioned_normalization(
            json.dumps(record["normalization"]),
            gold_label=gold_label,
            max_experience_words=max_experience_words,
        )
        call = record["call"]
        if (
            not isinstance(call, Mapping)
            or not isinstance(call.get("attempts"), int)
            or call["attempts"] <= 0
        ):
            raise ConsolidationError(
                f"normalization cache {sample_id} has an invalid call record"
            )
        parsed[sample_id] = {
            "normalization": normalization,
            "call": {
                "result": normalization,
                "attempts": call["attempts"],
                "cached": True,
            },
        }
    return parsed
