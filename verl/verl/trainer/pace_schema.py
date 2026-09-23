"""Strict task schemas and JSONL utilities for PACE."""

from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence


LABELS = frozenset({"sarcastic", "non-sarcastic"})
REASONING_FIELDS = ("visual_evidence", "textual_evidence", "explanation", "label")

# Shared with vLLM constrained decoding so every stage uses one output contract.
REASONING_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "visual_evidence": {"type": "string", "minLength": 1, "maxLength": 600},
        "textual_evidence": {"type": "string", "minLength": 1, "maxLength": 600},
        "explanation": {"type": "string", "minLength": 1, "maxLength": 1200},
        "label": {"enum": ["sarcastic", "non-sarcastic"]},
    },
    "required": list(REASONING_FIELDS),
    "additionalProperties": False,
}


class SchemaError(ValueError):
    """Raised when an artifact violates a PACE data contract."""


@dataclass(frozen=True)
class Reasoning:
    visual_evidence: str
    textual_evidence: str
    explanation: str
    label: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class SarcasmSample:
    sample_id: str
    image_path: str
    text: str
    label: str
    source: str | None = None
    language: str | None = None
    reference_reasoning: Reasoning | None = None
    extraction_metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        if self.reference_reasoning is not None:
            value["reference_reasoning"] = self.reference_reasoning.to_dict()
        return value


def _englishish(value: str) -> bool:
    letters = re.findall(r"[A-Za-z]", value)
    non_ascii_letters = re.findall(r"[^\x00-\x7F\d\W_]", value)
    return bool(letters) and len(letters) >= len(non_ascii_letters)


def validate_label(value: Any, *, field: str = "label") -> str:
    if not isinstance(value, str) or value not in LABELS:
        raise SchemaError(f"{field} must be exactly one of {sorted(LABELS)}, got {value!r}")
    return value


def validate_reasoning(value: Any, *, require_english: bool = True) -> Reasoning:
    if not isinstance(value, Mapping):
        raise SchemaError(f"reasoning must be a JSON object, got {type(value).__name__}")
    actual = set(value)
    required = set(REASONING_FIELDS)
    if actual != required:
        missing = sorted(required - actual)
        extra = sorted(actual - required)
        raise SchemaError(f"reasoning fields must be exactly {list(REASONING_FIELDS)}; missing={missing}, extra={extra}")

    narrative: dict[str, str] = {}
    for field in REASONING_FIELDS[:3]:
        item = value[field]
        if not isinstance(item, str) or not item.strip():
            raise SchemaError(f"reasoning.{field} must be a non-empty string")
        if require_english and not _englishish(item):
            raise SchemaError(f"reasoning.{field} must be in English")
        narrative[field] = item.strip()

    label = validate_label(value["label"], field="reasoning.label")
    return Reasoning(label=label, **narrative)


def _strip_thinking_wrapper(raw: str) -> str:
    """Remove Qwen's visible thinking wrapper before parsing the final answer."""
    if not isinstance(raw, str) or not raw.strip():
        raise SchemaError("model output is empty")
    text = raw.strip()
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1].strip()
    if "<think>" in text:
        raise SchemaError("model output contains an unterminated thinking block")
    return text


def parse_json_object(raw: str) -> dict[str, Any]:
    text = _strip_thinking_wrapper(raw)
    if text.startswith("```") or text.endswith("```"):
        raise SchemaError("model output must not contain Markdown fences")
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise SchemaError(f"model output is not a single JSON object: {exc}") from exc
    if not isinstance(value, dict):
        raise SchemaError("model output must be a JSON object")
    return value


def parse_reasoning(raw: str, *, require_english: bool = True) -> Reasoning:
    return validate_reasoning(parse_json_object(raw), require_english=require_english)


def validate_sample(value: Any, *, require_image: bool = True) -> SarcasmSample:
    if not isinstance(value, Mapping):
        raise SchemaError("sample must be a JSON object")
    sample_id = value.get("sample_id")
    image_path = value.get("image_path")
    text = value.get("text")
    if not isinstance(sample_id, str) or not sample_id.strip():
        raise SchemaError("sample_id must be a non-empty string")
    if not isinstance(image_path, str) or not image_path.strip():
        raise SchemaError(f"sample {sample_id!r}: image_path must be a non-empty string")
    resolved_image = Path(image_path).expanduser().resolve()
    if require_image and (not resolved_image.is_file()):
        raise SchemaError(f"sample {sample_id!r}: image does not exist: {resolved_image}")
    if not isinstance(text, str) or not text.strip():
        raise SchemaError(f"sample {sample_id!r}: text must be a non-empty string")
    label = validate_label(value.get("label"), field=f"sample {sample_id!r}.label")

    reference = value.get("reference_reasoning")
    parsed_reference = validate_reasoning(reference) if reference is not None else None
    if parsed_reference is not None and parsed_reference.label != label:
        raise SchemaError(
            f"sample {sample_id!r}: reference label {parsed_reference.label!r} does not match gold label {label!r}"
        )
    extraction_metadata = value.get("extraction_metadata", {})
    if not isinstance(extraction_metadata, Mapping):
        raise SchemaError(f"sample {sample_id!r}: extraction_metadata must be an object")
    return SarcasmSample(
        sample_id=sample_id.strip(),
        image_path=str(resolved_image),
        text=text.strip(),
        label=label,
        source=_optional_string(value.get("source")),
        language=_optional_string(value.get("language")),
        reference_reasoning=parsed_reference,
        extraction_metadata=dict(extraction_metadata),
    )


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise SchemaError(f"optional metadata must be a string or null, got {type(value).__name__}")
    return value.strip() or None


def read_jsonl(path: str | Path) -> Iterable[tuple[int, dict[str, Any]]]:
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SchemaError(f"{source}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(value, dict):
                raise SchemaError(f"{source}:{line_number}: expected a JSON object")
            yield line_number, value


def load_records_by_id(path: str | Path, *, allow_missing: bool = False) -> dict[str, dict[str, Any]]:
    source = Path(path)
    if allow_missing and not source.exists():
        return {}
    records: dict[str, dict[str, Any]] = {}
    for line_number, record in read_jsonl(source):
        sample_id = record.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise SchemaError(f"{source}:{line_number}: missing sample_id")
        if sample_id in records:
            raise SchemaError(f"{source}:{line_number}: duplicate sample_id {sample_id!r}")
        records[sample_id] = record
    return records


def append_jsonl(path: str | Path, record: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n")
        handle.flush()


def generate_json_with_retries(
    generator: Any,
    messages: Sequence[Mapping[str, Any]],
    *,
    image_path: str | Path | None,
    parser: Callable[[str], Any],
    max_attempts: int,
    failure_log_path: str | Path,
    context: Mapping[str, Any],
) -> tuple[Any | None, str | None, str | None, int]:
    """Generate and validate JSON, retrying invalid outputs at most three times."""
    if not 1 <= int(max_attempts) <= 3:
        raise SchemaError("json max_attempts must be between 1 and 3")
    current_messages = list(messages)
    attempts: list[dict[str, Any]] = []
    last_raw: str | None = None
    last_error: str | None = None
    original_max_new_tokens = getattr(generator, "max_new_tokens", None)
    for attempt in range(1, int(max_attempts) + 1):
        if isinstance(original_max_new_tokens, int) and attempt > 1:
            generator.max_new_tokens = original_max_new_tokens * 2
        try:
            raw = generator.generate(current_messages, image_path=image_path)
            last_raw = raw
        except Exception as exc:  # generation failures are logged with JSON failures
            raw = None
            last_raw = None
            last_error = f"generation failed: {type(exc).__name__}: {exc}"
            attempts.append({"attempt": attempt, "error": last_error, "raw_output": None})
            error_text = str(exc).casefold()
            error_type = type(exc).__name__.casefold()
            fatal_accelerator_error = (
                "device-side assert" in error_text
                or "illegal memory access" in error_text
                or "out of memory" in error_text
                or "outofmemory" in error_type
            )
            if fatal_accelerator_error:
                if isinstance(original_max_new_tokens, int):
                    generator.max_new_tokens = original_max_new_tokens
                append_jsonl(
                    failure_log_path,
                    {
                        **dict(context),
                        "max_attempts": int(max_attempts),
                        "attempts": attempts,
                        "error": last_error,
                        "fatal": True,
                    },
                )
                raise RuntimeError(
                    f"fatal accelerator failure while generating {context.get('artifact', 'JSON')} "
                    f"for sample {context.get('sample_id', '<unknown>')}: {exc}"
                ) from exc
        else:
            try:
                parsed = parser(raw)
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                attempts.append({"attempt": attempt, "error": last_error, "raw_output": raw})
            else:
                if isinstance(original_max_new_tokens, int):
                    generator.max_new_tokens = original_max_new_tokens
                return parsed, raw, None, attempt
        if attempt < int(max_attempts):
            correction_messages: list[dict[str, Any]] = []
            if raw is not None:
                correction_messages.append(
                    {
                        "role": "assistant",
                        "content": (
                            "I will discard the invalid output and generate a fresh JSON object."
                        ),
                    }
                )
            correction_messages.append(
                {
                    "role": "user",
                    "content": (
                        "Your previous response did not satisfy the required JSON contract. "
                        "Generate a fresh response and return exactly one valid JSON object, with no "
                        "Markdown, prose, or extra fields. When the validation error reports a "
                        "word limit, rewrite the value to at least ten words below the stated maximum "
                        "while preserving the required information; combine overlapping clauses instead "
                        "of merely removing one or two words. "
                        f"Validation error: {last_error}"
                    ),
                }
            )
            current_messages = [*current_messages, *correction_messages]
    if isinstance(original_max_new_tokens, int):
        generator.max_new_tokens = original_max_new_tokens
    append_jsonl(
        failure_log_path,
        {
            **dict(context),
            "max_attempts": int(max_attempts),
            "attempts": attempts,
            "error": last_error or "invalid JSON output",
        },
    )
    return None, last_raw, last_error or "invalid JSON output", len(attempts)



def upsert_jsonl(path: str | Path, record: Mapping[str, Any], *, key: str = "sample_id") -> None:
    """Atomically replace one keyed record while preserving deterministic order."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    record_key = record.get(key)
    if not isinstance(record_key, str) or not record_key:
        raise SchemaError(f"upsert record requires a non-empty {key}")
    records = list(read_jsonl(target)) if target.exists() else []
    values: list[dict[str, Any]] = []
    replaced = False
    for _, value in records:
        if value.get(key) == record_key:
            if not replaced:
                values.append(dict(record))
                replaced = True
            continue
        values.append(value)
    if not replaced:
        values.append(dict(record))
    fd, temp_name = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for value in values:
                handle.write(json.dumps(value, ensure_ascii=True, sort_keys=True) + "\n")
        os.replace(temp_name, target)
    except BaseException:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise
