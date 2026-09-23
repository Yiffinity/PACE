"""Streaming ALARM-style weighted reference consolidation."""

from __future__ import annotations

from collections import Counter
from concurrent.futures import Future, ProcessPoolExecutor
from datetime import datetime
from multiprocessing import get_context
import json
import os
from pathlib import Path
import tempfile
import traceback
from queue import Empty
from typing import Any, Iterable, Mapping, Sequence

from tqdm.auto import tqdm

from verl.trainer.pace_config import get_required, project_path
from verl.trainer.pace_pool import (
    ConsolidationError,
    NORMALIZATION_METHOD,
    _active_ids,
    _build_consolidation_generator,
    _close_normalization_worker,
    _compose_experience,
    _deduplicate_generated_add,
    _initialize_normalization_worker,
    _load_candidates,
    _load_normalization_cache,
    _parse_boundary_merge,
    _parse_merge,
    _run_candidate_normalization,
    _run_candidate_normalization_worker,
    _run_json_stage,
)
from verl.trainer.pace_prompts import (
    CONSOLIDATION_SYSTEM_PROMPT,
    EXCLUSION_BOUNDARY_MERGE_SYSTEM_PROMPT,
    EXPERIENCE_MERGE_SYSTEM_PROMPT,
    exclusion_boundary_abstraction_system_prompt,
    exclusion_boundary_compression_system_prompt,
    experience_compression_system_prompt,
)
from verl.trainer.pace_schema import (
    SchemaError,
    append_jsonl,
    load_records_by_id,
    parse_json_object,
    read_jsonl,
    upsert_jsonl,
)


STRUCTURAL_ACTIONS = frozenset({"ADD", "MODIFY", "UPVOTE", "DOWNVOTE"})
LOG_ACTIONS = frozenset({*STRUCTURAL_ACTIONS, "SKIP"})
REFERENCE_FLOW_METHOD = "sarcasm_mechanism_boundary_v2"


def _compression_field_word_budgets(
    mechanism: str,
    exclusion_boundary: str,
    target_words: int,
) -> tuple[int, int]:
    mechanism_words = len(mechanism.strip().split())
    boundary_words = len(exclusion_boundary.strip().split())
    source_words = mechanism_words + boundary_words
    if source_words <= 0 or target_words < 10:
        raise ConsolidationError("invalid experience compression word budget")
    mechanism_budget = round(target_words * mechanism_words / source_words)
    mechanism_budget = min(target_words - 5, max(5, mechanism_budget))
    return mechanism_budget, target_words - mechanism_budget


def _boundary_compression_word_budget(
    *,
    mechanism_words: int,
    profile_total_words: int,
    accepted_maximum_words: int,
    tolerance_margin_words: int,
) -> int:
    available = accepted_maximum_words - mechanism_words
    if available < 5:
        raise ConsolidationError(
            "the immutable mechanism leaves fewer than 5 words under the "
            "accepted complete-experience limit"
        )
    preferred = profile_total_words - mechanism_words
    if preferred >= 5:
        return preferred
    return max(5, available - tolerance_margin_words)


def _parse_compressed_merge(
    raw: str,
    *,
    accepted_maximum_words: int,
    target_maximum_words: int,
    maximum_mechanism_words: int,
    maximum_boundary_words: int,
) -> dict[str, str]:
    """Accept the configured tolerance but make over-limit retry feedback actionable."""
    merged = _parse_merge(raw, max_experience_words=1000)
    mechanism_words = len(merged["mechanism"].split())
    boundary_words = len(merged["exclusion_boundary"].split())
    total_words = mechanism_words + boundary_words
    if not merged["mechanism"].endswith((".", "!", "?")):
        raise SchemaError("compressed mechanism must be a complete sentence")
    if not merged["exclusion_boundary"].endswith((".", "!", "?")):
        raise SchemaError("compressed exclusion_boundary must be a complete sentence")
    if total_words > accepted_maximum_words:
        raise SchemaError(
            "complete experience has "
            f"{total_words} words; maximum is {accepted_maximum_words}. "
            f"Resynthesize toward {target_maximum_words} words from the supplied "
            "structured sources: mechanism has "
            f"{mechanism_words} words (target {maximum_mechanism_words}) and "
            f"exclusion_boundary has {boundary_words} words "
            f"(target {maximum_boundary_words}). Group surface variants under "
            "shared mechanism-level categories rather than deleting isolated words"
        )
    return merged


def _parse_compressed_boundary_merge(
    raw: str,
    *,
    mechanism: str,
    accepted_maximum_words: int,
    target_total_words: int,
    target_boundary_words: int,
) -> dict[str, str]:
    """Accept the tolerance while directing an over-limit boundary resynthesis."""
    merged = _parse_boundary_merge(
        raw,
        mechanism=mechanism,
        max_experience_words=1000,
    )
    boundary_words = len(merged["exclusion_boundary"].split())
    total_words = len(merged["experience"].split())
    if not merged["exclusion_boundary"].endswith((".", "!", "?")):
        raise SchemaError(
            "compressed exclusion_boundary must be a complete sentence"
        )
    if total_words > accepted_maximum_words:
        raise SchemaError(
            "complete experience has "
            f"{total_words} words; maximum is {accepted_maximum_words}. "
            f"Resynthesize the boundary toward {target_boundary_words} words "
            f"({target_total_words} including the immutable mechanism); the "
            f"current boundary has {boundary_words} words. Group surface variants "
            "under shared countercondition categories rather than deleting "
            "isolated words"
        )
    return merged


def _parse_boundary_abstraction(
    raw: str,
    *,
    maximum_categories: int,
    maximum_category_words: int = 18,
) -> dict[str, list[str]]:
    value = parse_json_object(raw)
    if set(value) != {"causal_categories"}:
        raise SchemaError(
            "boundary abstraction fields must be exactly ['causal_categories']"
        )
    categories = value["causal_categories"]
    if not isinstance(categories, list) or not 1 <= len(categories) <= maximum_categories:
        raise SchemaError(
            "causal_categories must contain between 1 and "
            f"{maximum_categories} high-level categories"
        )
    normalized: list[str] = []
    seen: set[str] = set()
    for index, category in enumerate(categories):
        if not isinstance(category, str) or not category.strip():
            raise SchemaError(f"causal_categories[{index}] must be non-empty")
        category = " ".join(category.split())
        words = len(category.split())
        if words > maximum_category_words:
            raise SchemaError(
                f"causal_categories[{index}] has {words} words; maximum is "
                f"{maximum_category_words}; abstract its shared causal role"
            )
        key = category.casefold().rstrip(".!?")
        if key in seen:
            raise SchemaError("causal_categories must not contain duplicates")
        seen.add(key)
        normalized.append(category.rstrip(".!?"))
    return {"causal_categories": normalized}


def _boundary_abstraction_schema(maximum_categories: int) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "causal_categories": {
                "type": "array",
                "items": {"type": "string", "minLength": 1},
                "minItems": 1,
                "maxItems": maximum_categories,
            }
        },
        "required": ["causal_categories"],
        "additionalProperties": False,
    }




def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(value, ensure_ascii=True, sort_keys=True, indent=2))
            handle.write("\n")
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _atomic_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=True, sort_keys=True))
                handle.write("\n")
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _active(items: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [dict(item) if not isinstance(item, dict) else item for item in items if item.get("active") is True]


def _rank_key(item: Mapping[str, Any]) -> tuple[int, int]:
    return (-int(item["importance"]), -int(item["created_order"]))


def _ordered_active(items: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return sorted(_active(items), key=_rank_key)


def _active_view(items: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    return {item["id"]: item["text"] for item in _ordered_active(items)}


def _allowed_relation_actions(gold_label: str) -> frozenset[str]:
    if gold_label == "sarcastic":
        return frozenset({"ADD", "MODIFY", "UPVOTE"})
    if gold_label == "non-sarcastic":
        return frozenset({"MODIFY", "UPVOTE", "DOWNVOTE", "SKIP"})
    raise ConsolidationError(f"invalid candidate gold label {gold_label!r}")


def _new_reference(
    reference_id: str,
    mechanism: str,
    exclusion_boundary: str,
    created_order: int,
    *,
    max_experience_words: int = 100,
) -> dict[str, Any]:
    text = _compose_experience(
        mechanism,
        exclusion_boundary,
        max_experience_words=max_experience_words,
    )
    return {
        "id": reference_id,
        "mechanism": mechanism.strip(),
        "exclusion_boundary": exclusion_boundary.strip(),
        "text": text,
        "active": True,
        "importance": 2,
        "revision": 1,
        "created_order": created_order,
        "modify_count": 0,
        "boundary_modify_count": 0,
        "upvote_count": 0,
        "downvote_count": 0,
    }


def _find_active(items: list[dict[str, Any]], reference_id: str) -> dict[str, Any]:
    try:
        return next(
            item
            for item in items
            if item["id"] == reference_id and item.get("active") is True
        )
    except StopIteration as exc:
        raise ConsolidationError(
            f"reference action targets inactive or missing ID {reference_id!r}"
        ) from exc


def _weighted_relation_schema(
    active_ids: set[str],
    gold_label: str,
) -> dict[str, Any]:
    allowed_actions = sorted(_allowed_relation_actions(gold_label))
    return {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": allowed_actions},
            "target_id": {
                "type": ["string", "null"],
                "enum": [None, *sorted(active_ids)],
            },
            "rationale": {"type": "string", "minLength": 1, "maxLength": 500},
        },
        "required": ["action", "target_id", "rationale"],
        "additionalProperties": False,
    }


def _parse_weighted_relation(
    raw: str,
    active_ids: set[str],
    gold_label: str,
) -> dict[str, Any]:
    value = parse_json_object(raw)
    required = {"action", "target_id", "rationale"}
    if set(value) != required:
        raise SchemaError(f"reference relation fields must be exactly {sorted(required)}")
    action = value["action"]
    target_id = value["target_id"]
    rationale = value["rationale"]
    allowed_actions = _allowed_relation_actions(gold_label)
    if action not in allowed_actions:
        raise SchemaError(
            f"{gold_label} candidate action must be one of "
            f"{sorted(allowed_actions)}"
        )
    if action in {"MODIFY", "UPVOTE", "DOWNVOTE"} and target_id not in active_ids:
        valid_ids = ", ".join(sorted(active_ids)) or "<none>"
        raise SchemaError(
            f"{action} target_id {target_id!r} is not active; copy exactly one "
            f"current active ID from this list: {valid_ids}"
        )
    if action in {"ADD", "SKIP"} and target_id is not None:
        raise SchemaError(f"{action} target_id must be null")
    if not isinstance(rationale, str) or not rationale.strip():
        raise SchemaError("reference relation rationale must be non-empty")
    return {
        "action": action,
        "target_id": target_id,
        "rationale": rationale.strip(),
    }


def _parse_action_record(
    record: Mapping[str, Any],
    active_ids: set[str],
    *,
    gold_label: str,
    max_experience_words: int = 100,
) -> dict[str, Any]:
    if record.get("candidate_label") != gold_label:
        raise ConsolidationError("reference action candidate_label does not match its source")
    action = record.get("action")
    if action not in LOG_ACTIONS or (
        action != "SKIP" and action not in _allowed_relation_actions(gold_label)
    ):
        raise ConsolidationError(
            f"invalid {gold_label} reference action {action!r}"
        )
    rationale = record.get("rationale")
    if not isinstance(rationale, str) or not rationale.strip():
        raise ConsolidationError("reference action rationale must be non-empty")

    target_id = record.get("target_id")
    experience = record.get("experience")
    mechanism = record.get("mechanism")
    exclusion_boundary = record.get("exclusion_boundary")
    modification_scope = record.get("modification_scope")
    assigned_id = record.get("assigned_id")

    if action == "ADD":
        if gold_label != "sarcastic":
            raise ConsolidationError("non-sarcastic candidate cannot ADD a reference")
        if target_id is not None or not isinstance(assigned_id, str):
            raise ConsolidationError("ADD requires assigned_id and null target_id")
        if modification_scope != "FULL":
            raise ConsolidationError("ADD modification_scope must be FULL")
        composed = _compose_experience(
            mechanism,
            exclusion_boundary,
            max_experience_words=max_experience_words,
        )
        if experience != composed:
            raise ConsolidationError("ADD experience does not match its structured fields")
    elif action == "MODIFY":
        if target_id not in active_ids or assigned_id is not None:
            raise ConsolidationError("MODIFY requires one active target_id")
        expected_scope = "FULL" if gold_label == "sarcastic" else "BOUNDARY"
        if modification_scope != expected_scope:
            raise ConsolidationError(
                f"{gold_label} MODIFY modification_scope must be {expected_scope}"
            )
        composed = _compose_experience(
            mechanism,
            exclusion_boundary,
            max_experience_words=max_experience_words,
        )
        if experience != composed:
            raise ConsolidationError(
                "MODIFY experience does not match its structured fields"
            )
    else:
        if action in {"UPVOTE", "DOWNVOTE"} and target_id not in active_ids:
            raise ConsolidationError(f"{action} requires one active target_id")
        if action == "SKIP" and target_id is not None:
            raise ConsolidationError("SKIP target_id must be null")
        if assigned_id is not None or any(
            value is not None
            for value in (
                experience,
                mechanism,
                exclusion_boundary,
                modification_scope,
            )
        ):
            raise ConsolidationError(f"{action} cannot write a reference")

    return {
        "action": action,
        "target_id": target_id,
        "experience": experience,
        "mechanism": mechanism,
        "exclusion_boundary": exclusion_boundary,
        "modification_scope": modification_scope,
        "assigned_id": assigned_id,
        "rationale": rationale.strip(),
    }


def _apply_structural_action(
    items: list[dict[str, Any]],
    action: Mapping[str, Any],
    *,
    max_experience_words: int = 100,
) -> None:
    kind = action["action"]
    if kind == "SKIP":
        return
    if kind == "ADD":
        assigned_id = action["assigned_id"]
        if any(item["id"] == assigned_id for item in items):
            raise ConsolidationError(f"duplicate reference ID {assigned_id}")
        numeric = assigned_id.removeprefix("exp-")
        if not numeric.isdigit():
            raise ConsolidationError(f"invalid generated reference ID {assigned_id!r}")
        items.append(
            _new_reference(
                assigned_id,
                action["mechanism"],
                action["exclusion_boundary"],
                int(numeric),
                max_experience_words=max_experience_words,
            )
        )
        return

    target = _find_active(items, action["target_id"])
    if kind == "UPVOTE":
        target["importance"] += 1
        target["upvote_count"] += 1
    elif kind == "DOWNVOTE":
        target["importance"] -= 1
        target["downvote_count"] += 1
        if target["importance"] <= 0:
            target["active"] = False
    elif kind == "MODIFY":
        if (
            action["modification_scope"] == "BOUNDARY"
            and action["mechanism"] != target["mechanism"]
        ):
            raise ConsolidationError(
                "boundary-only MODIFY attempted to change the target mechanism"
            )
        target["mechanism"] = action["mechanism"]
        target["exclusion_boundary"] = action["exclusion_boundary"]
        target["text"] = _compose_experience(
            target["mechanism"],
            target["exclusion_boundary"],
            max_experience_words=max_experience_words,
        )
        target["importance"] += 1
        target["modify_count"] += 1
        if action["modification_scope"] == "BOUNDARY":
            target["boundary_modify_count"] += 1
        target["revision"] += 1


def _enforce_reference_limit(
    items: list[dict[str, Any]],
    target_size: int,
) -> list[str]:
    previously_active = {
        item["id"] for item in items if item.get("active") is True
    }
    eligible = sorted(
        (item for item in items if int(item["importance"]) > 0),
        key=_rank_key,
    )
    retained_ids = {item["id"] for item in eligible[:target_size]}
    for item in items:
        item["active"] = item["id"] in retained_ids
    return sorted(previously_active - retained_ids)


def _replay_state(
    candidates: list[dict[str, Any]],
    actions_path: Path,
    evaluations_path: Path,
    *,
    target_size: int = 100,
    max_experience_words: int = 100,
) -> tuple[list[dict[str, Any]], int, Counter[str], int, bool]:
    if evaluations_path.exists() and evaluations_path.stat().st_size > 0:
        raise ConsolidationError(
            "legacy per-reference evaluation artifacts exist; rerun once with "
            "--fresh"
        )
    actions = load_records_by_id(actions_path, allow_missing=True)
    action_order = list(actions)
    candidate_order = [candidate["sample_id"] for candidate in candidates]
    if len(action_order) > len(candidate_order) or action_order != candidate_order[: len(action_order)]:
        raise ConsolidationError("reference actions do not match the candidate prefix")

    items: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for index, candidate in enumerate(candidates[: len(action_order)]):
        record = actions[candidate["sample_id"]]
        if record.get("method") != REFERENCE_FLOW_METHOD:
            raise ConsolidationError(
                "reference actions use an incompatible operation set; rerun once "
                "with --fresh"
            )
        calls = record.get("calls")
        if isinstance(calls, Mapping):
            for call in calls.values():
                if isinstance(call, Mapping) and call.get("fallback_action") in {
                    "SKIP",
                    "UPVOTE",
                }:
                    raise ConsolidationError(
                        "legacy semantic fallback found in reference action "
                        f"{index + 1} ({candidate['sample_id']}); archive the action "
                        "file and truncate it before this record, then resume. "
                        "Normalization records can be retained."
                    )
        action = _parse_action_record(
            record,
            _active_ids(items),
            gold_label=candidate["gold_label"],
            max_experience_words=max_experience_words,
        )
        _apply_structural_action(
            items,
            action,
            max_experience_words=max_experience_words,
        )
        _enforce_reference_limit(items, target_size)
        counts[action["action"]] += 1

    next_id = 1 + max(
        (
            int(item["id"].removeprefix("exp-"))
            for item in items
            if item["id"].removeprefix("exp-").isdigit()
        ),
        default=0,
    )
    return items, len(action_order), counts, next_id, False




def _export_pool(pool_path: Path, items: Sequence[Mapping[str, Any]]) -> None:
    active = _ordered_active(items)
    inactive = sorted(
        (item for item in items if item.get("active") is not True),
        key=lambda item: int(item["created_order"]),
    )
    ordered = [*active, *inactive]
    _atomic_json(
        pool_path,
        {
            "experiences": [
                {
                    "id": item["id"],
                    "text": item["text"],
                    "active": bool(item["active"]),
                }
                for item in ordered
            ]
        },
    )


def _write_working_set(
    path: Path,
    *,
    items: Sequence[Mapping[str, Any]],
    processed: int,
    total_candidates: int,
) -> None:
    _atomic_json(
        path,
        {
            "processed": processed,
            "total_candidates": total_candidates,
            "references": list(items),
        },
    )


def _operational_paths(config: Mapping[str, Any]) -> dict[str, Path]:
    pool_path = project_path(
        config,
        config.get("experience_pool", {}).get(
            "path", "experience_pool/experience_pool.json"
        ),
    )
    return {
        "pool": pool_path,
        "working": pool_path.with_name("weighted_reference_set.json"),
        "actions": pool_path.with_name("reference_actions.jsonl"),
        "evaluations": pool_path.with_name("reference_evaluations.jsonl"),
        "normalizations": pool_path.with_name("consolidation_normalizations.jsonl"),
        "legacy_actions": pool_path.with_name("consolidation_actions.jsonl"),
        "legacy_audits": pool_path.with_name("consolidation_audits.jsonl"),
        "legacy_manual": pool_path.with_name("consolidation_manual_deactivations.jsonl"),
    }


def _archive_operational_outputs(
    paths: Mapping[str, Path],
    *,
    include_normalizations: bool = False,
) -> Path | None:
    keys = [
        "pool",
        "working",
        "actions",
        "evaluations",
        "legacy_actions",
        "legacy_audits",
        "legacy_manual",
    ]
    if include_normalizations:
        keys.append("normalizations")
    movable = [paths[key] for key in keys if paths[key].exists()]
    movable.extend(sorted(paths["pool"].parent.glob("dedup_checkpoint_*")))
    evaluation_dir = paths["pool"].parent / "reference_evaluation"
    if evaluation_dir.exists():
        movable.append(evaluation_dir)
    if not movable:
        return None
    archive = paths["pool"].parent / (
        "archive_pre_weighted_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    archive.mkdir(parents=True, exist_ok=False)
    for source in movable:
        os.replace(source, archive / source.name)
    return archive


def _save_normalization(
    path: Path,
    candidate: Mapping[str, Any],
    normalization: Mapping[str, Any],
    call: Mapping[str, Any],
) -> None:
    append_jsonl(
        path,
        {
            "sample_id": candidate["sample_id"],
            "method": NORMALIZATION_METHOD,
            "gold_label": candidate["gold_label"],
            "normalization": dict(normalization),
            "call": dict(call),
        },
    )


def _segment_paths_and_candidates(
    config: Mapping[str, Any],
    limit: int | None,
) -> tuple[dict[str, Path], list[dict[str, Any]], int]:
    extraction_dir = project_path(
        config,
        config.get("experience_extraction", {}).get(
            "output_path", "experience_extraction"
        ),
    )
    comparative_path = extraction_dir / "comparative_experiences.jsonl"
    all_candidates = _load_candidates(comparative_path)
    candidate_sample_path = config.get("experience_pool", {}).get(
        "candidate_sample_path"
    )
    if candidate_sample_path:
        group_path = project_path(config, candidate_sample_path)
        seen_ids: set[str] = set()
        for line_number, record in read_jsonl(group_path):
            sample_id = record.get("sample_id")
            if not isinstance(sample_id, str) or not sample_id:
                raise ConsolidationError(
                    f"{group_path}:{line_number}: missing sample_id"
                )
            if sample_id in seen_ids:
                raise ConsolidationError(
                    f"{group_path}:{line_number}: duplicate sample_id {sample_id!r}"
                )
            seen_ids.add(sample_id)
        comparative_ids = set(load_records_by_id(comparative_path))
        missing = seen_ids - comparative_ids
        if missing:
            raise ConsolidationError(
                f"{len(missing)} group samples have no comparative extraction record"
            )
        all_candidates = [
            candidate
            for candidate in all_candidates
            if candidate["source_sample_id"] in seen_ids
        ]
    if limit is not None and limit <= 0:
        raise ConsolidationError("consolidation limit must be positive")
    candidates = all_candidates if limit is None else all_candidates[:limit]
    return _operational_paths(config), candidates, len(all_candidates)


def run_weighted_consolidation_segment(
    config: Mapping[str, Any],
    *,
    limit: int | None = None,
) -> dict[str, Any]:
    paths, candidates, total_candidates = _segment_paths_and_candidates(config, limit)
    pool_config = config.get("experience_pool", {})
    reference_config = pool_config.get("references", {})
    generation = pool_config.get("generation", {})
    target_size = int(reference_config.get("target_size", 100))
    if target_size <= 0:
        raise ConsolidationError("experience_pool.references.target_size must be positive")
    max_input_tokens = int(generation.get("max_input_tokens", 24576))
    max_experience_words = int(generation.get("max_experience_words", 70))
    max_accepted_experience_words = int(
        generation.get("max_accepted_experience_words", max_experience_words)
    )
    if max_accepted_experience_words < max_experience_words:
        raise ConsolidationError(
            "experience_pool.generation.max_accepted_experience_words must be "
            "greater than or equal to max_experience_words"
        )
    json_config = generation.get("json", {})
    json_max_attempts = int(json_config.get("max_attempts", 3))
    failure_log_path = project_path(
        config,
        json_config.get(
            "failure_log_path", "logs/consolidation_json_failures.jsonl"
        ),
    )
    normalizations = _load_normalization_cache(
        paths["normalizations"],
        candidates,
        max_experience_words=max_experience_words,
    )
    items, processed, counts, next_id, _legacy_final = _replay_state(
        candidates,
        paths["actions"],
        paths["evaluations"],
        target_size=target_size,
        max_experience_words=max_accepted_experience_words,
    )
    _export_pool(paths["pool"], items)
    _write_working_set(
        paths["working"],
        items=items,
        processed=processed,
        total_candidates=len(candidates),
    )
    if processed == len(candidates):
        return {
            "status": "consolidation_complete",
            "processed": processed,
            "candidates": len(candidates),
            "total_candidates": total_candidates,
            "active": len(_active(items)),
            **{action.lower(): counts[action] for action in LOG_ACTIONS},
        }

    teacher_path = str(Path(get_required(config, "models.teacher.path")).resolve())
    show_progress = bool(generation.get("show_progress", True))
    generator = _build_consolidation_generator(
        teacher_path,
        generation,
        max_input_tokens=max_input_tokens,
    )
    pipeline = generation.get("pipeline", {})
    pipeline_enabled = bool(pipeline.get("enabled", False))
    prefetch = int(pipeline.get("prefetch_candidates", 8))
    normalization_executor: ProcessPoolExecutor | None = None
    normalization_futures: dict[int, Future[Any]] = {}
    next_submit = processed
    normalization_worker_started = False

    def submit_normalizations() -> None:
        nonlocal next_submit, normalization_worker_started
        if normalization_executor is None:
            return
        while next_submit < len(candidates) and len(normalization_futures) < prefetch:
            index = next_submit
            next_submit += 1
            candidate = candidates[index]
            if candidate["sample_id"] in normalizations:
                continue
            normalization_worker_started = True
            normalization_futures[index] = normalization_executor.submit(
                _run_candidate_normalization_worker,
                candidate=candidate,
                generation=generation,
                max_input_tokens=max_input_tokens,
                max_experience_words=max_experience_words,
                json_max_attempts=json_max_attempts,
                failure_log_path=failure_log_path,
                teacher_path=teacher_path,
            )

    if pipeline_enabled:
        if prefetch <= 0:
            raise ConsolidationError("normalization prefetch must be positive")
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if visible is not None and len([value for value in visible.split(",") if value.strip()]) < 2:
            raise ConsolidationError(
                "weighted consolidation pipeline requires CUDA_VISIBLE_DEVICES=0,1"
            )
        normalization_max_new_tokens = int(
            generation.get("normalization", {}).get(
                "max_new_tokens", generation.get("max_new_tokens", 512)
            )
        )
        normalization_chat = generation.get("normalization", {}).get(
            "chat_template_kwargs", generation.get("chat_template_kwargs", {})
        )
        normalization_executor = ProcessPoolExecutor(
            max_workers=1,
            mp_context=get_context("spawn"),
            initializer=_initialize_normalization_worker,
            initargs=(
                teacher_path,
                pipeline.get("normalization_device_map", "cuda:1"),
                normalization_max_new_tokens,
                bool(generation.get("deterministic", True)),
                normalization_chat,
                bool(generation.get("empty_cache_after_generate", False)),
            ),
        )
        submit_normalizations()

    progress = tqdm(
        total=len(candidates),
        initial=processed,
        disable=not show_progress,
        desc="Building weighted references",
        unit="candidate",
        dynamic_ncols=True,
    )
    try:
        for index in range(processed, len(candidates)):
            candidate = candidates[index]
            cached = normalizations.get(candidate["sample_id"])
            if cached is not None:
                normalization = dict(cached["normalization"])
                normalization_call = dict(cached["call"])
                prompt_tokens = 0
            elif normalization_executor is not None:
                future = normalization_futures.pop(index)
                normalization, normalization_call, prompt_tokens = future.result()
                _save_normalization(
                    paths["normalizations"],
                    candidate,
                    normalization,
                    normalization_call,
                )
                normalizations[candidate["sample_id"]] = {
                    "normalization": normalization,
                    "call": normalization_call,
                }
                submit_normalizations()
            else:
                normalization, normalization_call, prompt_tokens = (
                    _run_candidate_normalization(
                        generator,
                        candidate=candidate,
                        generation=generation,
                        max_input_tokens=max_input_tokens,
                        max_experience_words=max_experience_words,
                        json_max_attempts=json_max_attempts,
                        failure_log_path=failure_log_path,
                        teacher_path=teacher_path,
                    )
                )
                _save_normalization(
                    paths["normalizations"],
                    candidate,
                    normalization,
                    normalization_call,
                )
                normalizations[candidate["sample_id"]] = {
                    "normalization": normalization,
                    "call": normalization_call,
                }

            calls: dict[str, Any] = {
                "normalization": normalization_call,
                "comparison": None,
                "merge_compression": None,
                "merge": None,
                "boundary_merge": None,
                "boundary_compression": None,
            }
            if not normalization["valid"]:
                decision = {
                    "candidate_label": candidate["gold_label"],
                    "action": "SKIP",
                    "target_id": None,
                    "experience": None,
                    "mechanism": None,
                    "exclusion_boundary": None,
                    "modification_scope": None,
                    "assigned_id": None,
                    "rationale": normalization["rationale"],
                }
            else:
                active_ids = _active_ids(items)
                messages = [
                    {"role": "system", "content": CONSOLIDATION_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": json.dumps(
                            {
                                "active_experience_pool": _active_view(items),
                                "candidate": {
                                    "gold_label": candidate["gold_label"],
                                    "mechanism": normalization["mechanism"],
                                    "exclusion_boundary": normalization[
                                        "exclusion_boundary"
                                    ],
                                },
                            },
                            ensure_ascii=True,
                            sort_keys=False,
                        ),
                    },
                ]
                relation, comparison_call, comparison_tokens = _run_json_stage(
                    generator,
                    stage_name="comparison",
                    artifact="weighted_reference_comparison",
                    sample_id=candidate["sample_id"],
                    messages=messages,
                    parser=lambda raw: _parse_weighted_relation(
                        raw,
                        active_ids,
                        candidate["gold_label"],
                    ),
                    generation=generation,
                    max_input_tokens=max_input_tokens,
                    json_max_attempts=json_max_attempts,
                    failure_log_path=failure_log_path,
                    teacher_path=teacher_path,
                    json_schema=_weighted_relation_schema(
                        active_ids,
                        candidate["gold_label"],
                    ),
                )
                calls["comparison"] = comparison_call
                prompt_tokens = max(prompt_tokens, comparison_tokens)

                mechanism = None
                exclusion_boundary = None
                experience = None
                modification_scope = None
                if relation["action"] == "ADD":
                    mechanism = normalization["mechanism"]
                    exclusion_boundary = normalization["exclusion_boundary"]
                    experience = _compose_experience(
                        mechanism,
                        exclusion_boundary,
                        max_experience_words=max_experience_words,
                    )
                    modification_scope = "FULL"
                elif relation["action"] == "MODIFY":
                    target = _find_active(items, relation["target_id"])
                    if candidate["gold_label"] == "sarcastic":
                        merge_messages = [
                            {
                                "role": "system",
                                "content": EXPERIENCE_MERGE_SYSTEM_PROMPT,
                            },
                            {
                                "role": "user",
                                "content": json.dumps(
                                    {
                                        "target": {
                                            "id": target["id"],
                                            "mechanism": target["mechanism"],
                                            "exclusion_boundary": target[
                                                "exclusion_boundary"
                                            ],
                                        },
                                        "candidate": {
                                            "mechanism": normalization["mechanism"],
                                            "exclusion_boundary": normalization[
                                                "exclusion_boundary"
                                            ],
                                        },
                                        "modification_rationale": relation[
                                            "rationale"
                                        ],
                                    },
                                    ensure_ascii=True,
                                    sort_keys=True,
                                ),
                            },
                        ]
                        merge, merge_call, merge_tokens = _run_json_stage(
                            generator,
                            stage_name="merge",
                            artifact="weighted_reference_merge",
                            sample_id=candidate["sample_id"],
                            messages=merge_messages,
                            parser=lambda raw: _parse_merge(
                                raw,
                                max_experience_words=max(
                                    1000, max_experience_words
                                ),
                            ),
                            generation=generation,
                            max_input_tokens=max_input_tokens,
                            json_max_attempts=json_max_attempts,
                            failure_log_path=failure_log_path,
                            teacher_path=teacher_path,
                        )
                        calls["merge"] = merge_call
                        prompt_tokens = max(prompt_tokens, merge_tokens)
                        if (
                            len(merge["experience"].split())
                            > max_accepted_experience_words
                        ):
                            compression_errors: list[str] = []
                            compression_generation_attempts = 0
                            attempted_targets: list[int] = []
                            compression_attempts = (
                                (
                                    95,
                                    "Merge synonymous clauses and replace surface "
                                    "lists with shared mechanism categories.",
                                ),
                                (
                                    90,
                                    "Reconstruct the rule from its meaning, retaining "
                                    "distinct conditions as compact qualifiers instead "
                                    "of enumerations.",
                                ),
                                (
                                    80,
                                    "Express the complete core mechanism and nearest "
                                    "exclusion boundary compactly while preserving every "
                                    "distinct valid condition.",
                                ),
                            )
                            for compression_target, compression_strategy in compression_attempts:
                                if compression_target > max_experience_words:
                                    continue
                                attempted_targets.append(compression_target)
                                mechanism_budget, boundary_budget = (
                                    _compression_field_word_budgets(
                                        merge["mechanism"],
                                        merge["exclusion_boundary"],
                                        compression_target,
                                    )
                                )
                                compression_messages = [
                                    {
                                        "role": "system",
                                        "content": experience_compression_system_prompt(
                                            target_maximum_words=compression_target,
                                            compression_strategy=compression_strategy,
                                        ),
                                    },
                                    {
                                        "role": "user",
                                        "content": json.dumps(
                                            {
                                                "source_target": {
                                                    "mechanism": target[
                                                        "mechanism"
                                                    ],
                                                    "exclusion_boundary": target[
                                                        "exclusion_boundary"
                                                    ],
                                                },
                                                "source_candidate": {
                                                    "mechanism": normalization[
                                                        "mechanism"
                                                    ],
                                                    "exclusion_boundary": normalization[
                                                        "exclusion_boundary"
                                                    ],
                                                },
                                                "modification_rationale": relation[
                                                    "rationale"
                                                ],
                                                "original_merged_word_counts": {
                                                    "mechanism": len(
                                                        merge["mechanism"].split()
                                                    ),
                                                    "exclusion_boundary": len(
                                                        merge[
                                                            "exclusion_boundary"
                                                        ].split()
                                                    ),
                                                },
                                                "maximum_mechanism_words": (
                                                    mechanism_budget
                                                ),
                                                "maximum_exclusion_boundary_words": (
                                                    boundary_budget
                                                ),
                                                "target_maximum_words": compression_target,
                                                "hard_maximum_words": max_experience_words,
                                            },
                                            ensure_ascii=True,
                                            sort_keys=True,
                                        ),
                                    },
                                ]
                                try:
                                    compressed, compression_call, compression_tokens = _run_json_stage(
                                        generator,
                                        stage_name="merge",
                                        artifact=(
                                            "weighted_reference_merge_compression_"
                                            f"{compression_target}"
                                        ),
                                        sample_id=candidate["sample_id"],
                                        messages=compression_messages,
                                        parser=lambda raw: _parse_compressed_merge(
                                            raw,
                                            accepted_maximum_words=(
                                                max_accepted_experience_words
                                            ),
                                            target_maximum_words=compression_target,
                                            maximum_mechanism_words=(
                                                mechanism_budget
                                            ),
                                            maximum_boundary_words=boundary_budget,
                                        ),
                                        generation=generation,
                                        max_input_tokens=max_input_tokens,
                                        json_max_attempts=json_max_attempts,
                                        failure_log_path=failure_log_path,
                                        teacher_path=teacher_path,
                                    )
                                except ConsolidationError as exc:
                                    compression_errors.append(str(exc))
                                    compression_generation_attempts += json_max_attempts
                                    continue
                                compression_generation_attempts += int(
                                    compression_call["attempts"]
                                )
                                prompt_tokens = max(
                                    prompt_tokens, compression_tokens
                                )
                                merge = compressed
                                break
                            else:
                                last_error = (
                                    compression_errors[-1]
                                    if compression_errors
                                    else "unknown compression failure"
                                )
                                raise ConsolidationError(
                                    "merge compression stopped at "
                                    f"{candidate['sample_id']} after "
                                    f"{compression_generation_attempts} invalid attempts; "
                                    "the relationship remains MODIFY and no action was "
                                    f"written. Last error: {last_error}"
                                )
                            calls["merge_compression"] = {
                                "attempts": compression_generation_attempts,
                                "target_word_limits": attempted_targets,
                                "field_word_limits": {
                                    "mechanism": mechanism_budget,
                                    "exclusion_boundary": boundary_budget,
                                },
                                "result": {
                                    "mechanism": merge["mechanism"],
                                    "exclusion_boundary": merge[
                                        "exclusion_boundary"
                                    ],
                                },
                                "errors": compression_errors,
                            }
                        mechanism = merge["mechanism"]
                        exclusion_boundary = merge["exclusion_boundary"]
                        experience = merge["experience"]
                        modification_scope = "FULL"
                    else:
                        mechanism_words = len(target["mechanism"].split())
                        maximum_boundary_words = max(
                            5, max_experience_words - mechanism_words
                        )
                        boundary_messages = [
                            {
                                "role": "system",
                                "content": EXCLUSION_BOUNDARY_MERGE_SYSTEM_PROMPT,
                            },
                            {
                                "role": "user",
                                "content": json.dumps(
                                    {
                                        "target_id": target["id"],
                                        "target_mechanism": target["mechanism"],
                                        "current_exclusion_boundary": target[
                                            "exclusion_boundary"
                                        ],
                                        "candidate_boundary_evidence": normalization[
                                            "exclusion_boundary"
                                        ],
                                        "hard_maximum_words": max_experience_words,
                                        "maximum_revised_boundary_words": (
                                            maximum_boundary_words
                                        ),
                                        "modification_rationale": relation[
                                            "rationale"
                                        ],
                                    },
                                    ensure_ascii=True,
                                    sort_keys=True,
                                ),
                            },
                        ]
                        boundary, boundary_call, boundary_tokens = _run_json_stage(
                            generator,
                            stage_name="merge",
                            artifact="weighted_reference_boundary_merge",
                            sample_id=candidate["sample_id"],
                            messages=boundary_messages,
                            parser=lambda raw: _parse_boundary_merge(
                                raw,
                                mechanism=target["mechanism"],
                                max_experience_words=max(
                                    1000, max_experience_words
                                ),
                            ),
                            generation=generation,
                            max_input_tokens=max_input_tokens,
                            json_max_attempts=json_max_attempts,
                            failure_log_path=failure_log_path,
                            teacher_path=teacher_path,
                            json_schema={
                                "type": "object",
                                "properties": {
                                    "revised_exclusion_boundary": {
                                        "type": "string",
                                        "minLength": 1,
                                    }
                                },
                                "required": ["revised_exclusion_boundary"],
                                "additionalProperties": False,
                            },
                        )
                        calls["boundary_merge"] = boundary_call
                        prompt_tokens = max(prompt_tokens, boundary_tokens)

                        if (
                            len(boundary["experience"].split())
                            > max_accepted_experience_words
                        ):
                            if max_accepted_experience_words - mechanism_words < 5:
                                raise ConsolidationError(
                                    f"boundary compression stopped at {candidate['sample_id']}: "
                                    "the immutable mechanism leaves fewer than 5 words for "
                                    "a valid exclusion boundary; the relationship remains "
                                    "MODIFY and no action was written"
                                )
                            boundary_compression_errors: list[str] = []
                            boundary_compression_attempts = 0
                            attempted_boundary_total_targets: list[int] = []
                            attempted_boundary_targets: list[int] = []
                            boundary_targets = (
                                (
                                    95,
                                    4,
                                    10,
                                    "Cluster all source cases into at most four causal "
                                    "countercondition categories; state each category once.",
                                ),
                                (
                                    90,
                                    3,
                                    15,
                                    "Cluster all source cases into at most three "
                                    "mechanism-level countercondition categories.",
                                ),
                                (
                                    80,
                                    2,
                                    20,
                                    "Express the same semantic coverage with at most two "
                                    "high-level causal countercondition categories.",
                                ),
                            )
                            abstraction_results: list[dict[str, Any]] = []
                            effective_total_targets: list[int] = []
                            for total_target, maximum_categories, tolerance_margin, compression_strategy in boundary_targets:
                                boundary_target = _boundary_compression_word_budget(
                                    mechanism_words=mechanism_words,
                                    profile_total_words=total_target,
                                    accepted_maximum_words=(
                                        max_accepted_experience_words
                                    ),
                                    tolerance_margin_words=tolerance_margin,
                                )
                                effective_total_target = (
                                    mechanism_words + boundary_target
                                )
                                if (
                                    total_target > max_experience_words
                                    or total_target in attempted_boundary_total_targets
                                ):
                                    continue
                                attempted_boundary_total_targets.append(total_target)
                                attempted_boundary_targets.append(boundary_target)
                                effective_total_targets.append(
                                    effective_total_target
                                )
                                abstraction_messages = [
                                    {
                                        "role": "system",
                                        "content": exclusion_boundary_abstraction_system_prompt(
                                            maximum_categories=maximum_categories,
                                            compression_strategy=compression_strategy,
                                        ),
                                    },
                                    {
                                        "role": "user",
                                        "content": json.dumps(
                                            {
                                                "source_target": {
                                                    "mechanism": target[
                                                        "mechanism"
                                                    ],
                                                    "exclusion_boundary": target[
                                                        "exclusion_boundary"
                                                    ],
                                                },
                                                "source_candidate": {
                                                    "gold_label": candidate[
                                                        "gold_label"
                                                    ],
                                                    "exclusion_boundary": normalization[
                                                        "exclusion_boundary"
                                                    ],
                                                },
                                                "modification_rationale": relation[
                                                    "rationale"
                                                ],
                                            },
                                            ensure_ascii=True,
                                            sort_keys=True,
                                        ),
                                    },
                                ]
                                try:
                                    abstraction, abstraction_call, abstraction_tokens = _run_json_stage(
                                        generator,
                                        stage_name="merge",
                                        artifact=(
                                            "weighted_reference_boundary_abstraction_"
                                            f"{maximum_categories}"
                                        ),
                                        sample_id=candidate["sample_id"],
                                        messages=abstraction_messages,
                                        parser=lambda raw: _parse_boundary_abstraction(
                                            raw,
                                            maximum_categories=maximum_categories,
                                        ),
                                        generation=generation,
                                        max_input_tokens=max_input_tokens,
                                        json_max_attempts=json_max_attempts,
                                        failure_log_path=failure_log_path,
                                        teacher_path=teacher_path,
                                        json_schema=_boundary_abstraction_schema(
                                            maximum_categories
                                        ),
                                    )
                                except ConsolidationError as exc:
                                    boundary_compression_errors.append(str(exc))
                                    boundary_compression_attempts += json_max_attempts
                                    continue
                                boundary_compression_attempts += int(
                                    abstraction_call["attempts"]
                                )
                                prompt_tokens = max(
                                    prompt_tokens, abstraction_tokens
                                )
                                abstraction_results.append(
                                    {
                                        "target_words": total_target,
                                        "effective_total_words": (
                                            effective_total_target
                                        ),
                                        "maximum_categories": maximum_categories,
                                        "categories": abstraction[
                                            "causal_categories"
                                        ],
                                        "attempts": abstraction_call["attempts"],
                                    }
                                )
                                compression_messages = [
                                    {
                                        "role": "system",
                                        "content": exclusion_boundary_compression_system_prompt(
                                            target_maximum_words=boundary_target,
                                            compression_strategy=compression_strategy,
                                        ),
                                    },
                                    {
                                        "role": "user",
                                        "content": json.dumps(
                                            {
                                                "immutable_target_mechanism": target[
                                                    "mechanism"
                                                ],
                                                "causal_categories": abstraction[
                                                    "causal_categories"
                                                ],
                                                "modification_rationale": relation[
                                                    "rationale"
                                                ],
                                                "target_maximum_words": boundary_target,
                                                "hard_maximum_words": max_experience_words,
                                            },
                                            ensure_ascii=True,
                                            sort_keys=True,
                                        ),
                                    },
                                ]
                                try:
                                    compressed_boundary, compression_call, compression_tokens = _run_json_stage(
                                        generator,
                                        stage_name="merge",
                                        artifact=(
                                            "weighted_reference_boundary_compression_"
                                            f"{boundary_target}"
                                        ),
                                        sample_id=candidate["sample_id"],
                                        messages=compression_messages,
                                        parser=lambda raw: _parse_compressed_boundary_merge(
                                            raw,
                                            mechanism=target["mechanism"],
                                            accepted_maximum_words=(
                                                max_accepted_experience_words
                                            ),
                                            target_total_words=total_target,
                                            target_boundary_words=boundary_target,
                                        ),
                                        generation=generation,
                                        max_input_tokens=max_input_tokens,
                                        json_max_attempts=json_max_attempts,
                                        failure_log_path=failure_log_path,
                                        teacher_path=teacher_path,
                                        json_schema={
                                            "type": "object",
                                            "properties": {
                                                "revised_exclusion_boundary": {
                                                    "type": "string",
                                                    "minLength": 1,
                                                }
                                            },
                                            "required": [
                                                "revised_exclusion_boundary"
                                            ],
                                            "additionalProperties": False,
                                        },
                                    )
                                except ConsolidationError as exc:
                                    boundary_compression_errors.append(str(exc))
                                    boundary_compression_attempts += json_max_attempts
                                    continue
                                boundary_compression_attempts += int(
                                    compression_call["attempts"]
                                )
                                prompt_tokens = max(
                                    prompt_tokens, compression_tokens
                                )
                                boundary = compressed_boundary
                                break
                            else:
                                last_error = (
                                    boundary_compression_errors[-1]
                                    if boundary_compression_errors
                                    else "unknown compression failure"
                                )
                                raise ConsolidationError(
                                    "boundary compression stopped at "
                                    f"{candidate['sample_id']} after "
                                    f"{boundary_compression_attempts} invalid attempts; "
                                    "the relationship remains MODIFY and no action was "
                                    f"written. Last error: {last_error}"
                                )
                            calls["boundary_compression"] = {
                                "attempts": boundary_compression_attempts,
                                "target_word_limits": (
                                    attempted_boundary_total_targets
                                ),
                                "target_boundary_word_limits": (
                                    attempted_boundary_targets
                                ),
                                "effective_total_word_limits": (
                                    effective_total_targets
                                ),
                                "abstractions": abstraction_results,
                                "result": {
                                    "exclusion_boundary": boundary[
                                        "exclusion_boundary"
                                    ]
                                },
                                "errors": boundary_compression_errors,
                            }

                        mechanism = target["mechanism"]
                        exclusion_boundary = boundary["exclusion_boundary"]
                        experience = boundary["experience"]
                        modification_scope = "BOUNDARY"

                decision = {
                    "candidate_label": candidate["gold_label"],
                    "action": relation["action"],
                    "target_id": relation["target_id"],
                    "experience": experience,
                    "mechanism": mechanism,
                    "exclusion_boundary": exclusion_boundary,
                    "modification_scope": modification_scope,
                    "assigned_id": (
                        f"exp-{next_id:06d}"
                        if relation["action"] == "ADD"
                        else None
                    ),
                    "rationale": relation["rationale"],
                }
                decision = _deduplicate_generated_add(decision, items)
                decision = {
                    "candidate_label": candidate["gold_label"],
                    "experience": None,
                    "mechanism": None,
                    "exclusion_boundary": None,
                    "modification_scope": None,
                    "assigned_id": None,
                    **decision,
                }

            parsed = _parse_action_record(
                decision,
                _active_ids(items),
                gold_label=candidate["gold_label"],
                max_experience_words=max_accepted_experience_words,
            )
            if parsed["action"] == "ADD":
                next_id += 1
            action_record = {
                "sample_id": candidate["sample_id"],
                "method": REFERENCE_FLOW_METHOD,
                "candidate_label": candidate["gold_label"],
                **parsed,
                "calls": calls,
                "attempts": sum(
                    call["attempts"]
                    for call in calls.values()
                    if isinstance(call, Mapping)
                ),
            }
            upsert_jsonl(paths["actions"], action_record)
            _apply_structural_action(
                items,
                parsed,
                max_experience_words=max_accepted_experience_words,
            )
            _enforce_reference_limit(items, target_size)
            counts[parsed["action"]] += 1
            processed = index + 1
            _export_pool(paths["pool"], items)
            _write_working_set(
                paths["working"],
                items=items,
                processed=processed,
                total_candidates=len(candidates),
            )
            progress.update(1)
            progress.set_postfix(
                active=len(_active(items)),
                prompt_tokens=prompt_tokens,
                ADD=counts["ADD"],
                MODIFY=counts["MODIFY"],
                UPVOTE=counts["UPVOTE"],
                DOWNVOTE=counts["DOWNVOTE"],
                SKIP=counts["SKIP"],
            )
    finally:
        if normalization_executor is not None:
            if normalization_worker_started:
                try:
                    normalization_executor.submit(_close_normalization_worker).result()
                except Exception:
                    pass
            normalization_executor.shutdown(wait=True, cancel_futures=True)
        generator.close()
        progress.close()

    return {
        "status": "consolidation_complete",
        "processed": processed,
        "candidates": len(candidates),
        "total_candidates": total_candidates,
        "active": len(_active(items)),
        **{action.lower(): counts[action] for action in LOG_ACTIONS},
    }


def _isolated_stage_worker(
    queue: Any,
    stage: str,
    config: Mapping[str, Any],
    limit: int | None,
) -> None:
    try:
        if stage != "segment":
            raise RuntimeError(f"unknown isolated reference stage {stage!r}")
        result = run_weighted_consolidation_segment(config, limit=limit)
    except BaseException as exc:
        queue.put(
            {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
        )
    else:
        queue.put({"ok": True, "result": result})


def _run_isolated_stage(
    stage: str,
    config: Mapping[str, Any],
    limit: int | None,
) -> dict[str, Any]:
    context = get_context("spawn")
    queue = context.Queue()
    process = context.Process(
        target=_isolated_stage_worker,
        args=(queue, stage, dict(config), limit),
    )
    process.start()
    process.join()
    try:
        message = queue.get(timeout=5)
    except Empty as exc:
        raise ConsolidationError(
            f"isolated {stage} process exited with code {process.exitcode} "
            "without returning a result"
        ) from exc
    if not message.get("ok"):
        raise ConsolidationError(
            f"isolated {stage} failed: {message.get('error')}\n"
            f"{message.get('traceback', '')}"
        )
    if process.exitcode != 0:
        raise ConsolidationError(
            f"isolated {stage} process exited with code {process.exitcode}"
        )
    return dict(message["result"])


def run_weighted_reference_pipeline(
    config: Mapping[str, Any],
    *,
    limit: int | None = None,
    fresh: bool = False,
    rebuild_from_normalizations: bool = False,
) -> dict[str, Any]:
    if fresh and rebuild_from_normalizations:
        raise ConsolidationError(
            "fresh and rebuild_from_normalizations are mutually exclusive"
        )
    paths = _operational_paths(config)
    archive = None
    if fresh:
        archive = _archive_operational_outputs(
            paths,
            include_normalizations=True,
        )
    elif rebuild_from_normalizations:
        archive = _archive_operational_outputs(paths)
    elif (
        not paths["actions"].exists()
        and any(
            paths[key].exists()
            for key in ("pool", "legacy_actions", "legacy_audits", "legacy_manual")
        )
    ):
        raise ConsolidationError(
            "legacy consolidation outputs exist; rerun once with --fresh"
        )

    isolate = bool(
        config.get("experience_pool", {})
        .get("references", {})
        .get("isolate_stages", True)
    )
    if isolate:
        completed = _run_isolated_stage("segment", config, limit)
    else:
        completed = run_weighted_consolidation_segment(config, limit=limit)
    return {
        **completed,
        "rebuild_archive": str(archive) if archive is not None else None,
    }
