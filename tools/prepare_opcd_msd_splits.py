#!/usr/bin/env python3
"""Prepare PACE experience-extraction, V_score, and OPD train JSONL files.

The effective gold-reference artifact is used only to choose clean extraction
samples. V_score is the complete official validation split of MMSD2.0 and
SarcNet. The OPD training file contains the remaining training samples after
removing the extraction samples and all rescued-reference sample IDs.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EFFECTIVE = PROJECT_ROOT / "data" / "gold_references" / "effective_gold_references.jsonl"
DEFAULT_RESCUED = PROJECT_ROOT / "data" / "gold_references" / "rescued_gold_references.jsonl"
DEFAULT_MMSD_ROOT = PROJECT_ROOT / "datasets" / "MMSD2.0"
DEFAULT_SARCNET_ROOT = PROJECT_ROOT / "datasets" / "sarcnet"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "opcd_msd_splits"

LABELS = {"sarcastic", "non-sarcastic"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(value)
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def normalize_label(value: Any) -> str:
    if isinstance(value, str):
        key = value.strip().lower().replace("_", "-")
        if key in {"sarcastic", "sarcasm", "ironic", "1"}:
            return "sarcastic"
        if key in {"non-sarcastic", "nonsarcastic", "not-sarcastic", "0"}:
            return "non-sarcastic"
    elif value in {0, False}:
        return "non-sarcastic"
    elif value in {1, True}:
        return "sarcastic"
    raise ValueError(f"unsupported label: {value!r}")


def sample_id_for_mmsd(image_id: Any) -> str:
    return f"mmsd2:na:{image_id}"


def load_mmsd_split(root: Path, variant: str, split: str) -> dict[str, dict[str, Any]]:
    if variant == "v2":
        data_dir = root / "data" / "text_json_final"
    elif variant == "clean":
        data_dir = root / "data" / "text_json_clean"
    else:
        raise ValueError(f"unsupported MMSD variant: {variant}")

    path = data_dir / ("valid.json" if split == "validation" else f"{split}.json")
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"{path}: expected a JSON array")

    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        image_id = row["image_id"]
        sample_id = sample_id_for_mmsd(image_id)
        if sample_id in result:
            raise ValueError(f"duplicate sample_id in {path}: {sample_id}")
        image_path = root / "data" / "dataset_image" / f"{image_id}.jpg"
        result[sample_id] = {
            "sample_id": sample_id,
            "image_path": str(image_path.resolve()),
            "text": str(row["text"]),
            "label": normalize_label(row["label"]),
            "source": "mmsd2",
            "language": None,
        }
    return result


def load_sarcnet_split(root: Path, language: str, split: str) -> dict[str, dict[str, Any]]:
    path = root / "data" / language / f"{split}.jsonl"
    if not path.is_file():
        raise FileNotFoundError(path)
    result: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        sample_id = row["sample_id"]
        if sample_id in result:
            raise ValueError(f"duplicate sample_id in {path}: {sample_id}")
        image_path = Path(str(row["image_path"]))
        if not image_path.is_absolute():
            image_path = path.parent / image_path
        result[sample_id] = {
            "sample_id": sample_id,
            "image_path": str(image_path.resolve()),
            "text": str(row["text"]),
            "label": normalize_label(row.get("multi_label", row["label"])),
            "source": "sarcnet",
            "language": language,
            "text_label": row.get("text_label"),
            "image_label": row.get("image_label"),
        }
    return result


def load_gold_records(path: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError(f"{path}: missing sample_id")
        if sample_id in result:
            raise ValueError(f"duplicate sample_id in {path}: {sample_id}")
        result[sample_id] = row
    return result


def reference_is_schema_usable(gold_record: dict[str, Any]) -> bool:
    """Return whether an effective reference can be embedded as strict reasoning."""
    analysis = gold_record.get("analysis")
    required = {"visual_evidence", "textual_evidence", "explanation", "label"}
    if not isinstance(analysis, dict) or set(analysis) != required:
        return False
    for field in ("visual_evidence", "textual_evidence", "explanation"):
        value = analysis[field]
        if not isinstance(value, str) or not value.strip():
            return False
        # PACE requires English narrative reasoning for model-side parsing.
        letters = sum(char.isascii() and char.isalpha() for char in value)
        non_ascii_letters = sum((not char.isascii()) and char.isalpha() for char in value)
        if not letters or letters < non_ascii_letters:
            return False
    try:
        normalize_label(analysis["label"])
    except ValueError:
        return False
    return True


def count_schema_usable(records: Iterable[dict[str, Any]]) -> int:
    return sum(reference_is_schema_usable(record) for record in records)


def make_extraction_record(
    sample: dict[str, Any], gold_record: dict[str, Any], selection_index: int
) -> dict[str, Any]:
    analysis = gold_record.get("analysis")
    required = {"visual_evidence", "textual_evidence", "explanation", "label"}
    if not isinstance(analysis, dict) or set(analysis) != required:
        raise ValueError(f"invalid effective reference for {sample['sample_id']}")
    reference = dict(analysis)
    if normalize_label(reference["label"]) != sample["label"]:
        raise ValueError(f"reference label mismatch for {sample['sample_id']}")
    return {
        **sample,
        "reference_reasoning": reference,
        "split": "experience_extraction",
        "selection_index": selection_index,
    }


def select_stratified(
    effective: dict[str, dict[str, Any]],
    train_records: dict[str, dict[str, Any]],
    rescued_ids: set[str],
    rng: random.Random,
    quotas: list[tuple[str, str | None, str, int]],
    exclude_rescued: bool,
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str | None, str], list[str]] = defaultdict(list)
    for sample_id, gold in effective.items():
        source = gold.get("source")
        language = gold.get("language")
        if source not in {"mmsd2", "sarcnet"}:
            continue
        if gold.get("reference_valid") is not True:
            continue
        if not reference_is_schema_usable(gold):
            continue
        if exclude_rescued and sample_id in rescued_ids:
            continue
        if sample_id not in train_records:
            raise ValueError(f"effective reference is not in the expected train split: {sample_id}")
        label = normalize_label(gold.get("gold_label"))
        groups[(source, language, label)].append(sample_id)

    selected_ids: list[str] = []
    for source, language, label, count in quotas:
        key = (source, language, label)
        candidates = groups[key]
        if len(candidates) < count:
            raise ValueError(
                f"not enough eligible samples for {key}: requested {count}, available {len(candidates)}"
            )
        chosen = rng.sample(candidates, count)
        selected_ids.extend(chosen)

    if len(selected_ids) != len(set(selected_ids)):
        raise ValueError("extraction quotas selected duplicate sample IDs")
    rng.shuffle(selected_ids)
    return [
        make_extraction_record(train_records[sample_id], effective[sample_id], index)
        for index, sample_id in enumerate(selected_ids, 1)
    ]


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_group: Counter[str] = Counter()
    labels: Counter[str] = Counter()
    for row in rows:
        source = row["source"]
        language = row.get("language") or "na"
        by_group[f"{source}:{language}"] += 1
        labels[f"{source}:{language}:{row['label']}"] += 1
    return {
        "total": len(rows),
        "by_group": dict(sorted(by_group.items())),
        "by_group_label": dict(sorted(labels.items())),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mmsd-root", default=DEFAULT_MMSD_ROOT)
    parser.add_argument("--sarcnet-root", default=DEFAULT_SARCNET_ROOT)
    parser.add_argument("--mmsd-variant", choices=("v2", "clean"), default="v2")
    parser.add_argument("--effective-gold", default=DEFAULT_EFFECTIVE)
    parser.add_argument("--rescued-gold", default=DEFAULT_RESCUED)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mmsd-extract", type=int, default=4000)
    parser.add_argument("--sarcnet-en-extract", type=int, default=300)
    parser.add_argument("--sarcnet-zh-extract", type=int, default=800)
    parser.add_argument(
        "--exclude-rescued-from-extraction",
        action="store_true",
        help="also exclude the 649 rescued IDs from extraction; default keeps their corrected effective references",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mmsd_root = Path(args.mmsd_root).expanduser().resolve()
    sarcnet_root = Path(args.sarcnet_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    mmsd_train = load_mmsd_split(mmsd_root, args.mmsd_variant, "train")
    mmsd_val = load_mmsd_split(mmsd_root, args.mmsd_variant, "validation")
    sarc_train = {}
    sarc_val = {}
    for language in ("en", "zh"):
        sarc_train.update(load_sarcnet_split(sarcnet_root, language, "train"))
        sarc_val.update(load_sarcnet_split(sarcnet_root, language, "validation"))

    train_records = {**mmsd_train, **sarc_train}
    validation_records = {**mmsd_val, **sarc_val}
    if set(train_records) & set(validation_records):
        raise ValueError("train/validation sample ID overlap detected")

    effective = load_gold_records(Path(args.effective_gold).expanduser().resolve())
    rescued = load_gold_records(Path(args.rescued_gold).expanduser().resolve())
    rescued_ids = set(rescued)

    quotas = [
        ("mmsd2", None, "sarcastic", args.mmsd_extract // 2),
        ("mmsd2", None, "non-sarcastic", args.mmsd_extract - args.mmsd_extract // 2),
        ("sarcnet", "en", "sarcastic", args.sarcnet_en_extract // 2),
        ("sarcnet", "en", "non-sarcastic", args.sarcnet_en_extract - args.sarcnet_en_extract // 2),
        ("sarcnet", "zh", "sarcastic", args.sarcnet_zh_extract // 2),
        ("sarcnet", "zh", "non-sarcastic", args.sarcnet_zh_extract - args.sarcnet_zh_extract // 2),
    ]
    if any(count <= 0 for _, _, _, count in quotas):
        raise ValueError("each extraction quota must be positive")

    extraction = select_stratified(
        effective,
        train_records,
        rescued_ids,
        random.Random(args.seed),
        quotas,
        exclude_rescued=args.exclude_rescued_from_extraction,
    )
    extraction_ids = {row["sample_id"] for row in extraction}

    missing_images = [
        row["image_path"]
        for row in [*extraction, *validation_records.values(), *train_records.values()]
        if not Path(row["image_path"]).is_file()
    ]
    if missing_images:
        preview = "\n".join(missing_images[:10])
        raise FileNotFoundError(f"missing {len(missing_images)} image files; first paths:\n{preview}")

    excluded_ids = extraction_ids | (rescued_ids & set(train_records))
    opd_train = [row for sample_id, row in sorted(train_records.items()) if sample_id not in excluded_ids]
    v_score = [row | {"split": "validation_score"} for _, row in sorted(validation_records.items())]

    if extraction_ids & set(validation_records):
        raise ValueError("extraction and V_score overlap detected")
    if set(row["sample_id"] for row in opd_train) & excluded_ids:
        raise ValueError("excluded sample leaked into OPD train")

    write_jsonl(output_dir / "experience_extraction.jsonl", extraction)
    write_jsonl(output_dir / "v_score.jsonl", v_score)
    write_jsonl(output_dir / "opcd_train.jsonl", opd_train)

    effective_target = [
        row for row in effective.values() if row.get("source") in {"mmsd2", "sarcnet"}
    ]
    effective_schema_usable = count_schema_usable(effective_target)
    manifest = {
        "seed": args.seed,
        "mmsd_variant": args.mmsd_variant,
        "effective_gold": str(Path(args.effective_gold).expanduser().resolve()),
        "rescued_gold": str(Path(args.rescued_gold).expanduser().resolve()),
        "exclude_rescued_from_extraction": args.exclude_rescued_from_extraction,
        "counts": {
            "effective_target_records": len(effective_target),
            "effective_schema_usable_records": effective_schema_usable,
            "effective_schema_rejected_records": len(effective_target) - effective_schema_usable,
            "rescued_target_ids": len(rescued_ids & set(train_records)),
            "experience_extraction": summarize(extraction),
            "v_score": summarize(v_score),
            "opcd_train": summarize(opd_train),
            "opcd_train_excluded": len(excluded_ids),
        },
        "files": {
            "experience_extraction": "experience_extraction.jsonl",
            "v_score": "v_score.jsonl",
            "opcd_train": "opcd_train.jsonl",
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(manifest["counts"], ensure_ascii=False, indent=2, sort_keys=True))
    print(f"Wrote split files to {output_dir}")


if __name__ == "__main__":
    main()
