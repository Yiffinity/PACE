#!/usr/bin/env python3
"""Prepare shared samples and three pairwise train groups for experience learning."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from prepare_opcd_msd_splits import (
    load_mmsd_split,
    load_sarcnet_split,
    normalize_label,
    read_jsonl,
    summarize,
    write_jsonl,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DOCMSU_ROOT = PROJECT_ROOT / "datasets" / "DocMSU"
DEFAULT_MMSD_ROOT = PROJECT_ROOT / "datasets" / "MMSD2.0"
DEFAULT_SARCNET_ROOT = PROJECT_ROOT / "datasets" / "sarcnet"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "pairwise_experience_splits"
GROUP_SOURCES = {
    "mmsd2_docmsu": frozenset({"mmsd2", "docmsu"}),
    "mmsd2_sarcnet": frozenset({"mmsd2", "sarcnet"}),
    "docmsu_sarcnet": frozenset({"docmsu", "sarcnet"}),
}


def load_docmsu_train(root: Path) -> dict[str, dict[str, Any]]:
    train_path = root / "data" / "train.jsonl"
    annotation_path = root / "docmsu_all.json"
    annotations = json.loads(annotation_path.read_text(encoding="utf-8"))
    if not isinstance(annotations, dict):
        raise ValueError(f"{annotation_path}: expected a JSON object")

    records: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(train_path):
        original_id = row.get("sample_id")
        if not isinstance(original_id, str) or not original_id:
            raise ValueError(f"{train_path}: missing sample_id")
        annotation = annotations.get(original_id)
        if not isinstance(annotation, dict):
            raise ValueError(f"{annotation_path}: missing annotation for {original_id}")
        label = normalize_label(row.get("label"))
        image_path = Path(str(row["image_path"]))
        if not image_path.is_absolute():
            image_path = train_path.parent / image_path
        sample_id = f"docmsu:na:{original_id}"
        if sample_id in records:
            raise ValueError(f"duplicate DocMSU sample_id {sample_id}")
        records[sample_id] = {
            "sample_id": sample_id,
            "original_id": original_id,
            "image_path": str(image_path.resolve()),
            "text": str(row["text"]),
            "label": label,
            "source": "docmsu",
            "language": "en",
            "text_label": annotation.get("text_label", []),
            "img_label": annotation.get("img_label", []),
            "split": "experience_extraction",
        }
    return records


def select_records(
    records: dict[str, dict[str, Any]],
    count: int,
    *,
    seed: int,
) -> list[dict[str, Any]]:
    if count <= 0:
        raise ValueError("sample count must be positive")
    ordered_ids = sorted(records)
    if count > len(ordered_ids):
        raise ValueError(f"requested {count} samples from only {len(ordered_ids)} records")
    selected_ids = random.Random(seed).sample(ordered_ids, count)
    return [records[sample_id] for sample_id in selected_ids]


def validate_images(records: list[dict[str, Any]]) -> None:
    missing = [row["image_path"] for row in records if not Path(row["image_path"]).is_file()]
    if missing:
        preview = "\n".join(missing[:10])
        raise FileNotFoundError(
            f"missing {len(missing)} selected images; first paths:\n{preview}"
        )


def build_pairwise_groups(
    selected: dict[str, list[dict[str, Any]]],
    *,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    """Shuffle the shared union once, then project it into the three pairs."""
    all_ids = [
        row["sample_id"]
        for dataset_records in selected.values()
        for row in dataset_records
    ]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("sample IDs overlap across datasets")

    by_id = {
        row["sample_id"]: row
        for dataset_records in selected.values()
        for row in dataset_records
    }
    union_ids = list(by_id)
    random.Random(seed + 401).shuffle(union_ids)
    all_selected = [by_id[sample_id] for sample_id in union_ids]
    groups = {
        group: [row for row in all_selected if row["source"] in sources]
        for group, sources in GROUP_SOURCES.items()
    }
    return all_selected, groups


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docmsu-root", default=str(DEFAULT_DOCMSU_ROOT))
    parser.add_argument("--mmsd-root", default=str(DEFAULT_MMSD_ROOT))
    parser.add_argument("--sarcnet-root", default=str(DEFAULT_SARCNET_ROOT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--docmsu-samples", type=int, default=5000)
    parser.add_argument("--mmsd-samples", type=int, default=8000)
    parser.add_argument("--sarcnet-samples", type=int, default=1998)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    docmsu_root = Path(args.docmsu_root).expanduser().resolve()
    mmsd_root = Path(args.mmsd_root).expanduser().resolve()
    sarcnet_root = Path(args.sarcnet_root).expanduser().resolve()
    docmsu = load_docmsu_train(docmsu_root)
    mmsd2 = load_mmsd_split(mmsd_root, "v2", "train")
    sarcnet: dict[str, dict[str, Any]] = {}
    for language in ("en", "zh"):
        sarcnet.update(
            load_sarcnet_split(
                sarcnet_root,
                language,
                "train",
            )
        )
    for records in (mmsd2, sarcnet):
        for row in records.values():
            row["split"] = "experience_extraction"

    selected = {
        "docmsu": select_records(docmsu, args.docmsu_samples, seed=args.seed + 101),
        "mmsd2": select_records(mmsd2, args.mmsd_samples, seed=args.seed + 211),
        "sarcnet": select_records(sarcnet, args.sarcnet_samples, seed=args.seed + 307),
    }
    all_selected, groups = build_pairwise_groups(selected, seed=args.seed)
    validate_images(all_selected)

    for source, records in selected.items():
        write_jsonl(output_dir / f"selected_{source}.jsonl", records)
    write_jsonl(output_dir / "all_selected.jsonl", all_selected)
    for group, records in groups.items():
        write_jsonl(output_dir / f"{group}.jsonl", records)

    manifest = {
        "seed": args.seed,
        "inputs": {
            "docmsu_train": str(docmsu_root / "data" / "train.jsonl"),
            "mmsd2_v2_train": str(
                mmsd_root / "data" / "text_json_final" / "train.json"
            ),
            "sarcnet_train": [
                str(sarcnet_root / "data" / language / "train.jsonl")
                for language in ("en", "zh")
            ],
        },
        "requested": {
            "docmsu": args.docmsu_samples,
            "mmsd2": args.mmsd_samples,
            "sarcnet": args.sarcnet_samples,
        },
        "selected": {
            source: summarize(records) for source, records in selected.items()
        },
        "all_selected": summarize(all_selected),
        "groups": {group: summarize(records) for group, records in groups.items()},
        "files": {
            **{
                f"selected_{source}": f"selected_{source}.jsonl"
                for source in selected
            },
            "all_selected": "all_selected.jsonl",
            **{group: f"{group}.jsonl" for group in groups},
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
