from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pace.config import AppConfig
from pace.data.parquet import SourceRecord, scan_all_training_records
from pace.schemas import ManifestEntry
from pace.utils.atomic import atomic_write_bytes, atomic_write_json
from pace.utils.hashing import stable_hash

SplitName = Literal["extraction", "consolidation", "opd"]
_SPLIT_NAMES: tuple[SplitName, ...] = ("extraction", "consolidation", "opd")


@dataclass(frozen=True, slots=True)
class ManifestBundle:
    extraction: tuple[ManifestEntry, ...]
    consolidation: tuple[ManifestEntry, ...]
    opd: tuple[ManifestEntry, ...]

    def for_split(self, split: SplitName) -> tuple[ManifestEntry, ...]:
        return getattr(self, split)


def _stable_rank(seed: int, namespace: str, sample_id: str) -> str:
    return stable_hash(
        {
            "seed": seed,
            "namespace": namespace,
            "sample_id": sample_id,
        }
    )


def _largest_remainder_counts(
    size: int,
    ratios: tuple[float, float, float],
) -> tuple[int, int, int]:
    raw = tuple(size * ratio for ratio in ratios)
    counts = [math.floor(value) for value in raw]
    remaining = size - sum(counts)
    priority = sorted(
        range(len(ratios)),
        key=lambda index: (-(raw[index] - counts[index]), index),
    )
    for index in priority[:remaining]:
        counts[index] += 1
    return counts[0], counts[1], counts[2]


def _stratum(record: SourceRecord) -> tuple[str, str, str]:
    language = record.language if record.source == "sarcnet" else "all"
    return record.source, language or "unknown", record.label.value


def split_training_records(
    records: tuple[SourceRecord, ...],
    *,
    seed: int,
    ratios: tuple[float, float, float],
) -> ManifestBundle:
    if abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError(f"secondary split ratios must sum to 1.0, got {sum(ratios)}")

    sample_ids = [record.sample_id for record in records]
    duplicates = sorted(
        sample_id
        for sample_id, count in Counter(sample_ids).items()
        if count > 1
    )
    if duplicates:
        preview = duplicates[:5]
        raise ValueError(f"duplicate source sample ids: {preview}")

    strata: dict[tuple[str, str, str], list[SourceRecord]] = defaultdict(list)
    for record in records:
        strata[_stratum(record)].append(record)

    assigned: dict[SplitName, list[ManifestEntry]] = {
        split: [] for split in _SPLIT_NAMES
    }
    for stratum_key in sorted(strata):
        ordered = sorted(
            strata[stratum_key],
            key=lambda record: (
                _stable_rank(seed, "secondary-split", record.sample_id),
                record.sample_id,
            ),
        )
        counts = _largest_remainder_counts(len(ordered), ratios)
        offset = 0
        for split, count in zip(_SPLIT_NAMES, counts, strict=True):
            for record in ordered[offset : offset + count]:
                assigned[split].append(
                    ManifestEntry(
                        sample_id=record.sample_id,
                        source=record.source,
                        secondary_split=split,
                        label=record.label,
                        language=record.language,
                        locator=record.locator,
                        stream_index=0,
                        metadata=dict(record.metadata),
                    )
                )
            offset += count
        if offset != len(ordered):
            raise AssertionError("secondary split allocation lost records")

    finalized: dict[SplitName, tuple[ManifestEntry, ...]] = {}
    for split in _SPLIT_NAMES:
        ordered_entries = sorted(
            assigned[split],
            key=lambda entry: (
                _stable_rank(seed, f"{split}-stream", entry.sample_id),
                entry.sample_id,
            ),
        )
        finalized[split] = tuple(
            entry.model_copy(update={"stream_index": index})
            for index, entry in enumerate(ordered_entries)
        )

    bundle = ManifestBundle(
        extraction=finalized["extraction"],
        consolidation=finalized["consolidation"],
        opd=finalized["opd"],
    )
    audit_manifest_bundle(bundle, expected_total=len(records))
    return bundle


def audit_manifest_bundle(
    bundle: ManifestBundle,
    *,
    expected_total: int | None = None,
) -> dict[str, object]:
    split_ids: dict[str, set[str]] = {}
    for split in _SPLIT_NAMES:
        entries = bundle.for_split(split)
        ids = [entry.sample_id for entry in entries]
        if len(ids) != len(set(ids)):
            raise ValueError(f"duplicate sample ids inside {split} manifest")
        if any(entry.secondary_split != split for entry in entries):
            raise ValueError(f"secondary_split mismatch inside {split} manifest")
        if [entry.stream_index for entry in entries] != list(range(len(entries))):
            raise ValueError(f"non-contiguous stream indices inside {split} manifest")
        split_ids[split] = set(ids)

    overlap_pairs: dict[str, int] = {}
    for left_index, left in enumerate(_SPLIT_NAMES):
        for right in _SPLIT_NAMES[left_index + 1 :]:
            overlap = split_ids[left] & split_ids[right]
            overlap_pairs[f"{left}:{right}"] = len(overlap)
            if overlap:
                preview = sorted(overlap)[:5]
                raise ValueError(
                    f"secondary manifests overlap for {left} and {right}: {preview}"
                )

    total = sum(len(ids) for ids in split_ids.values())
    if expected_total is not None and total != expected_total:
        raise ValueError(
            f"manifest total mismatch: expected {expected_total}, observed {total}"
        )
    return {
        "total": total,
        "split_counts": {
            split: len(split_ids[split]) for split in _SPLIT_NAMES
        },
        "overlap_counts": overlap_pairs,
    }


def _entry_statistics(entries: tuple[ManifestEntry, ...]) -> dict[str, object]:
    by_source = Counter(entry.source for entry in entries)
    by_source_label = Counter(
        f"{entry.source}/{entry.label.value}" for entry in entries
    )
    by_stratum = Counter(
        f"{entry.source}/{entry.language or 'all'}/{entry.label.value}"
        for entry in entries
    )
    return {
        "total": len(entries),
        "by_source": dict(sorted(by_source.items())),
        "by_source_label": dict(sorted(by_source_label.items())),
        "by_stratum": dict(sorted(by_stratum.items())),
    }


def manifest_statistics(bundle: ManifestBundle) -> dict[str, object]:
    return {
        "total": sum(len(bundle.for_split(split)) for split in _SPLIT_NAMES),
        "by_split": {
            split: _entry_statistics(bundle.for_split(split))
            for split in _SPLIT_NAMES
        },
    }


def _jsonl_bytes(entries: tuple[ManifestEntry, ...]) -> bytes:
    return b"".join(
        entry.model_dump_json().encode("utf-8") + b"\n" for entry in entries
    )


def write_manifest_bundle(
    output_directory: str | Path,
    bundle: ManifestBundle,
    *,
    provenance: dict[str, object],
) -> dict[str, object]:
    output = Path(output_directory)
    audit = audit_manifest_bundle(bundle)
    for split in _SPLIT_NAMES:
        atomic_write_bytes(
            output / f"{split}.jsonl",
            _jsonl_bytes(bundle.for_split(split)),
        )

    payloads = {
        split: [
            entry.model_dump(mode="json") for entry in bundle.for_split(split)
        ]
        for split in _SPLIT_NAMES
    }
    stats: dict[str, object] = {
        "schema_version": 1,
        "provenance": provenance,
        "audit": audit,
        "statistics": manifest_statistics(bundle),
        "manifest_hash": stable_hash(payloads),
    }
    atomic_write_json(output / "stats.json", stats)
    return stats


def build_training_manifests(
    config: AppConfig,
    *,
    output_directory: str | Path | None = None,
) -> tuple[ManifestBundle, Path, dict[str, object]]:
    records = scan_all_training_records(
        config.paths.datasets_root,
        mmsd2_config=config.paths.mmsd2_config,
    )
    ratios = (
        config.data.extraction_ratio,
        config.data.consolidation_ratio,
        config.data.opd_ratio,
    )
    bundle = split_training_records(
        records,
        seed=config.run.seed,
        ratios=ratios,
    )
    output = (
        Path(output_directory)
        if output_directory is not None
        else config.run.artifacts_root / "manifests" / config.run.name
    )
    stats = write_manifest_bundle(
        output,
        bundle,
        provenance={
            "seed": config.run.seed,
            "ratios": {
                split: ratio
                for split, ratio in zip(_SPLIT_NAMES, ratios, strict=True)
            },
            "datasets_root": str(config.paths.datasets_root),
            "mmsd2_config": config.paths.mmsd2_config,
        },
    )
    return bundle, output, stats
