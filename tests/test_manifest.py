from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from pace.data.manifest import (
    audit_manifest_bundle,
    split_training_records,
    write_manifest_bundle,
)
from pace.data.parquet import (
    ParquetSourceSpec,
    SourceRecord,
    scan_training_source,
)
from pace.schemas import Label, ManifestEntry, RecordLocator


def _record(
    source: str,
    label: Label,
    index: int,
    *,
    language: str | None = None,
) -> SourceRecord:
    return SourceRecord(
        sample_id=f"{source}:{language or 'na'}:{label.value}:{index}",
        source=source,
        label=label,
        language=language,
        locator=RecordLocator(
            format="parquet",
            relative_file=f"{source}/train.parquet",
            row_index=index,
        ),
        metadata={},
    )


def _balanced_records() -> tuple[SourceRecord, ...]:
    records: list[SourceRecord] = []
    for source, language in (
        ("mmsd2", None),
        ("docmsu", None),
        ("sarcnet", "en"),
        ("sarcnet", "zh"),
    ):
        for label in Label:
            records.extend(_record(source, label, index, language=language) for index in range(10))
    return tuple(records)


def test_secondary_split_is_deterministic_stratified_and_disjoint() -> None:
    records = _balanced_records()
    first = split_training_records(
        records,
        seed=42,
        ratios=(0.6, 0.2, 0.2),
    )
    second = split_training_records(
        tuple(reversed(records)),
        seed=42,
        ratios=(0.6, 0.2, 0.2),
    )
    assert first == second
    assert audit_manifest_bundle(first, expected_total=80)["overlap_counts"] == {
        "extraction:consolidation": 0,
        "extraction:opd": 0,
        "consolidation:opd": 0,
    }

    for entries, expected_per_stratum in (
        (first.extraction, 6),
        (first.consolidation, 2),
        (first.opd, 2),
    ):
        counts = Counter((entry.source, entry.language, entry.label) for entry in entries)
        assert set(counts.values()) == {expected_per_stratum}


def test_different_seed_changes_assignments() -> None:
    records = _balanced_records()
    first = split_training_records(records, seed=42, ratios=(0.6, 0.2, 0.2))
    second = split_training_records(records, seed=43, ratios=(0.6, 0.2, 0.2))
    assert {entry.sample_id for entry in first.extraction} != {
        entry.sample_id for entry in second.extraction
    }


def test_manifest_writer_emits_valid_jsonl(tmp_path: Path) -> None:
    bundle = split_training_records(
        _balanced_records(),
        seed=42,
        ratios=(0.6, 0.2, 0.2),
    )
    stats = write_manifest_bundle(
        tmp_path,
        bundle,
        provenance={"seed": 42},
    )
    assert stats["audit"]["total"] == 80
    assert len(stats["manifest_hash"]) == 64

    lines = (tmp_path / "extraction.jsonl").read_text(encoding="utf-8").splitlines()
    parsed = [ManifestEntry.model_validate_json(line) for line in lines]
    assert parsed == list(bundle.extraction)
    assert json.loads((tmp_path / "stats.json").read_text())["schema_version"] == 1


def test_parquet_source_scan_uses_stable_row_locators(tmp_path: Path) -> None:
    pyarrow = pytest.importorskip("pyarrow")
    parquet = pytest.importorskip("pyarrow.parquet")
    directory = tmp_path / "example"
    directory.mkdir()
    table = pyarrow.table(
        {
            "id": ["a", "b", "c"],
            "label": [0, 1, 0],
            "unused": ["x", "y", "z"],
        }
    )
    parquet.write_table(table, directory / "train-00000-of-00001.parquet")

    records = scan_training_source(
        tmp_path,
        ParquetSourceSpec(
            source="mmsd2",
            relative_directory=Path("example"),
            label_column="label",
            id_column="id",
        ),
    )
    assert [record.sample_id for record in records] == [
        "mmsd2:na:a",
        "mmsd2:na:b",
        "mmsd2:na:c",
    ]
    assert [record.locator.row_index for record in records] == [0, 1, 2]
    assert records[1].label is Label.SARCASTIC
