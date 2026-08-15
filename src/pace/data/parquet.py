from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pace.schemas import Label, RecordLocator


@dataclass(frozen=True, slots=True)
class ParquetSourceSpec:
    source: str
    relative_directory: Path
    label_column: str
    id_column: str | None
    language: str | None = None
    metadata_columns: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class SourceRecord:
    sample_id: str
    source: str
    label: Label
    language: str | None
    locator: RecordLocator
    metadata: dict[str, Any]


def training_source_specs(mmsd2_config: str = "mmsd-v2") -> tuple[ParquetSourceSpec, ...]:
    return (
        ParquetSourceSpec(
            source="mmsd2",
            relative_directory=Path("MMSD2.0") / mmsd2_config,
            label_column="label",
            id_column="id",
        ),
        ParquetSourceSpec(
            source="docmsu",
            relative_directory=Path("DocMSU/sarcasm-detection"),
            label_column="label",
            id_column=None,
        ),
        ParquetSourceSpec(
            source="sarcnet",
            relative_directory=Path("sarcnet/en"),
            label_column="multi_label",
            id_column="id",
            language="en",
            metadata_columns=("text_label", "image_label"),
        ),
        ParquetSourceSpec(
            source="sarcnet",
            relative_directory=Path("sarcnet/zh"),
            label_column="multi_label",
            id_column="id",
            language="zh",
            metadata_columns=("text_label", "image_label"),
        ),
    )


def _label_from_raw(value: object) -> Label:
    if isinstance(value, Label):
        return value
    if isinstance(value, bool):
        return Label.SARCASTIC if value else Label.NON_SARCASTIC
    if isinstance(value, int) and value in (0, 1):
        return Label.SARCASTIC if value == 1 else Label.NON_SARCASTIC
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "sarcastic"}:
        return Label.SARCASTIC
    if normalized in {"0", "false", "non-sarcastic", "nonsarcastic"}:
        return Label.NON_SARCASTIC
    raise ValueError(f"unsupported sarcasm label: {value!r}")


def _pyarrow_parquet():
    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:
        raise RuntimeError(
            "PyArrow is required for dataset manifests; install PACE with the data extra"
        ) from exc
    return parquet


def scan_training_source(
    datasets_root: str | Path,
    spec: ParquetSourceSpec,
) -> tuple[SourceRecord, ...]:
    parquet = _pyarrow_parquet()
    root = Path(datasets_root)
    source_directory = root / spec.relative_directory
    files = sorted(source_directory.glob("train-*.parquet"))
    if not files:
        raise FileNotFoundError(f"no training Parquet files found in {source_directory}")

    requested_columns = [spec.label_column, *spec.metadata_columns]
    if spec.id_column is not None:
        requested_columns.append(spec.id_column)

    records: list[SourceRecord] = []
    for file_path in files:
        parquet_file = parquet.ParquetFile(file_path)
        available = set(parquet_file.schema_arrow.names)
        missing = sorted(set(requested_columns) - available)
        if missing:
            raise ValueError(f"{file_path} is missing required columns: {missing}")

        row_index = 0
        for batch in parquet_file.iter_batches(
            batch_size=8192,
            columns=requested_columns,
        ):
            columns = batch.to_pydict()
            for batch_index in range(batch.num_rows):
                if spec.id_column is None:
                    original_id = f"{file_path.stem}:{row_index}"
                else:
                    raw_id = columns[spec.id_column][batch_index]
                    if raw_id is None or not str(raw_id).strip():
                        raise ValueError(
                            f"empty id at {file_path.relative_to(root)} row {row_index}"
                        )
                    original_id = str(raw_id).strip()

                language_component = spec.language or "na"
                sample_id = f"{spec.source}:{language_component}:{original_id}"
                metadata = {
                    column: columns[column][batch_index]
                    for column in spec.metadata_columns
                }
                records.append(
                    SourceRecord(
                        sample_id=sample_id,
                        source=spec.source,
                        label=_label_from_raw(
                            columns[spec.label_column][batch_index]
                        ),
                        language=spec.language,
                        locator=RecordLocator(
                            format="parquet",
                            relative_file=file_path.relative_to(root).as_posix(),
                            row_index=row_index,
                        ),
                        metadata=metadata,
                    )
                )
                row_index += 1

    return tuple(records)


def scan_all_training_records(
    datasets_root: str | Path,
    *,
    mmsd2_config: str = "mmsd-v2",
) -> tuple[SourceRecord, ...]:
    records: list[SourceRecord] = []
    for spec in training_source_specs(mmsd2_config):
        records.extend(scan_training_source(datasets_root, spec))
    return tuple(records)
