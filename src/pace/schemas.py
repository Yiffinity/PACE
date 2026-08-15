from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class StrictSchema(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class Label(StrEnum):
    SARCASTIC = "sarcastic"
    NON_SARCASTIC = "non-sarcastic"

    @property
    def binary(self) -> int:
        return 1 if self is Label.SARCASTIC else 0


class SarcasmAnalysis(StrictSchema):
    visual_evidence: str = Field(min_length=1)
    textual_evidence: str = Field(min_length=1)
    explanation: str = Field(min_length=1)
    label: Label


class ExperienceReflection(StrictSchema):
    corrected_analysis: SarcasmAnalysis
    has_new_rule: bool
    experience: str | None = None

    @field_validator("experience")
    @classmethod
    def normalize_experience(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None


class MultimodalSample(StrictSchema):
    sample_id: str = Field(min_length=1)
    source: Literal["mmsd2", "docmsu", "sarcnet", "redeval", "mmsd3", "cfms"]
    official_split: str = Field(min_length=1)
    text: str
    label: Label
    image_paths: tuple[str, ...] = Field(min_length=1)
    language: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class RecordLocator(StrictSchema):
    """Pointer to a source record without duplicating image/text payloads."""

    format: Literal["parquet", "json"]
    relative_file: str = Field(min_length=1)
    row_index: int = Field(ge=0)


class ManifestEntry(StrictSchema):
    """One immutable sample assignment in a secondary training split."""

    sample_id: str = Field(min_length=1)
    source: str = Field(min_length=1)
    official_split: Literal["train"] = "train"
    secondary_split: Literal["extraction", "consolidation", "opd"]
    label: Label
    language: str | None = None
    locator: RecordLocator
    stream_index: int = Field(ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ApiAttempt(StrictSchema):
    request_hash: str
    provider: str
    model: str
    sample_id: str
    stage: str
    attempt: int = Field(gt=0)
    success: bool
    status_code: int | None = None
    error_type: str | None = None
    error_message: str | None = None


class ExperienceUtility(StrictSchema):
    relevant_score: float
    global_score: float
    relevant_deltas: tuple[float, ...]
    global_deltas: tuple[float, ...]

    @property
    def accepted(self) -> bool:
        return self.relevant_score > 0 and self.global_score >= 0


class ExperienceRecord(StrictSchema):
    experience_id: str
    text: str = Field(min_length=1)
    source_sample_id: str
    source: str
    teacher_id: str
    pool_version_before: int = Field(ge=0)
    status: Literal["emerging", "accepted", "rejected"]
    utility: ExperienceUtility | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


def parse_sarcasm_analysis(raw: str) -> SarcasmAnalysis:
    return SarcasmAnalysis.model_validate_json(raw)


def parse_experience_reflection(raw: str) -> ExperienceReflection:
    return ExperienceReflection.model_validate_json(raw)
