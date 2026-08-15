from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RunConfig(StrictConfigModel):
    name: str = "pace"
    seed: int = 42
    profile: Literal["smoke", "pilot", "full", "extended"] = "extended"
    artifacts_root: Path


class PathsConfig(StrictConfigModel):
    datasets_root: Path
    mmsd2_config: Literal["mmsd-v2"] = "mmsd-v2"
    model_2b: Path
    model_4b: Path
    model_8b: Path
    retrieval_model: Path


class DataConfig(StrictConfigModel):
    extraction_ratio: float = Field(gt=0, lt=1)
    consolidation_ratio: float = Field(gt=0, lt=1)
    opd_ratio: float = Field(gt=0, lt=1)
    smoke_extraction_per_source_class: int = Field(gt=0)
    smoke_consolidation_per_source_class: int = Field(gt=0)
    smoke_opd_per_source_class: int = Field(gt=0)

    @model_validator(mode="after")
    def ratios_sum_to_one(self) -> DataConfig:
        total = self.extraction_ratio + self.consolidation_ratio + self.opd_ratio
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"secondary split ratios must sum to 1.0, got {total}")
        return self


class ApiProviderConfig(StrictConfigModel):
    base_url: str
    model: str
    key_file: Path
    supports_json_schema: bool | None = None
    supports_vision: bool | None = None


class ApiConfig(StrictConfigModel):
    selected_provider: str
    max_attempts: int = Field(default=3, gt=0)
    timeout_seconds: float = Field(default=180.0, gt=0)
    temperature: float = Field(default=0.0, ge=0)
    stream: bool = False
    per_key_concurrency: int = Field(default=1, gt=0)
    global_concurrency: int | None = Field(default=None, gt=0)
    providers: dict[str, ApiProviderConfig]

    @model_validator(mode="after")
    def selected_provider_exists(self) -> ApiConfig:
        if self.selected_provider not in self.providers:
            raise ValueError(f"unknown API provider: {self.selected_provider}")
        return self

    @property
    def selected(self) -> ApiProviderConfig:
        return self.providers[self.selected_provider]


class CurationConfig(StrictConfigModel):
    batch_size: int | None = Field(default=None, gt=0)
    global_bank_size: int = Field(default=64, gt=0)
    relevant_per_class_k: int = Field(default=10, gt=0)
    allow_bank_overlap: bool = True
    image_embedding_weight: float = Field(default=0.5, ge=0)
    text_embedding_weight: float = Field(default=0.5, ge=0)

    @model_validator(mode="after")
    def validate_balances(self) -> CurationConfig:
        if self.global_bank_size % 2:
            raise ValueError("global_bank_size must be even for class balance")
        weight = self.image_embedding_weight + self.text_embedding_weight
        if abs(weight - 1.0) > 1e-9:
            raise ValueError(f"retrieval modality weights must sum to 1.0, got {weight}")
        return self


class B0Config(StrictConfigModel):
    profile: Literal["smoke", "pilot", "full", "extended"] = "extended"
    targets_per_source_class: dict[str, int]
    balance_sources: bool = True
    balance_classes: bool = True
    balance_sarcnet_languages: bool = True

    @model_validator(mode="after")
    def selected_profile_exists(self) -> B0Config:
        if self.profile not in self.targets_per_source_class:
            raise ValueError(f"missing B0 target count for profile {self.profile}")
        if any(value <= 0 for value in self.targets_per_source_class.values()):
            raise ValueError("all B0 target counts must be positive")
        return self

    @property
    def target_count(self) -> int:
        return self.targets_per_source_class[self.profile]


class ContextConfig(StrictConfigModel):
    include_all_experiences: Literal[True] = True
    allow_truncation: Literal[False] = False
    overflow_strategy: Literal["error"] = "error"
    warning_ratios: tuple[float, ...] = (0.80, 0.90, 0.95)
    response_reserve_tokens: int = Field(default=256, ge=0)
    safety_margin_tokens: int = Field(default=512, ge=0)

    @model_validator(mode="after")
    def validate_warning_ratios(self) -> ContextConfig:
        if tuple(sorted(self.warning_ratios)) != self.warning_ratios:
            raise ValueError("context warning ratios must be sorted")
        if not all(0 < ratio < 1 for ratio in self.warning_ratios):
            raise ValueError("context warning ratios must be between 0 and 1")
        return self


class TrainingConfig(StrictConfigModel):
    dtype: Literal["bfloat16"] = "bfloat16"
    epochs: int = Field(default=3, gt=0)
    micro_batch_size: int = Field(default=1, gt=0)
    sft_gradient_accumulation: int = Field(default=8, gt=0)
    opd_gradient_accumulation: int = Field(default=4, gt=0)
    full_sft_learning_rate: float = Field(gt=0)
    lora_sft_learning_rate: float = Field(gt=0)
    full_opd_learning_rate: float = Field(gt=0)
    lora_opd_learning_rate: float = Field(gt=0)
    scheduler: Literal["cosine"] = "cosine"
    warmup_ratio: float = Field(ge=0, lt=1)
    weight_decay: float = Field(ge=0)
    max_sequence_length: int = Field(gt=0)
    max_new_tokens: int = Field(gt=0)
    gradient_clip_norm: float = Field(gt=0)
    rollout_temperature: float = Field(gt=0)
    rollout_top_p: float = Field(gt=0, le=1)
    keep_invalid_rollouts: Literal[True] = True
    gold_ce_weight: float = Field(default=1.0, ge=0)
    lora_rank: int = Field(default=32, gt=0)
    lora_alpha: int = Field(default=64, gt=0)
    lora_dropout: float = Field(default=0.05, ge=0, lt=1)


class EvaluationConfig(StrictConfigModel):
    generation_temperature: Literal[0.0] = 0.0
    invalid_json_is_incorrect: Literal[True] = True
    checkpoint_metric: Literal["mean_source_macro_f1"] = "mean_source_macro_f1"


class AppConfig(StrictConfigModel):
    run: RunConfig
    paths: PathsConfig
    data: DataConfig
    api: ApiConfig
    curation: CurationConfig
    b0: B0Config
    context: ContextConfig
    training: TrainingConfig
    evaluation: EvaluationConfig


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"configuration root must be a mapping: {path}")
    return value


def deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def load_config(path: str | Path, overlays: list[str | Path] | None = None) -> AppConfig:
    config_path = Path(path)
    raw = _read_yaml(config_path)
    for overlay in overlays or []:
        raw = deep_merge(raw, _read_yaml(Path(overlay)))
    return AppConfig.model_validate(raw)
