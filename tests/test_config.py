from pathlib import Path

import pytest
from pydantic import ValidationError

from pace.config import DataConfig, deep_merge, load_config

ROOT = Path(__file__).resolve().parents[1]


def test_base_config_is_valid() -> None:
    config = load_config(ROOT / "configs/base.yaml")
    assert config.paths.mmsd2_config == "mmsd-v2"
    assert config.b0.profile == "extended"
    assert config.b0.target_count == 256
    assert config.curation.relevant_per_class_k == 10
    assert config.context.allow_truncation is False


def test_overlay_changes_only_requested_values() -> None:
    config = load_config(
        ROOT / "configs/base.yaml",
        [ROOT / "configs/experiments/smoke.yaml"],
    )
    assert config.run.profile == "smoke"
    assert config.b0.target_count == 8
    assert config.api.selected_provider == "micu"


def test_invalid_split_ratios_are_rejected() -> None:
    with pytest.raises(ValidationError, match="must sum to 1.0"):
        DataConfig(
            extraction_ratio=0.5,
            consolidation_ratio=0.2,
            opd_ratio=0.2,
            smoke_extraction_per_source_class=1,
            smoke_consolidation_per_source_class=1,
            smoke_opd_per_source_class=1,
        )


def test_deep_merge_preserves_nested_base_values() -> None:
    assert deep_merge({"a": {"b": 1, "c": 2}}, {"a": {"b": 3}}) == {"a": {"b": 3, "c": 2}}
