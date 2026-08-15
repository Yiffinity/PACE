import json

import pytest

from pace.context import ContextBudget, ExperienceContextOverflowError
from pace.utils.atomic import atomic_write_json
from pace.utils.hashing import stable_hash


def test_hash_is_independent_of_mapping_order() -> None:
    assert stable_hash({"a": 1, "b": 2}) == stable_hash({"b": 2, "a": 1})


def test_atomic_json_write(tmp_path) -> None:
    target = tmp_path / "nested" / "value.json"
    atomic_write_json(target, {"value": "ok"})
    assert json.loads(target.read_text(encoding="utf-8")) == {"value": "ok"}
    assert not list(target.parent.glob(f".{target.name}.*"))


def test_context_budget_warns_and_overflows() -> None:
    warning = ContextBudget(
        sample_id="sample-1",
        context_limit=100,
        system_tokens=10,
        image_tokens=20,
        text_tokens=20,
        experience_tokens=25,
        prefix_tokens=5,
        response_reserve_tokens=5,
        safety_margin_tokens=5,
    )
    assert warning.warning_thresholds_crossed((0.8, 0.9, 0.95)) == (0.8, 0.9)
    warning.ensure_fits()

    overflow = ContextBudget(
        sample_id="sample-2",
        context_limit=100,
        system_tokens=10,
        image_tokens=20,
        text_tokens=20,
        experience_tokens=50,
        prefix_tokens=5,
        response_reserve_tokens=5,
        safety_margin_tokens=5,
    )
    with pytest.raises(ExperienceContextOverflowError, match="sample-2"):
        overflow.ensure_fits()
