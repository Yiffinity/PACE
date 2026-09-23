"""Runtime acceptance checks for the one-step PACE training pilot."""

from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

import torch

from verl.trainer.pace_schema import SchemaError, parse_reasoning


def update_pilot_summary(path: str | Path, updates: Mapping[str, Any]) -> None:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file():
        value = json.loads(target.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise RuntimeError(f"pilot summary must be a JSON object: {target}")
    else:
        value = {}
    value.update(updates)
    fd, temp_name = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=True, sort_keys=True, indent=2)
            handle.write("\n")
        os.replace(temp_name, target)
    except BaseException:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def training_pilot_updates(batch: Any, tokenizer: Any, metrics: Mapping[str, Any]) -> dict[str, Any]:
    outputs = tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
    valid = 0
    for output in outputs:
        try:
            parse_reasoning(output)
        except SchemaError:
            continue
        valid += 1

    loss_value = metrics.get("actor/distillation/loss", metrics.get("distillation/loss"))
    try:
        loss = float(loss_value)
    except (TypeError, ValueError):
        loss = math.nan
    teacher_logprobs = batch.batch.get("teacher_logprobs")
    teacher_scores_finite = bool(
        isinstance(teacher_logprobs, torch.Tensor)
        and teacher_logprobs.numel() > 0
        and torch.isfinite(teacher_logprobs).all().item()
    )
    return {
        "rollout_samples": len(outputs),
        "rollout_json_valid": valid,
        "rollout_valid": bool(outputs) and valid == len(outputs),
        "distillation_loss": loss if math.isfinite(loss) else None,
        "teacher_logprobs_finite": teacher_scores_finite,
        "kl_finite": math.isfinite(loss) and teacher_scores_finite,
        "training_step_success": True,
    }
