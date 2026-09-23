#!/usr/bin/env python3
"""Command-line entry points for the PACE_PLUS workflow."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "verl"))

# Stage A and static preflight do not require Ray. Avoid executing VeRL's
# heavyweight package initializer; the training subprocess imports full VeRL.
if "verl" not in sys.modules:
    verl_package = ModuleType("verl")
    verl_package.__path__ = [str(PROJECT_ROOT / "verl" / "verl")]
    verl_package.__package__ = "verl"
    sys.modules["verl"] = verl_package

from verl.trainer.pace_plus_benchmark import run_benchmark
from verl.trainer.pace_plus_config import get_required, load_config, project_path
from verl.trainer.pace_plus_data import normalize_record, to_verl_record
from verl.trainer.pace_plus_evaluation import run_evaluation
from verl.trainer.pace_plus_extraction import run_extraction, run_reasoning_generation
from verl.trainer.pace_plus_pool import load_active_experience_text
from verl.trainer.pace_plus_preflight import validate_model_pair
from verl.trainer.pace_plus_reflection import embedded_reflection_prompt_document
from verl.trainer.pace_plus_schema import REASONING_JSON_SCHEMA, SchemaError, read_jsonl
from verl.trainer.pace_plus_weighted_pool import run_weighted_reference_pipeline


def _json(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=True, sort_keys=True, indent=2))


def _write_summary(config: Mapping[str, Any], updates: Mapping[str, Any]) -> None:
    path = project_path(config, config.get("pilot", {}).get("summary_path", "pilot_summary.json"))
    if path.exists():
        value = json.loads(path.read_text(encoding="utf-8"))
    else:
        value = {
            "samples": 0,
            "teacher_valid": 0,
            "student_valid": 0,
            "teacher_verified": 0,
            "experience_non_null": 0,
            "experience_null": 0,
            "rollout_valid": False,
            "kl_finite": False,
            "training_step_success": False,
            "teacher_unchanged": False,
        }
    value.update(updates)
    path.write_text(json.dumps(value, ensure_ascii=True, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _model_context_limit(model_path: str | Path) -> int:
    config_path = Path(model_path).expanduser().resolve() / "config.json"
    try:
        model_config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read model context limit from {config_path}: {exc}") from exc
    candidates = (
        model_config.get("max_position_embeddings"),
        (model_config.get("text_config") or {}).get("max_position_embeddings"),
    )
    limits = [value for value in candidates if isinstance(value, int) and value > 0]
    if not limits:
        raise RuntimeError(f"model config does not declare max_position_embeddings: {config_path}")
    return min(limits)


def prepare_data(
    config: Mapping[str, Any],
    source_path: str | Path,
    output_path: str | Path,
    *,
    split: str,
    limit: int | None,
) -> dict[str, int | str]:
    try:
        import datasets
    except ImportError as exc:
        raise RuntimeError("datasets is required to create VeRL parquet data") from exc
    source = Path(source_path).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    records = []
    seen: set[str] = set()
    for line_number, raw in read_jsonl(source):
        try:
            sample = normalize_record(raw, base_dir=source.parent, require_image=True)
        except SchemaError as exc:
            raise RuntimeError(f"{source}:{line_number}: {exc}") from exc
        if sample.sample_id in seen:
            raise RuntimeError(f"{source}:{line_number}: duplicate sample_id {sample.sample_id!r}")
        seen.add(sample.sample_id)
        records.append(to_verl_record(sample, split=split, index=len(records)))
        if limit is not None and len(records) >= limit:
            break
    if not records:
        raise RuntimeError(f"no valid records in {source}")
    output.parent.mkdir(parents=True, exist_ok=True)
    datasets.Dataset.from_list(records).to_parquet(str(output))
    return {"samples": len(records), "output": str(output)}


def validate_data(source_path: str | Path, *, require_reference: bool) -> dict[str, int]:
    source = Path(source_path).expanduser().resolve()
    samples = 0
    references = 0
    seen: set[str] = set()
    for line_number, raw in read_jsonl(source):
        sample = normalize_record(raw, base_dir=source.parent, require_image=True)
        if sample.sample_id in seen:
            raise RuntimeError(f"{source}:{line_number}: duplicate sample_id {sample.sample_id!r}")
        seen.add(sample.sample_id)
        samples += 1
        references += sample.reference_reasoning is not None
    if require_reference and references != samples:
        raise RuntimeError(f"{samples - references} samples do not include inline reference reasoning")
    return {"samples": samples, "inline_references": references}


def _preflight(config: Mapping[str, Any]) -> dict[str, Any]:
    if get_required(config, "models.teacher.trainable") is not False:
        raise RuntimeError("models.teacher.trainable must be false")
    if get_required(config, "models.student.trainable") is not True:
        raise RuntimeError("models.student.trainable must be true")
    opcd = config.get("opcd", {})
    extraction_generation = config.get("experience_extraction", {}).get("generation", {})
    pool_generation = config.get("experience_pool", {}).get("generation", {})
    evaluation_generation = config.get("evaluation", {}).get("generation", {})
    if extraction_generation.get("chat_template_kwargs", {}).get("enable_thinking") is not True:
        raise RuntimeError("experience_extraction.generation.chat_template_kwargs.enable_thinking must be true")
    if pool_generation.get("chat_template_kwargs", {}).get("enable_thinking") is not False:
        raise RuntimeError("experience_pool.generation.chat_template_kwargs.enable_thinking must be false")
    for stage_name, expected_thinking in (
        ("normalization", False),
        ("comparison", False),
        ("merge", False),
    ):
        stage_generation = pool_generation.get(stage_name, {})
        if stage_generation.get("chat_template_kwargs", {}).get("enable_thinking") is not expected_thinking:
            raise RuntimeError(
                "experience_pool.generation."
                f"{stage_name}.chat_template_kwargs.enable_thinking must be "
                f"{str(expected_thinking).lower()}"
            )
    if evaluation_generation.get("chat_template_kwargs", {}).get("enable_thinking") is not False:
        raise RuntimeError("evaluation.generation.chat_template_kwargs.enable_thinking must be false")
    for stage, generation in (
        ("experience_extraction", extraction_generation),
        ("experience_pool", pool_generation),
        ("evaluation", evaluation_generation),
    ):
        attempts = int(generation.get("json", {}).get("max_attempts", 3))
        if attempts != 3:
            raise RuntimeError(f"{stage}.generation.json.max_attempts must be exactly 3")

    required = {
        "enabled": True,
        "on_policy": True,
        "teacher_uses_experience": True,
        "student_uses_experience": False,
        "same_student_prefix": True,
        "kl_type": "reverse",
    }
    for key, expected in required.items():
        if opcd.get(key) != expected:
            raise RuntimeError(f"opcd.{key} must be {expected!r}")

    total_gpus = int(opcd.get("n_gpus_per_node", 0))
    actor_gpus = int(opcd.get("actor_gpus_per_node", 0))
    teacher_gpus = int(opcd.get("teacher_gpus_per_node", 0))
    if total_gpus != 2 or actor_gpus != 1 or teacher_gpus != 1:
        raise RuntimeError(
            "PACE_PLUS is configured for exactly two GPUs: "
            "opcd.n_gpus_per_node=2, actor_gpus_per_node=1, teacher_gpus_per_node=1"
        )
    if actor_gpus + teacher_gpus != total_gpus:
        raise RuntimeError("actor and teacher GPU pools must sum to opcd.n_gpus_per_node")

    artifacts = {}
    for key in ("reference_reasoning_source_path", "precomputed_teacher_reasoning_path"):
        value = config.get("experience_extraction", {}).get(key)
        if value is None:
            continue
        path = project_path(config, value)
        if not path.is_file():
            raise RuntimeError(f"experience_extraction.{key} does not exist: {path}")
        artifacts[key] = str(path)

    prompt_document = embedded_reflection_prompt_document()
    artifacts["comparative_prompt"] = prompt_document.path

    schema_path = project_path(
        config, config.get("opcd", {}).get("json_schema_path", "configs/pace_plus_reasoning.schema.json")
    )
    try:
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"opcd.json_schema_path is not valid JSON: {schema_path}: {exc}") from exc
    if not isinstance(schema, dict) or schema.get("type") != "object" or schema.get("additionalProperties") is not False:
        raise RuntimeError(f"opcd.json_schema_path must be an object schema with additionalProperties=false: {schema_path}")
    if schema != REASONING_JSON_SCHEMA:
        raise RuntimeError(f"opcd.json_schema_path must match the PACE_PLUS reasoning schema: {schema_path}")

    result = validate_model_pair(
        get_required(config, "models.teacher.path"),
        get_required(config, "models.student.path"),
    )
    result["resources"] = {
        "total_gpus": total_gpus,
        "actor_gpus": actor_gpus,
        "teacher_gpus": teacher_gpus,
    }
    result["artifacts"] = artifacts
    return result


def _training_command(
    config: Mapping[str, Any],
    parquet: Path,
    experience_path: Path,
    *,
    pilot: bool,
) -> list[str]:
    opcd = config["opcd"]
    training_root = project_path(
        config, config.get("output", {}).get("training_dir", "training")
    )
    output_dir = training_root / ("pilot" if pilot else "opcd")
    output_dir.mkdir(parents=True, exist_ok=True)
    steps = 1 if pilot else int(opcd["total_training_steps"])
    summary = project_path(config, config.get("pilot", {}).get("summary_path", "pilot_summary.json"))
    schema_path = project_path(
        config, opcd.get("json_schema_path", "configs/pace_plus_reasoning.schema.json")
    )
    student_thinking = bool(opcd.get("chat_template_kwargs", {}).get("enable_thinking", False))
    response_length = int(opcd["max_response_length"])
    student_context = _model_context_limit(get_required(config, "models.student.path"))
    teacher_context = _model_context_limit(get_required(config, "models.teacher.path"))
    max_prompt_length = student_context - response_length
    if max_prompt_length <= 0:
        raise RuntimeError("student context length must exceed opcd.max_response_length")
    reward_path = PROJECT_ROOT / "verl" / "verl" / "trainer" / "pace_plus_reward.py"
    command = [
        sys.executable,
        "-m",
        "verl.trainer.main_ppo",
        "algorithm.adv_estimator=grpo",
        f"data.train_files={parquet}",
        f"data.val_files={parquet}",
        "data.prompt_key=prompt",
        "data.image_key=images",
        f"+data.apply_chat_template_kwargs.enable_thinking={str(student_thinking).lower()}",
        f"data.train_batch_size={opcd['train_batch_size']}",
        f"data.max_prompt_length={max_prompt_length}",
        f"data.max_response_length={opcd['max_response_length']}",
        "data.filter_overlong_prompts=False",
        "data.truncation=error",
        "data.dataloader_num_workers=0",
        f"actor_rollout_ref.model.path={get_required(config, 'models.student.path')}",
        f"actor_rollout_ref.actor.optim.lr={opcd['learning_rate']}",
        "actor_rollout_ref.actor.optim.override_optimizer_config={foreach:false}",
        "actor_rollout_ref.model.use_remove_padding=False",
        "actor_rollout_ref.model.enable_gradient_checkpointing=True",
        "actor_rollout_ref.actor.strategy=fsdp2",
        f"actor_rollout_ref.actor.ppo_mini_batch_size={opcd['ppo_mini_batch_size']}",
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1",
        "actor_rollout_ref.actor.use_dynamic_bsz=False",
        f"actor_rollout_ref.actor.ppo_max_token_len_per_gpu={student_context}",
        "actor_rollout_ref.actor.use_kl_loss=False",
        "actor_rollout_ref.actor.use_torch_compile=False",
        f"actor_rollout_ref.actor.fsdp_config.fsdp_size={opcd['actor_gpus_per_node']}",
        "actor_rollout_ref.actor.fsdp_config.param_offload=True",
        "actor_rollout_ref.actor.fsdp_config.optimizer_offload=True",
        f"actor_rollout_ref.rollout.name={opcd['rollout_engine']}",
        "actor_rollout_ref.rollout.n=1",
        "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1",
        f"actor_rollout_ref.rollout.tensor_model_parallel_size={opcd['rollout_tensor_parallel_size']}",
        f"actor_rollout_ref.rollout.max_model_len={student_context}",
        f"actor_rollout_ref.rollout.max_num_batched_tokens={student_context}",
        f"actor_rollout_ref.rollout.max_num_seqs={opcd['rollout_max_num_seqs']}",
        f"actor_rollout_ref.rollout.gpu_memory_utilization={opcd['gpu_memory_utilization']}",
        "actor_rollout_ref.rollout.enable_chunked_prefill=True",
        "actor_rollout_ref.rollout.enable_prefix_caching=False",
        f"actor_rollout_ref.rollout.agent.num_workers={opcd['agent_loop_workers']}",
        "+actor_rollout_ref.rollout.agent.agent_loop_manager_class="
        "verl.trainer.pace_plus_agent_loop.PacePlusAgentLoopManager",
        "algorithm.use_kl_in_reward=False",
        "reward.reward_model.enable=False",
        "reward.num_workers=1",
        f"reward.custom_reward_function.path={reward_path}",
        "reward.custom_reward_function.name=compute_score",
        "distillation.enabled=True",
        f"distillation.n_gpus_per_node={opcd['teacher_gpus_per_node']}",
        f"distillation.nnodes={opcd['nnodes']}",
        "distillation.teacher_models.teacher_model.model_path="
        f"{get_required(config, 'models.teacher.path')}",
        "distillation.teacher_models.teacher_model.inference.tensor_model_parallel_size="
        f"{opcd['teacher_tensor_parallel_size']}",
        "distillation.teacher_models.teacher_model.inference.data_parallel_size=1",
        "distillation.teacher_models.teacher_model.inference.gpu_memory_utilization="
        f"{opcd['teacher_gpu_memory_utilization']}",
        f"distillation.teacher_models.teacher_model.inference.max_model_len={teacher_context}",
        f"distillation.teacher_models.teacher_model.inference.max_num_batched_tokens={teacher_context}",
        "distillation.teacher_models.teacher_model.inference.max_num_seqs="
        f"{opcd['teacher_max_num_seqs']}",
        "distillation.teacher_models.teacher_model.inference.enforce_eager=True",
        f"distillation.distillation_loss.loss_mode={opcd['distillation_loss_mode']}",
        "distillation.distillation_loss.use_task_rewards=False",
        "distillation.distillation_loss.use_policy_gradient=False",
        "distillation.distillation_loss.loss_max_clamp=10.0",
        "distillation.distillation_loss.log_prob_min_clamp=-10.0",
        "+trainer.pace_plus=True",
        f"+trainer.pace_plus_json_schema_path={schema_path}",
        f"+trainer.pace_plus_teacher_path={get_required(config, 'models.teacher.path')}",
        f"+trainer.pace_plus_student_path={get_required(config, 'models.student.path')}",
        f"+trainer.pace_plus_pilot_summary={summary}",
        f"+trainer.pace_plus_pilot={str(pilot).lower()}",
        f"+trainer.pace_plus_experience_path={experience_path}",
        f"+trainer.pace_plus_teacher_max_model_len={teacher_context}",
        "trainer.val_before_train=False",
        "trainer.critic_warmup=0",
        "trainer.logger=['console']",
        "trainer.project_name=PACE_PLUS",
        f"trainer.resume_mode={'disable' if pilot else 'auto'}",
        f"trainer.save_freq={-1 if pilot else opcd['save_freq']}",
        f"trainer.n_gpus_per_node={opcd['actor_gpus_per_node']}",
        f"trainer.nnodes={opcd['nnodes']}",
        "trainer.test_freq=-1",
        "trainer.total_epochs=1",
        f"trainer.total_training_steps={steps}",
        f"trainer.default_local_dir={output_dir}",
    ]
    if pilot:
        command.append(f"+trainer.rollout_data_dir={output_dir / 'pilot_rollouts'}")
    return command


def run_training(config: Mapping[str, Any], data: str | Path, *, pilot: bool) -> None:
    preflight = _preflight(config)
    experience_path = project_path(
        config,
        config.get("experience_pool", {}).get(
            "path", "experience_pool/experience_pool.json"
        ),
    )
    load_active_experience_text(experience_path)
    data_path = Path(data).expanduser().resolve()
    if data_path.suffix == ".parquet":
        parquet = data_path
    else:
        parquet = project_path(config, "training/pilot.parquet" if pilot else "training/opcd_train.parquet")
        limit = int(config.get("pilot", {}).get("samples", 20)) if pilot else None
        prepare_data(config, data_path, parquet, split="train", limit=limit)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT / "verl") + os.pathsep + env.get("PYTHONPATH", "")
    try:
        subprocess.run(
            _training_command(config, parquet, experience_path, pilot=pilot),
            cwd=PROJECT_ROOT,
            env=env,
            check=True,
        )
    except BaseException:
        if pilot:
            _write_summary(config, {"status": "training_failed", "training_step_success": False})
        raise
    if pilot:
        after = validate_model_pair(
            get_required(config, "models.teacher.path"),
            get_required(config, "models.student.path"),
        )
        teacher_unchanged = preflight["teacher"]["weight_manifest"] == after["teacher"]["weight_manifest"]
        _write_summary(config, {"teacher_unchanged": teacher_unchanged})
        summary_path = project_path(config, config.get("pilot", {}).get("summary_path", "pilot_summary.json"))
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        passed = all(
            summary.get(key) is True
            for key in ("rollout_valid", "kl_finite", "training_step_success", "teacher_unchanged")
        )
        _write_summary(config, {"status": "passed" if passed else "failed"})
        if not passed:
            raise RuntimeError(f"PACE_PLUS pilot acceptance failed; inspect {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs/pace_plus_msd.yaml"))
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("preflight")

    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--data", required=True)
    validate_parser.add_argument("--require-inline-reference", action="store_true")

    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--data", required=True)
    prepare_parser.add_argument("--output", required=True)
    prepare_parser.add_argument("--split", default="train")
    prepare_parser.add_argument("--limit", type=int)

    extract_parser = subparsers.add_parser("extract")
    extract_parser.add_argument("--data", required=True)
    extract_parser.add_argument("--limit", type=int)
    extract_parser.add_argument("--pilot", action="store_true")
    extract_parser.add_argument("--teacher-model")
    extract_parser.add_argument("--student-model")

    reasoning_parser = subparsers.add_parser("generate-reasonings")
    reasoning_parser.add_argument("--data", required=True)
    reasoning_parser.add_argument("--limit", type=int)
    reasoning_parser.add_argument("--pilot", action="store_true")
    reasoning_parser.add_argument("--teacher-model")
    reasoning_parser.add_argument("--student-model")
    consolidate_parser = subparsers.add_parser("consolidate")
    consolidate_parser.add_argument("--limit", type=int)
    consolidate_parser.add_argument("--teacher-model")
    consolidate_parser.add_argument(
        "--rebuild-from-normalizations", action="store_true"
    )
    consolidate_parser.add_argument("--fresh", action="store_true")
    consolidate_parser.add_argument("--group")

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--data", required=True)
    train_parser.add_argument("--pilot", action="store_true")
    train_parser.add_argument("--group")
    train_parser.add_argument("--teacher-model")
    train_parser.add_argument("--student-model")

    eval_parser = subparsers.add_parser("eval")
    eval_parser.add_argument("--checkpoint", required=True)
    eval_parser.add_argument("--data", required=True)
    eval_parser.add_argument("--limit", type=int)

    benchmark_parser = subparsers.add_parser("benchmark")
    benchmark_parser.add_argument("--devices", default="0,1")
    benchmark_parser.add_argument(
        "--roles", nargs="+", choices=("teacher", "student"), default=("teacher", "student")
    )
    benchmark_parser.add_argument(
        "--datasets", nargs="+", default=("mmsd2", "sarcnet", "docmsu", "redeval")
    )
    benchmark_parser.add_argument("--limit", type=int)
    benchmark_parser.add_argument("--retry-invalid", action="store_true")
    benchmark_parser.add_argument("--teacher-model")
    benchmark_parser.add_argument("--student-model")
    benchmark_parser.add_argument("--output")
    benchmark_parser.add_argument("--teacher-experience-pool")
    benchmark_parser.add_argument("--expected-active-experiences", type=int)

    args = parser.parse_args()
    config = load_config(args.config)
    teacher_model = getattr(args, "teacher_model", None) or os.environ.get("PACE_PLUS_TEACHER_MODEL")
    student_model = getattr(args, "student_model", None) or os.environ.get("PACE_PLUS_STUDENT_MODEL")
    if teacher_model:
        config["models"]["teacher"]["path"] = str(project_path(config, teacher_model))
    if student_model:
        config["models"]["student"]["path"] = str(project_path(config, student_model))

    if args.command == "preflight":
        _json(_preflight(config))
    elif args.command == "validate":
        _json(validate_data(args.data, require_reference=args.require_inline_reference))
    elif args.command == "prepare":
        _json(prepare_data(config, args.data, args.output, split=args.split, limit=args.limit))
    elif args.command == "extract":
        limit = int(config.get("pilot", {}).get("samples", 20)) if args.pilot else args.limit
        result = run_extraction(config, args.data, limit=limit)
        if args.pilot:
            _write_summary(config, result)
        _json(result)
    elif args.command == "generate-reasonings":
        limit = int(config.get("pilot", {}).get("samples", 20)) if args.pilot else args.limit
        result = run_reasoning_generation(config, args.data, limit=limit)
        if args.pilot:
            _write_summary(config, result)
        _json(result)
    elif args.command == "consolidate":
        if args.group:
            groups = config.get("experience_groups", {})
            if not isinstance(groups, Mapping) or args.group not in groups:
                available = (
                    ", ".join(sorted(groups)) if isinstance(groups, Mapping) else ""
                )
                raise RuntimeError(
                    f"unknown experience group {args.group!r}; available: {available}"
                )
            group = groups[args.group]
            if not isinstance(group, Mapping):
                raise RuntimeError(
                    f"experience_groups.{args.group} must be an object"
                )
            required_group_fields = {"data_path", "extraction_path", "pool_path"}
            missing_group_fields = required_group_fields - set(group)
            if missing_group_fields:
                raise RuntimeError(
                    f"experience_groups.{args.group} is missing required fields: "
                    f"{', '.join(sorted(missing_group_fields))}"
                )
            config.setdefault("experience_extraction", {})["output_path"] = group[
                "extraction_path"
            ]
            config.setdefault("experience_pool", {})["path"] = group["pool_path"]
            config["experience_pool"]["candidate_sample_path"] = group["data_path"]
        result = run_weighted_reference_pipeline(
            config,
            limit=args.limit,
            fresh=args.fresh,
            rebuild_from_normalizations=args.rebuild_from_normalizations,
        )
        if args.group:
            result["group"] = args.group
        _write_summary(
            config,
            {
                "experience_pool_built": result["processed"] == result["candidates"],
                "experience_pool_active": result["active"],
                "experience_pool_processed": result["processed"],
                "experience_pool_total_candidates": result["total_candidates"],
            },
        )
        _json(result)
    elif args.command == "train":
        if args.group:
            groups = config.get("experience_groups", {})
            if not isinstance(groups, Mapping) or args.group not in groups:
                available = (
                    ", ".join(sorted(groups)) if isinstance(groups, Mapping) else ""
                )
                raise RuntimeError(
                    f"unknown experience group {args.group!r}; available: {available}"
                )
            group = groups[args.group]
            if not isinstance(group, Mapping):
                raise RuntimeError(
                    f"experience_groups.{args.group} must be an object"
                )
            config.setdefault("experience_pool", {})["path"] = group["pool_path"]
            training_root = config.setdefault("output", {}).get(
                "training_dir", "training"
            )
            config["output"]["training_dir"] = str(
                Path(training_root) / args.group
            )
        run_training(config, args.data, pilot=args.pilot)
    elif args.command == "eval":
        _json(run_evaluation(config, args.checkpoint, args.data, limit=args.limit))
    elif args.command == "benchmark":
        if args.output:
            output = str(Path(args.output).expanduser().resolve())
            config["benchmark_evaluation"]["output_path"] = output
            config["benchmark_evaluation"]["pause_file"] = str(Path(output) / ".pause")
        _json(
            run_benchmark(
                config,
                devices=[value.strip() for value in args.devices.split(",")],
                roles=args.roles,
                datasets=args.datasets,
                limit=args.limit,
                retry_invalid=args.retry_invalid,
                teacher_experience_pool=args.teacher_experience_pool,
                expected_active_experiences=args.expected_active_experiences,
            )
        )


if __name__ == "__main__":
    main()
