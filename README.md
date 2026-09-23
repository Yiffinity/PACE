# PACE_PLUS

Code for **PACEing the Evolution: Generalizing Multimodal Sarcasm Detection via Transferable Experience and On-Policy Distillation**.

[Installation](#requirements) · [Quick start](#one-command-pipeline) · [Direct CLI](#direct-cli)

![PACE_PLUS overview](assets/overview.png)

## Repository layout

```text
configs/                 YAML configuration and the reasoning JSON schema
prompts/                 Readable comparative prompt checked against the code
scripts/                 Pipeline and individual-stage launchers
tools/                   Data preparation and command-line entry points
verl/                    Training runtime and PACE_PLUS trainer extensions
assets/                  Paper overview figure used in this README
tests/                   Unit and contract tests
```

Generated samples, model checkpoints, caches, logs, evaluation reports, and other run artifacts are excluded from the public release through `.gitignore`. Place them in the local directories described below when running the code.

## Requirements

- Linux with NVIDIA CUDA and two visible GPUs for the full teacher/student pipeline.
- Python 3.10 or newer.
- One compatible multimodal teacher checkpoint and one smaller student checkpoint.
- The source datasets with their image files.

The default configuration expects the following repository-relative model layout:

```text
models/
├── Qwen3.5-9B/       # frozen teacher
└── Qwen3.5-4B/       # trainable student
```

You can use different checkpoints without editing the repository:

```bash
export PACE_PLUS_TEACHER_MODEL=models/Qwen3.5-9B
export PACE_PLUS_STUDENT_MODEL=models/Qwen3.5-4B
```

Install the runtime from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
bash setup.sh
```

The core versions in [requirements.txt](requirements.txt) match the inspected local runtime: PyTorch 2.11.0, vLLM 0.20.2, Transformers 5.16.1, and Ray 2.58.0. The installer also resolves the vendored runtime's declared dependencies. For an existing environment, set `PYTHON_BIN` to its interpreter. GPU execution still requires compatible CUDA drivers and enough memory for both checkpoints.

## Data layout

The default split builder reads the following files:

```text
datasets/
├── MMSD2.0/data/text_json_final/{train,valid,test}.json
├── MMSD2.0/data/dataset_image/
├── DocMSU/data/{train,test}.jsonl
├── DocMSU/docmsu_all.json
└── sarcnet/data/{en,zh}/{train,valid,test}.jsonl
```

Each JSON/JSONL sample must provide an image path, post text, and a binary label. Relative image paths are resolved relative to the annotation file. Dataset licenses and access conditions remain the responsibility of the user.

## One-command pipeline

From the repository root, run:

```bash
bash scripts/run_pace_plus.sh
```

The launcher checks checkpoint metadata, prepares shared source samples, runs blind teacher/student reasoning, extracts comparative cues, consolidates the three source-pair pools, and starts OPED training for `mmsd2_docmsu` by default. Each pool filters the shared extraction by its own source IDs; training reads only the selected pair. Completed reasoning records and consolidation actions are reused when their fingerprints match.

The implementation requires independent review of label-conditioned teacher corrections. If reviews are pending, extraction stops before consolidation and training. Review the image, text, gold label, and corrected reasoning in `experience_extraction/pairwise_train_shared/teacher_correction_review_queue.jsonl`, save decisions to `teacher_correction_reviews.jsonl` in the same directory, then rerun the command. The pipeline is not fully unattended when this queue is nonempty.

<details>
<summary>Correction review record format</summary>

Write one JSON object per reviewed sample. Copy `sample_id` and `correction_fingerprint` from the queue so stale reviews cannot be reused. For an accepted correction:

```json
{"sample_id":"<id-from-queue>","correction_fingerprint":"<fingerprint-from-queue>","decision":"ACCEPT","revised_reasoning":null,"rationale":"<evidence-based review rationale>"}
```

For `REVISE`, supply a `revised_reasoning` object matching [the reasoning schema](configs/pace_plus_reasoning.schema.json), including the correct label. All queued corrections must be reviewed before extraction proceeds.

</details>

To run only a specific stage:

```bash
bash scripts/run_pace_plus.sh preflight
bash scripts/run_pace_plus.sh prepare
bash scripts/run_pace_plus.sh reason
bash scripts/run_pace_plus.sh extract
bash scripts/run_pace_plus.sh consolidate
bash scripts/run_pace_plus.sh train
```

Useful environment overrides are repository-relative by default:

```bash
PACE_PLUS_GROUP=mmsd2_docmsu \
PACE_PLUS_TRAIN_DATA=data/pairwise_experience_splits/mmsd2_docmsu.jsonl \
CUDA_VISIBLE_DEVICES=0,1 \
bash scripts/run_pace_plus.sh train
```

The three supported source pairs are `mmsd2_docmsu`, `mmsd2_sarcnet`, and `docmsu_sarcnet`. Keep the remaining source dataset and RedEval out of mining and training for each corresponding evaluation setting. Benchmark dataset selection is explicit; the default benchmark command enumerates all configured datasets.

## Direct CLI

The main command is `tools/pace_plus_cli.py`. It exposes the same stages for automation:

```bash
python3 tools/pace_plus_cli.py --config configs/pace_plus_msd.yaml preflight
python3 tools/pace_plus_cli.py --config configs/pace_plus_msd.yaml generate-reasonings \
  --data data/pairwise_experience_splits/all_selected.jsonl
python3 tools/pace_plus_cli.py --config configs/pace_plus_msd.yaml extract \
  --data data/pairwise_experience_splits/all_selected.jsonl
python3 tools/pace_plus_cli.py --config configs/pace_plus_msd.yaml consolidate \
  --group mmsd2_docmsu
python3 tools/pace_plus_cli.py --config configs/pace_plus_msd.yaml train \
  --data data/pairwise_experience_splits/mmsd2_docmsu.jsonl --group mmsd2_docmsu
```

All public prompts enforce the same information boundary as the method: blind model calls receive the raw image and text; gold labels are introduced only for verified correction and training supervision; the student does not receive the consolidated experience pool at inference time.
