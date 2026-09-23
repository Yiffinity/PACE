# PACE_PLUS

Code for **PACEing the Evolution: Generalizing Multimodal Sarcasm Detection via Transferable Experience and On-Policy Distillation**.

[Overview](#method-overview) · [Installation](#requirements) · [Quick start](#one-command-pipeline) · [Configuration](#configuration) · [Tests](#tests)

PACE_PLUS learns transferable sarcasm experiences from multiple source domains and distills them into a smaller multimodal model. The release contains source code, configuration, tests, and the manuscript's overview figure. Datasets, model weights, generated reasoning, and run reports are not included.

![PACE_PLUS overview](assets/overview.png)

## Method overview

1. **Multi-source Contrastive MSD Experience Mining (MCEM)** compares blind teacher and student reasoning on heterogeneous source datasets.
2. A weighted experience pool consolidates reusable sarcasm mechanisms through `ADD`, `MODIFY`, `UPVOTE`, and `DOWNVOTE`, retaining applicability conditions and exclusion boundaries.
3. **On-Policy Experience Distillation (OPED)** gives the frozen teacher access to the pool and minimizes reverse KL on student-generated trajectories, using the same generated prefix for teacher scoring.
4. The trained student performs inference from the image and text only.

The figure shows the manuscript's conceptual workflow, including its LoRA update illustration. The bundled launcher's native VeRL training configuration uses FSDP2 and does not enable LoRA by default. Existing configuration keys and filenames use `opcd` for the OPED stage.

| Component | Main implementation |
| --- | --- |
| Blind reasoning and verified teacher correction | [pace_plus_extraction.py](verl/verl/trainer/pace_plus_extraction.py) |
| Comparative cue prompts | [pace_plus_reflection.py](verl/verl/trainer/pace_plus_reflection.py) |
| Weighted experience consolidation | [pace_plus_weighted_pool.py](verl/verl/trainer/pace_plus_weighted_pool.py) |
| Experience-conditioned student trajectories | [pace_plus_agent_loop.py](verl/verl/trainer/pace_plus_agent_loop.py) |
| Pipeline configuration | [pace_plus_msd.yaml](configs/pace_plus_msd.yaml) |

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

## Configuration

| Variable | Purpose |
| --- | --- |
| `PYTHON_BIN` | Python interpreter; defaults to `python3` |
| `PACE_PLUS_CONFIG` | Pipeline YAML; defaults to `configs/pace_plus_msd.yaml` |
| `PACE_PLUS_TEACHER_MODEL` / `PACE_PLUS_STUDENT_MODEL` | Override local checkpoint directories |
| `PACE_PLUS_GROUP` | Source pair to train |
| `PACE_PLUS_TRAIN_DATA` | Optional training JSONL or parquet for that pair |
| `CUDA_VISIBLE_DEVICES` | Visible GPU IDs; defaults to `0,1` |

Change generation limits, pool capacity, and training steps in the YAML. The default recipe selects source samples for mining and trains on the selected pair; it does not claim to reproduce every manuscript setting automatically. Custom training files must preserve the source/target separation.

Optional runtime features use explicit environment settings: set `VERL_SHM_CACHE_DIR` before enabling shared-memory checkpoint caching, or `ASCEND_HOME_PATH` for a CANN toolkit installation. Neither feature is enabled by the default NVIDIA recipe.

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

## Tests

Run the public tests without launching GPU training:

```bash
python3 -m unittest discover -s tests -p 'test_*.py'
```

Tests that need local checkpoints or benchmark datasets skip when those optional resources are absent. The launcher tests verify stage order, source-pair selection, and that pending correction review prevents training.

## Anonymous release and license

Project paths in the public configuration are relative to the repository or supplied through environment variables. Keep local credentials, checkpoints, dataset caches, generated outputs, and review records outside version control. The release omits project-author and submission metadata; upstream copyright and attribution notices are retained.

The overview figure is derived from the accompanying manuscript. See [LICENSE](LICENSE) for the project license and [the vendored VeRL license](verl/LICENSE) and [notices](verl/Notice.txt) for third-party attribution.
