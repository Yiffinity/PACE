# PACE Implementation Plan

## 1. Objective

Implement a runnable, reproducible version of **PACE: Progressive Adaptation
through Curated Experience** for multimodal sarcasm detection.

The implementation has three operational phases:

1. **Phase A - Experience Curation**: an external multimodal API proposes
   transferable experiences, and a frozen local teacher validates their utility.
2. **Phase B0 - Filtered Format and Task SFT**: an external API predicts without
   gold labels or experiences; only correct, valid JSON generations become SFT
   targets for the student.
3. **Phase B1 - On-Policy Experience Distillation (OPD)**: a student that cannot
   read the experience pool learns from the token distribution of an
   experience-enhanced frozen teacher on the student's own prefixes.

The first delivery target is a complete, testable implementation and a small
end-to-end smoke run. Full-scale experiments are enabled by configuration but
are not required for the first successful run.

## 2. Fixed Method Decisions

### 2.1 Output contract

All model-generated content and experiences are in English. Normal inference,
B0 SFT targets, and student rollouts use strict JSON:

```json
{
  "visual_evidence": "...",
  "textual_evidence": "...",
  "explanation": "...",
  "label": "sarcastic"
}
```

`label` must be exactly `sarcastic` or `non-sarcastic`.

### 2.2 Local model candidates

```text
/home/yjt/model/Qwen3-VL-2B-Instruct
/home/yjt/model/Qwen3-VL-4B-Instruct
/home/yjt/model/Qwen3-VL-8B-Instruct
```

All three checkpoints are Qwen3-VL models and share the same tokenizer. The
main OPD matrix is:

| Teacher | Student | Experience pool |
| --- | --- | --- |
| 8B | 4B | `pool_teacher_8b` |
| 8B | 2B | `pool_teacher_8b` |
| 4B | 2B | `pool_teacher_4b` |

`pool_teacher_2b` is built and evaluated for direct inference. `2B -> 2B`
self-distillation is implemented as a configuration but deferred unless an
additional experiment is needed.

### 2.3 Retrieval model

Use the frozen checkpoint:

```text
/home/yjt/model/siglip2-so400m-patch14-384
```

Image and text embeddings are L2-normalized and fused with configurable
weights, defaulting to `0.5 / 0.5`.

### 2.4 External API providers

Primary provider:

```yaml
base_url: https://www.micuapi.ai/v1
model: gpt-5.5
key_file: /home/yjt/PACE/apikey/gpt_apikey
```

Manually selected backup provider:

```yaml
base_url: https://api.siliconflow.cn/v1
model: Qwen/Qwen3.5-397B-A17B
key_file: /home/yjt/PACE/apikey/qwen_apikey
```

There is no automatic provider failover. A run is pinned to one provider.
Keys are stored one per non-empty line, loaded only at runtime, redacted from
logs, and scheduled round-robin. Before implementation begins, key files are
changed from mode `664` to `600`.

### 2.5 Reproducibility defaults

```yaml
seed: 42
api_temperature: 0.0
api_stream: false
api_max_attempts: 3
allow_experience_truncation: false
```

All split manifests, prompts, schemas, model paths, generation settings,
package versions, and checkpoints are recorded with every run.

## 3. Data Protocol

### 3.1 Experience sources

Only official training splits are used to construct experience and train
students:

1. MMSD2.0 `mmsd-v2`
2. DocMSU
3. SarcNet English and Chinese

For SarcNet, `multi_label` is the binary gold label. `text_label` and
`image_label` are retained as metadata and optional analysis signals, but are
not substituted for the task label.

### 3.2 Secondary split

Within each official training split, create deterministic, non-overlapping
manifests:

```yaml
extraction: 0.60
consolidation: 0.20
opd: 0.20
```

The ratios are configurable. Splitting is stratified by source and gold label;
SarcNet is additionally stratified by language. A canonical sample identifier
contains source, official split, language where applicable, and original ID.

Hard assertions:

- No sample ID appears in more than one secondary split.
- Official test data never enters a training or validation manifest.
- Official validation data never enters experience construction, SFT, or OPD.
- RedEval, MMSD3.0, and CFMS never enter prompts used for model selection.
- Image availability and decode errors are audited before any paid API call.

### 3.3 Official validation

Official validation splits are used only to select student checkpoints. After
each configured evaluation interval, compute Macro-F1 on:

1. MMSD2.0 validation
2. DocMSU validation
3. Combined SarcNet English and Chinese validation

The checkpoint score is the unweighted mean of these three Macro-F1 values.
No final test or OOD metric participates in checkpoint selection.

### 3.4 Held-out evaluation

Evaluate final checkpoints on:

- MMSD2.0 test
- DocMSU test
- SarcNet English test
- SarcNet Chinese test
- RedEval (`0 = non-sarcastic`, `1 = sarcastic`)
- MMSD3.0 test, after missing text/images are supplied
- CFMS `testdata.json`

CFMS v1 reports binary classification, JSON validity, and saves generated
explanations. It does not automatically score sarcasm targets or explanation
quality.

## 4. Repository Layout

The intended project layout is:

```text
PACE/
  plan.md
  README.md
  pyproject.toml
  configs/
    base.yaml
    data/
    api/
    curation/
    training/
    experiments/
  src/pace/
    __init__.py
    cli.py
    config.py
    schemas.py
    data/
    api/
    retrieval/
    models/
    experience/
    training/
    evaluation/
    utils/
  scripts/
  tests/
  artifacts/              # ignored generated output
    manifests/
    api_cache/
    api_failures/
    embeddings/
    experience_pools/
    sft_targets/
    checkpoints/
    predictions/
    metrics/
```

Generated artifacts are never mixed into source directories. Every artifact
contains or references a run ID and config hash.

## 5. Core Schemas

Use typed, versioned schemas for at least:

- `MultimodalSample`
- `SarcasmAnalysis`
- `ExperienceReflection`
- `ExperienceCandidate`
- `ExperienceUtility`
- `ExperienceRecord`
- `ApiAttempt`
- `ApiFailure`
- `RunState`
- `PredictionRecord`
- `MetricReport`

An experience record must preserve:

- stable experience ID
- short English experience text
- source sample ID and source dataset
- source prediction and gold label
- teacher ID
- pool version before evaluation
- relevant/global validation IDs
- before/after gold probabilities
- per-sample deltas
- `S_rel` and `S_global`
- accepted/rejected status and reason
- prompt/schema/model/config hashes
- timestamps and resume metadata

## 6. Phase A - Experience Curation

### 6.1 API call 1: blind prediction

Input:

- image
- text
- all currently consolidated experiences for that pool

The gold label is never included. Output is strict `SarcasmAnalysis` JSON.

### 6.2 API call 2: gold-aware reflection

Input:

- image and text
- all current consolidated experiences
- the exact first response
- gold label
- whether the first label was correct

Output:

```json
{
  "corrected_analysis": {
    "visual_evidence": "...",
    "textual_evidence": "...",
    "explanation": "...",
    "label": "sarcastic"
  },
  "has_new_rule": true,
  "experience": "A short transferable rule or null"
}
```

Rules:

- A wrong blind prediction always triggers an extraction attempt.
- A correct blind prediction relies on the API's `has_new_rule` judgment.
- `experience` may be null even after a wrong prediction.
- An experience must not copy names, brands, or irrelevant sample details.
- It must state a transferable applicability condition and decision insight.
- At most one candidate is emitted per source sample.

### 6.3 Parallel batch snapshot semantics

API work is parallelized without hiding the experience dependency:

1. Freeze pool snapshot `E_r` for a curation batch.
2. Run blind predictions in parallel for all batch samples.
3. Run reflection calls in parallel after each corresponding blind response.
4. Validate all non-null candidates independently against `E_r`.
5. Commit accepted candidates in deterministic sample order.
6. The next batch observes the newly committed pool.

Default curation batch size is the number of available keys for the selected
provider. Global and per-key concurrency are configurable.

### 6.4 API cache and failure behavior

The request hash includes provider, model, prompt version, schema version,
sample ID, image digest, and experience snapshot hash. Raw successful responses
are atomically cached before parsing so resume never repeats a paid successful
request.

After three failed attempts:

- log a redacted failure record;
- append the sample to a persistent retry queue;
- skip it and continue the current pass;
- never switch providers automatically.

On resume, failed samples are processed in recorded order as a tail of the data
stream and observe the latest pool. This is intentionally auditable but may
produce a different pool from a zero-failure run.

## 7. Experience Retrieval and Utility

### 7.1 Validation pool

The three consolidation partitions form one cross-source validation pool.
They never participate in extraction, B0 SFT, or B1 OPD.

### 7.2 Global bank

Default total size is 64:

- 32 sarcastic
- 32 non-sarcastic
- as source-balanced as integer allocation permits

The bank is fixed for a manifest/config and reused for every candidate.

### 7.3 Relevant bank

For each source sample, retrieve separately from the consolidation pool:

- top-10 sarcastic samples
- top-10 non-sarcastic samples

`relevant_per_class_k` is configurable. Relevant and global banks may overlap;
overlap is not removed.

### 7.4 Gold-label probability

Utility scoring uses an internal deterministic label prompt rather than the
free-form JSON output. The teacher scores single-token codes:

```text
S -> sarcastic
N -> non-sarcastic
```

At startup, assert that the chosen verbalizers are one token and distinct for
the shared Qwen3-VL tokenizer. Normalize their two logits with a binary softmax
to obtain `P_T(y | x, E)`.

For validation sample `j` and candidate `e`:

```text
delta_j(e) = log P_T(y_j | x_j, E + e) - log P_T(y_j | x_j, E)
S_rel(e) = mean(delta_j over relevant bank)
S_global(e) = mean(delta_j over global bank)
```

Accept exactly when:

```text
S_rel > 0 and S_global >= 0
```

The frozen teacher used for utility is the same teacher used later by OPD.
This creates independent 2B, 4B, and 8B experience pools.

## 8. Experience Context Policy

V1 always passes all consolidated experiences in acceptance order.

V1 does **not** implement:

- truncation
- top-k experience retrieval for teacher prompts
- summarization or compression
- experience eviction
- context-window extension
- sharded teacher aggregation

Before every local teacher call, compute the exact token budget including
system prompt, images, text, experiences, prefix, response reserve, and safety
margin. Emit warnings at 80%, 90%, and 95% utilization.

If the prompt would exceed the context window:

1. do not run the forward pass;
2. persist experience pool, stream position, RNG state, and student checkpoint;
3. log model ID, sample ID, pool size, experience tokens, total tokens, and
   context limit;
4. raise `ExperienceContextOverflowError`.

No experience may be silently omitted. Overflow handling beyond this explicit
failure is a documented future task to be selected after observing real pool
sizes and token distributions.

## 9. Phase B0 - Filtered SFT

### 9.1 Target construction

For each OPD `source x gold class` bucket, iterate a deterministic candidate
queue:

1. Send image and text to the API.
2. Do not send gold label.
3. Do not send any experience.
4. Request one strict `SarcasmAnalysis` JSON response.
5. Compare the returned label with gold only after the response is stored.
6. Accept only valid, complete JSON with a correct label.

Rejected reasons include API failure, invalid JSON, missing fields, invalid
label, and prediction mismatch. They are recorded but not used as SFT targets.

### 9.2 Dataset profiles

| Profile | Correct targets per source/class | Nominal total |
| --- | ---: | ---: |
| smoke | 8 | 48 |
| pilot | 64 | 384 |
| full | 128 | 768 |
| extended | 256 | 1536 |

The formal default is `extended`. The target is an upper bound: if a bucket is
exhausted before collecting 256 correct unique samples, use all available
correct samples, record the shortfall, and apply a source/class-balanced
sampler during SFT. SarcNet is also balanced by language where possible.

The same accepted targets are reused for 2B and 4B student training. Students
have independent optimizer state and checkpoints.

### 9.3 SFT objective

Teacher-force the complete API JSON target. Compute cross-entropy only on
assistant output tokens; mask system, user, image, and padding positions.

## 10. Phase B1 - OPD

### 10.1 Inputs and rollout

Teacher input:

```text
image + text + all consolidated experiences + student prefix
```

Student input:

```text
image + text + student prefix
```

The student generates its own response online with defaults:

```yaml
do_sample: true
temperature: 1.0
top_p: 1.0
max_new_tokens: 256
```

Rollouts are regenerated from the current student policy. Greedy generation is
not the main training setting.

### 10.2 Reverse KL

At every student-generated response prefix, compute exact full-vocabulary:

```text
KL(student || experience-enhanced frozen teacher)
```

The loss covers response positions only. Teacher logits are detached; only the
student receives gradients. Invalid JSON rollouts are retained in the main
objective so training remains on-policy. A `drop_invalid_rollouts` switch is
implemented only for ablation.

### 10.3 Gold CE

The main performance objective is:

```text
L_main = L_reverse_KL + lambda_gold_ce * L_gold_ce
lambda_gold_ce = 1.0 by default
```

Gold CE uses the same internal `S/N` student label-scoring prompt. Gold is not
provided during rollout and does not replace the student's generated label.

Required controls:

1. B0 SFT only
2. B0 + gold CE only
3. B0 + pure OPD
4. B0 + experience OPD + gold CE (main)
5. B0 + no-experience-teacher OPD + gold CE
6. Drop-invalid-rollout ablation

## 11. Student Tuning Modes

Both tuning modes are implemented.

### 11.1 Full tuning

Update all student parameters in BF16 with gradient checkpointing. Test memory
use before launching a full run.

### 11.2 LoRA

Defaults:

```yaml
rank: 32
alpha: 64
dropout: 0.05
```

Apply LoRA to language attention/MLP and the multimodal merger. Freeze the
vision tower.

### 11.3 Selection protocol

Run paired LoRA/full pilots for `8B -> 2B` and `8B -> 4B` with identical data,
pool, seed, step budget, and checkpoint selection.

Prefer LoRA for the corresponding student size only when:

- mean validation Macro-F1 is within 1.0 absolute point of full tuning;
- no source falls more than 2.0 points below full tuning;
- JSON validity is at least 99%.

Otherwise use full tuning for formal experiments and retain LoRA as an
efficiency result.

## 12. Training Defaults

| Setting | B0 SFT | B1 OPD |
| --- | ---: | ---: |
| dtype | BF16 | BF16 |
| epochs | 3 | 3 |
| micro batch | 1 | 1 |
| gradient accumulation | 8 | 4 |
| full LR | 2e-5 | 1e-5 |
| LoRA LR | 2e-4 | 1e-4 |
| scheduler | cosine | cosine |
| warmup ratio | 0.03 | 0.03 |
| weight decay | 0.01 | 0.01 |
| max sequence length | 8192 | 8192 |
| max generated tokens | 256 | 256 |
| gradient clipping | 1.0 | 1.0 |

Every value is configurable. Smoke runs may reduce sequence length, bank size,
sample counts, and generation length, but must exercise the same code paths.

## 13. Direct Inference and Evaluation

Generation evaluation uses temperature 0. Invalid JSON or an invalid label is
counted as an incorrect prediction and separately included in validity metrics.
Do not silently repair an evaluated response.

Report per dataset:

- accuracy
- sarcastic-class precision, recall, and F1
- Macro-F1
- Weighted-F1
- JSON validity
- counts of missing/decode-failed samples

Experience reporting includes:

- candidate count
- null candidate count
- accepted/rejected count
- acceptance rate by source, class, language, and teacher
- pool size and token growth
- distributions of `S_rel`, `S_global`, and per-sample delta
- direct teacher performance with and without its own pool

Save raw predictions and parsed results so metric code can be rerun without
model inference.

## 14. Security and Reliability

- Never print, serialize, or commit API keys.
- Redact authorization headers and provider error bodies that echo secrets.
- Set key files to mode `600`.
- Use atomic writes for API cache, pool state, retry queues, and checkpoints.
- Include schema and prompt versions in cache keys.
- Validate image MIME type and decode locally before upload.
- Add timeouts, bounded retries, exponential backoff, and jitter.
- Never auto-switch providers within a run.
- Make resume idempotent for successful API responses.
- Record expected non-determinism from remote API behavior and failed-sample
  reordering.

## 15. Test Strategy

### 15.1 Unit tests

- strict JSON parsing and label normalization
- API key loading without secret leakage
- cache-key determinism
- atomic cache writes
- retry queue semantics
- deterministic stratified manifests
- split leakage assertions
- SigLIP embedding fusion and retrieval ordering
- class-balanced global/relevant banks
- single-token S/N verbalizer assertion
- utility formula and threshold boundaries
- experience pool state transitions
- context warning and overflow error behavior
- SFT loss masking
- reverse-KL numeric correctness against a small reference tensor
- gold CE composition
- invalid-rollout keep/drop switch
- metric computation including invalid JSON

### 15.2 Integration tests

- fake OpenAI-compatible server with multiple keys and controlled failures
- Phase A two-call flow with resume
- parallel batch snapshot commit ordering
- B0 filtered target collection with bucket exhaustion
- local tiny/mock teacher utility pass
- one B0 optimizer step
- one OPD rollout and optimizer step
- checkpoint evaluation and selection

### 15.3 Smoke test

Default smoke quotas:

```yaml
extraction_per_source_class: 4
consolidation_per_source_class: 64
opd_per_source_class: 16
b0_correct_per_source_class: 8
```

The smoke run must complete end to end with a dedicated artifact directory,
without being mistaken for a formal result.

## 16. Implementation Milestones

### Milestone 1 - Foundation

- Create package, configuration loader, logging, typed schemas, and CLI.
- Add dependency metadata, formatting/lint/test configuration, and `.gitignore`.
- Secure key permissions.
- Add provider configuration without making paid calls.
- Pass foundational unit tests.

### Milestone 2 - Data and manifests

- Implement dataset adapters for all local formats.
- Generate deterministic secondary splits and audits.
- Implement held-out adapters and image validation.
- Pass leakage and schema tests.

### Milestone 3 - API subsystem

- Implement multi-key async provider pool, strict schemas, cache, retries, and
  retry queue.
- Implement an opt-in capability probe using synthetic text/image input.
- Implement Phase A and B0 prompt builders.
- Validate against a fake server before a real provider.

### Milestone 4 - Retrieval and experience pools

- Implement cached SigLIP embeddings.
- Build global and relevant banks.
- Implement local teacher S/N scoring, utility calculation, pool state, and
  context telemetry.
- Complete a small candidate accept/reject integration test.

### Milestone 5 - B0 training

- Build filtered SFT targets.
- Implement full and LoRA SFT.
- Add checkpoint evaluation and selection.
- Verify JSON validity improvement on a small sample.

### Milestone 6 - OPD

- Implement online student rollout and aligned teacher-prefix forward pass.
- Implement exact reverse KL, gold CE, and required masks.
- Add full/LoRA training and resume.
- Complete one optimizer-step and tiny overfit tests.

### Milestone 7 - Evaluation and handoff

- Implement direct teacher, student, in-domain, OOD, and multi-image evaluation.
- Add ablation configs and result aggregation.
- Run the end-to-end smoke profile.
- Document commands, expected artifacts, limitations, and future overflow work.

## 17. Completion Criteria

The initial implementation is complete when:

1. All unit and integration tests pass.
2. Data manifests are deterministic and leakage checks pass.
3. A fake API end-to-end test exercises parallelism, caching, failure logging,
   and resume without duplicate successful requests.
4. At least one real API capability probe succeeds or its incompatibility is
   documented with a provider-specific adapter plan.
5. A small experience pool can be generated and validated with a frozen local
   teacher.
6. B0 produces filtered targets and completes at least one student update.
7. OPD completes an online rollout and update using reverse KL + gold CE.
8. Direct inference and metric reports work on a small held-out subset.
9. Context use is measured and no experience is silently truncated.
10. README commands reproduce the smoke workflow from a clean environment.

## 18. Explicit Future Work

The following is intentionally out of v1 scope:

- automatic strategy for experience context overflow
- distributed multi-node training
- automatic provider failover
- automatic CFMS target/explanation scoring
- full MMSD3.0 evaluation before missing content is supplied
- `2B -> 2B` self-distillation experiment
- complex human-defined sarcasm taxonomies

These items must not be silently approximated in the first implementation.
