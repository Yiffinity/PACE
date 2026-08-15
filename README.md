# PACE

Implementation of **Progressive Adaptation through Curated Experience** for
multimodal sarcasm detection.

The frozen implementation contract is documented in [plan.md](plan.md).

## Current status

Milestone 1 is in progress. The package currently provides configuration,
strict schemas, secure API key loading, atomic artifact utilities, context
budget checks, and a diagnostic CLI. Paid API calls are not made by setup or
tests.

## Development setup

```bash
python3 -m venv .venv
.venv/bin/pip install -e '.[dev]'
.venv/bin/pytest
```

Install data and training dependencies only on machines that need them:

```bash
.venv/bin/pip install -e '.[data,train,dev]'
```

Validate the base configuration:

```bash
.venv/bin/pace validate-config --config configs/base.yaml
.venv/bin/pace doctor --config configs/base.yaml
```

Use an overlay for a provider or experiment:

```bash
.venv/bin/pace validate-config \
  --config configs/base.yaml \
  --overlay configs/api/micu.yaml \
  --overlay configs/experiments/smoke.yaml
```

API key files contain one key per non-empty line and must have mode `600`.
Their contents are never written to configs, logs, caches, or checkpoints.

