# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research codebase for detecting and steering temporal awareness in LLMs. Investigates whether LLMs encode intertemporal preference in their activations, and whether those preferences can be steered via activation engineering.

## Commands

```bash
# Install
pip install -e ".[dev]"

# Run tests
pytest                          # all tests
pytest tests/inference/         # specific module
pytest -m "not slow"            # skip slow tests (multi-architecture/benchmark)
pytest tests/inference/test_model_runner_unit.py::TestModelRunner::test_encode  # single test

# Lint & format
black .
ruff check .

# Probe training & evaluation
python scripts/probes/train_temporal_probes_caa.py
python scripts/probes/validate_dataset_split.py

# Intertemporal experiment pipeline
python scripts/intertemporal/generate_prompt_dataset.py
python scripts/intertemporal/query_llm_preference.py
python scripts/intertemporal/run_intertemporal_experiment.py
```

## Architecture

### `src/` — Core Library

```
src/
├── inference/           # Model loading and inference
│   ├── model_runner.py  # ModelRunner: primary interface to all models
│   ├── backends/        # Backend adapters: TransformerLens, NNsight, Pyvene, HuggingFace, MLX
│   ├── interventions/   # Activation intervention types and factory
│   └── generated_trajectory.py  # GeneratedTrajectory: token logprobs over a sequence
├── binary_choice/       # BinaryChoiceRunner (extends ModelRunner)
│   └── binary_choice_runner.py  # Forced-continuation experiments → LabeledSimpleBinaryChoice
├── activation_patching/ # Coarse activation patching (layer/component sweeps)
├── attribution_patching/ # EAP and EAP-IG circuit attribution
├── intertemporal/       # Intertemporal preference experiments
│   ├── common/          # ContrastivePair, PreferenceTypes, project_paths.py
│   ├── data/            # Dataset configs and loading
│   └── experiments/     # ExperimentContext for full pipeline runs
└── common/              # Shared utilities
    ├── base_schema.py   # BaseSchema: all dataclasses inherit this
    ├── auto_export.py   # auto_export(): auto-populate __init__.py __all__
    ├── choice/          # BinaryChoice, GroupedBinaryChoice, SimpleBinaryChoice
    ├── math/            # Entropy, diversity, trajectory metrics
    ├── token_tree.py    # TokenTree: branching structure for binary choices
    └── analysis/        # Metrics and analysis over TokenTrees
```

### Key Abstractions

- **`ModelRunner`** (`src/inference/model_runner.py`): Wraps any supported model and backend. Entry point for all inference (forward, generate, run_with_cache, run_with_intervention). Auto-detects chat/reasoning models. Never use `self._model` directly outside of ModelRunner and its backends.

- **`BinaryChoiceRunner`** (`src/binary_choice/binary_choice_runner.py`): Extends `ModelRunner` with `choose()` for forced-continuation binary preference experiments. Returns a `LabeledSimpleBinaryChoice` with a `TokenTree`.

- **`BaseSchema`** (`src/common/base_schema.py`): All dataclasses must inherit this. Provides `.to_dict()`, `.from_dict()`, `.from_json()`, and deterministic `.get_id()` via BLAKE2b hashing.

- **`auto_export`** (`src/common/auto_export.py`): Used in every `__init__.py` to auto-expose all public symbols. Pattern: `__all__ = auto_export(__file__, __name__, globals())`.

- **Outputs** are written to `out/` (gitignored): `out/experiments/`, `out/preference_datasets/`, `out/prompt_datasets/`. Paths are managed via `src/intertemporal/common/project_paths.py`.

### Backends

Multiple inference backends are supported, selected via `ModelBackend` enum: `TRANSFORMERLENS`, `NNSIGHT`, `PYVENE`, `HUGGINGFACE`, `MLX`. Backend selection is automatic via `get_recommended_backend_inference()` or explicit in `ModelRunner(backend=...)`.

## Code Style

1. **All imports always on top** — no inline or function-level imports unless strictly necessary for circular dependency resolution.

2. **Use `auto_export` in all `__init__.py` files** — add `__all__ = auto_export(__file__, __name__, globals())`.

3. **Inherit `BaseSchema` for all dataclasses** that need serialization.

4. **Clean code** — no dead code, no commented-out code, no debug prints.
