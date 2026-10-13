These MIT-licensed components come from https://github.com/CognitiveAISystems/RATE
at commit `8abce2dce9ceacd411d46d4a99c287f7975892f5` (retrieved 2026-10-10).
The upstream LICENSE is preserved here.

- `elmur/`: `offline_rl_baselines/ELMUR/`, including its architectural README.
- `env_encoders/`: `RATE/env_encoders/`.
- `tmaze.py`: `src/envs/tmaze/tmaze.py`.
- `tmaze_dataset.py`: `src/envs_datasets/tmaze_dataset.py`.

Python import namespaces were adjusted to `elmur_klpo.vendor` and files formatted
with Ruff. The algorithms are otherwise unchanged. Upstream examples in the
vendored README describe the original RATE framework, not this package's CLI.
