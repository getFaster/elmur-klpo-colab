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

The T-Maze module imports Gymnasium 0.29.1 as `gym` instead of the retired
Gym package. This changes the environment base class, spaces, and seeding
provider; its existing reset/step return conventions and task logic are
preserved for the upstream dataset generator and Step 2 callers.

Step 3 repair additionally preserves byte-exact reference sources inside
`rate_reference/sources.zip` at the same commit. `rate_reference/manifest.json` records
SHA-256 hashes. This includes the raw model and its layers/encoders, trainer,
scheduler, dataloaders, native command, and README. Parity tests import an
independent copy of the raw model/layers/encoders by changing import prefixes
in memory; stored reference files are not edited or formatted. `rate_lr_scheduler.py` is the same scheduler, formatted with Ruff. Their original
MIT license is the LICENSE in this directory. The active pre-existing vendored
model remains unchanged.
