# Data preparation hints for RATE / ELMUR

Consult the [official RATE repository](https://github.com/CognitiveAISystems/RATE/tree/main)
before generating replacement trajectories or training readers from scratch.
RATE includes ELMUR, dataset-download instructions, data-collection scripts,
dataset loaders and task-specific training configurations.

## Start with the released trajectories

1. Check RATE's [data collection and download instructions](https://github.com/CognitiveAISystems/RATE/blob/main/README.md#data-collection).
2. Locate the exact task in the released data and inspect its upstream loader
   and training configuration. Record the source revision and dataset revision.
3. Prefer the corresponding released expert/demonstration trajectories for
   offline reader preparation. Verify collector quality rather than assuming
   every released trajectory is expert-quality.
4. Treat dataset access, offline reader training, native environment execution
   and closed-loop evaluation as separate capabilities. A renderer failure
   does not demonstrate that expert trajectories are unavailable. Offline
   training on recorded RGB/actions does not require rendering new episodes.

## POPGym / AutoencodeEasy

RATE's README provides a downloadable POPGym archive. Adapt its commands to
keep large files under this repository's ignored `runs/` directory:

```bash
mkdir -p runs/step3_data
wget https://huggingface.co/datasets/avanturist/popgym-datasets-48-tasks/resolve/main/POPGym.zip -O runs/step3_data/POPGym.zip
unzip -q runs/step3_data/POPGym.zip -d runs/step3_data/
```

The dataset source is
[avanturist/popgym-datasets-48-tasks](https://huggingface.co/datasets/avanturist/popgym-datasets-48-tasks).
Inspect the extracted layout and locate `popgym-AutoencodeEasy-v0` before
configuring a loader. RATE's scripts refer to `data/POPGym/popgym-AutoencodeEasy-v0/`;
adapt that prefix to the actual local extraction directory.

Relevant upstream code:

- [POPGym dataset loader](https://github.com/CognitiveAISystems/RATE/blob/main/src/envs_datasets/popgym_dataset.py): loads `obs`, `action`, `reward` and `done` arrays.
- [POPGym training configurations](https://github.com/CognitiveAISystems/RATE/blob/main/run_experiments/POPGym.sh).
- [POPGym collection script](https://github.com/CognitiveAISystems/RATE/blob/main/src/additional/gen_popgym_data/gen_popgym_data.py): downloads PPO collector checkpoints and generates trajectories. Collector checkpoints are distinct from the released trajectory archive.

Generating native Autoencode episodes and obtaining suit labels from internal
environment state is a different data source. Describe it explicitly; do not
report it as using RATE's downloaded demonstrations.

## MIKASA / RememberColor3-v0

Use RATE's original RL task `RememberColor3-v0` and matching recorded RGB/action
trajectories. Start with its
[ELMUR RememberColor3 configuration](https://github.com/CognitiveAISystems/RATE/blob/main/run_experiments/MIKASA_Robo/ELMUR_1/rc3.sh)
and [MIKASA dataset loader](https://github.com/CognitiveAISystems/RATE/blob/main/src/envs_datasets/mikasa_robo_dataset.py).
The configuration expects a directory ending in
`MIKASA-Robo/unbatched/RememberColor3-v0`.

The loader reads `rgb`, `action`, `success` and `done`. It transposes recorded
RGB from channels-last to channels-first and optionally divides by 255.
It uses recorded **success signals** for its return-to-go inputs. Check these
choices against the selected model configuration; do not silently substitute
success signals for native dense rewards in manager-training experiments.

The matching original RL release was located, downloaded and validated on
2026-10-13:

- Repository: [avanturist/mikasa-robo](https://huggingface.co/datasets/avanturist/mikasa-robo).
- Archive: [RememberColor3-v0.zip](https://huggingface.co/datasets/avanturist/mikasa-robo/resolve/c362f52143c61e7f778bdb5d410ce4c22a7185d4/RememberColor3-v0.zip).
- Dataset revision: `c362f52143c61e7f778bdb5d410ce4c22a7185d4`.
- Published and verified SHA-256: `388b16d0984cb19d6a6cd20fe0029a67fef100aed378a195435f1f0acf5b7d82`.
- Local episodes: `runs/step3_data/remember_color_original/RememberColor3-v0/`.
- Validation manifest: [experiments/step3/remember_color_dataset.json](experiments/step3/remember_color_dataset.json).

Reproduce the pinned download without installing a simulator:

```bash
UV_CACHE_DIR=/tmp/elmur-klpo-uv-cache uv run --with huggingface_hub python - <<'PY'
from huggingface_hub import hf_hub_download

hf_hub_download(
    repo_id="avanturist/mikasa-robo",
    repo_type="dataset",
    filename="RememberColor3-v0.zip",
    revision="c362f52143c61e7f778bdb5d410ce4c22a7185d4",
    local_dir="runs/step3_data/remember_color_download",
)
PY
mkdir -p runs/step3_data/remember_color_original
unzip -q runs/step3_data/remember_color_download/RememberColor3-v0.zip -d runs/step3_data/remember_color_original/
```

The archive contains 1,000 episodes of 60 steps each. Every episode was read
and validated: RGB is `uint8[60,128,128,6]`, actions are `float32[60,8]`, and
reward/success/done are length 60. All episodes record at least one success
and exactly one done flag at the final step. The archive checksum matches its
published Hugging Face LFS hash. This verifies the downloaded data, not a new
closed-loop policy evaluation.

Raw stored actions include values outside `[-1,1]` (11,177 of 480,000 action
elements). Preserve the source data and explicitly reconcile training targets
with the upstream controller's clipping/scaling semantics; do not silently
alter the arrays or assume the current bounded reader reproduces every raw
target. Dataset files do not provide explicit episode seeds, so shared-source
seed independence must not be claimed from filename splits alone.

Follow the [original MIKASA project](https://github.com/CognitiveAISystems/MIKASA-Robo)
for the corresponding collector and controller implementation. Newer
`RememberColor3-VLA-v0` datasets are a different task/version and must not be
silently substituted for the original RL benchmark.

## Validate and report provenance

Record source URLs, revisions, download/extraction commands, checksums,
trajectory counts, observation/action shapes, normalization, reward definitions,
episode boundaries and any filtering. Inspect complete episodes before training.
Keep train/validation/test splits disjoint by trajectory and, where available,
collector/environment seed. Keep raw archives and large arrays under ignored
`runs/`; commit only compact manifests and measurements.

The initial Step 3 implementation generated Autoencode episodes with native
labels and attempted online RememberColor preparation. It did not download
RATE's trajectory datasets. Its failed Autoencode readiness gate therefore
does not evaluate a reader prepared from RATE's released demonstrations.
The repaired offline path is `elmur_klpo/step3/preparation.py`, with the
recorded-data loader in `offline_preparation.py`, configured by
`configs/step3_remembercolor_reader.json`. It preserves the existing
800/100/100 filename-disjoint split and raw, unclipped action targets. The
reader now uses the pinned upstream Tanh output head and RGB division by 255;
raw target values outside its output bounds remain in the MSE. The former
unbounded-head variant and reconstruction loss are historical. See the new
[repair report](step_3_implementation.md) for the exact 9,600-update manual
command and changes to section training/checkpoints. Use a fresh output root.
The held-out test split is not used for checkpoint selection; offline imitation
cannot satisfy the separate native-return memory-readiness gate.

The initial GPU run uses a separate Python 3.12 / PyTorch 2.7.1+cu126
environment at `runs/step3_gpu`, preserving the CPU environment and lockfile:

```bash
UV_PROJECT_ENVIRONMENT=runs/step3_gpu \
UV_CACHE_DIR=/tmp/elmur-klpo-uv-cache \
uv run --no-sync python -m elmur_klpo.experiment \
  --config configs/step3_remembercolor_reader.json \
  --output runs/step3_remembercolor_reader_gpu \
  --device cuda
```

That command and its checkpoint format describe the historical run. Current
preparation saves `reader_last.pt` and `reader_best.pt`, with section-level
optimizer counters and resume state. The current change stopped after
engineering parity/resume checks; no new full-budget preparation or closed-loop
robot evaluation was executed. The former reconstruction auxiliary loss has
been removed; upstream MoE auxiliary loss is retained.
