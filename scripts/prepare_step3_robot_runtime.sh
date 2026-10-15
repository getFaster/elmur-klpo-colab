#!/usr/bin/env bash
# Isolate upstream Python/NumPy/Torch requirements from the pinned main runtime.
set -euo pipefail
native_env="${1:-runs/step3_native}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/elmur-klpo-uv-cache}"
if [ ! -x "$native_env/bin/python" ]; then
    uv venv "$native_env" --python 3.11
fi
uv pip install --python "$native_env/bin/python" 'torch==2.2.1+cpu' \
    --index-url https://download.pytorch.org/whl/cpu
uv pip install --python "$native_env/bin/python" \
    'numpy==1.23.5' 'gymnasium==0.29.1' 'mani_skill==3.0.0b15' \
    'setuptools>=69,<81'
# The complete MIKASA extras include training/plotting tooling; the worker only
# needs ManiSkill and the task definitions. Keep those dependencies separate.
uv pip install --python "$native_env/bin/python" --no-deps \
    'git+https://github.com/CognitiveAISystems/MIKASA-Robo.git@509b875f3d207c287497c0a897661062de928bb0'
