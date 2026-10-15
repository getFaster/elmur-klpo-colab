"""Pinned RATE configuration, source integrity, and native write-event contract."""

import ast
import hashlib
import json
import shlex
import sys
import tempfile
import types
import zipfile
from pathlib import Path

COMMIT = "8abce2dce9ceacd411d46d4a99c287f7975892f5"
REFERENCE = Path(__file__).resolve().parents[1] / "vendor" / "rate_reference"
_REFERENCE_TEMP = None


def read_reference(path: str) -> str:
    """Read a byte-exact source file from the pinned source archive."""
    with zipfile.ZipFile(REFERENCE / "sources.zip") as archive:
        return archive.read(path).decode()


def reference_package() -> str:
    """Extract independent raw packages to temporary storage for parity imports."""
    global _REFERENCE_TEMP
    name = "pinned_rate_reference"
    if _REFERENCE_TEMP is None:
        _REFERENCE_TEMP = tempfile.TemporaryDirectory(prefix="elmur-rate-parity-")
        with zipfile.ZipFile(REFERENCE / "sources.zip") as archive:
            archive.extractall(_REFERENCE_TEMP.name)
        package = types.ModuleType(name)
        package.__path__ = [_REFERENCE_TEMP.name]
        sys.modules[name] = package
    return name


BUDGETS = {"autoencode": 57_600, "remember_color": 9_600}


def _value(value: str) -> object:
    if value == "True":
        return True
    if value == "False":
        return False
    if value == "None":
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def reference_settings(task: str) -> dict:
    """Resolve the actual command plus model defaults, including explicit flags."""
    if task == "autoencode":
        path = "offline_rl_baselines/ELMUR/README.md"
        line = next(
            s
            for s in read_reference(path).splitlines()
            if s.startswith("python src/train.py") and "AutoencodeEasy" in s
        )
    elif task == "remember_color":
        path = "run_experiments/MIKASA_Robo/ELMUR_1/rc3.sh"
        line = next(
            s for s in read_reference(path).splitlines() if s.startswith("python3")
        )
    else:
        raise ValueError("Unknown task")
    args = dict(
        arg[2:].split("=", 1)
        for arg in shlex.split(line)
        if arg.startswith("--") and "=" in arg
    )
    tree = ast.parse(read_reference("offline_rl_baselines/ELMUR/model.py"))
    model_class = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ELMURModel"
    )
    constructor = next(
        n
        for n in model_class.body
        if isinstance(n, ast.FunctionDef) and n.name == "__init__"
    )
    names = [a.arg for a in constructor.args.args][-len(constructor.args.defaults) :]
    model = {
        name: ast.literal_eval(default)
        for name, default in zip(names, constructor.args.defaults, strict=True)
    }
    model.update(
        {
            key[6:].replace("-", "_"): _value(value)
            for key, value in args.items()
            if key.startswith("model.")
        }
    )
    model["state_dim"] = 6  # Upstream trainer resolves POPGym's -1 from observations.
    model["dtype"] = "float32"
    training = {
        key[9:].replace("-", "_"): _value(value)
        for key, value in args.items()
        if key.startswith("training.")
    }
    training["max_segments"] = training["sections"]
    return {
        "commit": COMMIT,
        "command_source": path,
        "command": line,
        "model": model,
        "training": training,
    }


def configuration_diff(config: dict) -> dict:
    """Emit every reference/local model and optimizer option, with differences."""
    ref = reference_settings(config["task"]["name"])
    rows = []
    for group in ("model", "training"):
        local = config["reader"] if group == "model" else config["preparation"]
        for key in sorted(set(ref[group]) | set(local)):
            rows.append(
                {
                    "field": f"{group}.{key}",
                    "reference": ref[group].get(key),
                    "local": local.get(key),
                    "matches": ref[group].get(key) == local.get(key),
                }
            )
    return {
        **ref,
        "comparison": rows,
        "explicit_cli_repairs": {
            "use_tok2mem": {"literal_cli": None, "local": True},
            "use_mem2tok": {"literal_cli": None, "local": True},
            "use_relative_bias": {"literal_cli": None, "local": True},
            "use_shared_memory": {"literal_cli": None, "local": False},
        },
        "cli_default_caveat": "src/train.py passes optional None flags through to ELMURModel; the native commands omit use_tok2mem/use_mem2tok/use_relative_bias. This repair explicitly uses model defaults True for a functioning memory system, not the literal CLI None overrides. use_shared_memory is explicitly False.",
    }


def verify_sources() -> dict:
    """Check byte-exact reference hashes and commit without network access."""
    manifest = json.loads((REFERENCE / "manifest.json").read_text())
    if manifest["commit"] != COMMIT:
        raise ValueError("Reference commit mismatch")
    with zipfile.ZipFile(REFERENCE / "sources.zip") as archive:
        for name, digest in manifest["files"].items():
            if hashlib.sha256(archive.read(name)).hexdigest() != digest:
                raise ValueError(f"Pinned source integrity failure: {name}")
    return {"commit": COMMIT, "verified_files": len(manifest["files"])}


def write_event_contract(model: dict, context_length: int) -> dict:
    """Machine-readable contract for the unmodified native fixed-LRU forward."""
    return {
        "commit": COMMIT,
        "events_per_forward": model["n_layer"],
        "event_order": list(range(model["n_layer"])),
        "event_index": "section_index * n_layer + layer_index",
        "physical_timesteps": "section_index*T through (section_index+1)*T-1; all layers share this physical interval",
        "next_eligible_predictions": "next section onward; no current-section imitation reward may be credited to these outgoing writes",
        "training_fill": "reset per batch; 3 sections produce only 3 writes per layer (out of 8 or 256 slots)",
        "frequency": "one write decision per layer per section, not per token",
        "candidate_source": "ELMURLayer.memory_ffn(u)[0], then training memory dropout",
        "candidate_shape": ["batch", model["memory_size"], model["d_model"]],
        "memory_shape": [
            model["n_layer"],
            "batch",
            model["memory_size"],
            model["d_model"],
        ],
        "positions_shape": [model["n_layer"], "batch", model["memory_size"]],
        "relative_tok_pos": "arange(offset+T-1, offset-1, -1); token observations remain chronological",
        "timestamp": "tok_pos[0] = section_offset + section_length - 1",
        "first_section_timestamp": context_length - 1,
        "read_flag": "use_tok2mem: cross_attention1, tokens query incoming memory",
        "write_flag": "use_mem2tok: cross_attention2, incoming memory queries all section tokens",
        "prediction_alignment": "layer hidden states and all current-section predictions do not depend on that layer's outgoing write; influence begins in next section",
        "causal_features": [
            "current layer incoming memory and timestamps",
            "current section observation embeddings",
            "post-token-FFN hidden states",
            "slot-conditioned candidate for every slot",
            "section end time and layer index",
        ],
        "forbidden_features": [
            "expert actions",
            "oracle deck state",
            "future observations",
            "return-to-go",
        ],
        "native_target": "first_empty = (empty.float().cumsum(-1)==1); full bank -> argmin(pos), first tie",
        "native_empty_blend": "lru_one_hot applies lambda even on the first empty slot; other first_empty mask entries use weight 1. Preserve code, not contradictory comment.",
        "noncontiguous_empty_caveat": "cumsum==1 can select occupied slots after first empty and before second empty; normal prefix-fill state selects one slot",
        "gradient_boundary": "incoming layer memories detached before each section; no future section task gradient into previous writes",
        "manager_integration": "deferred to execution item 6; not compatible with legacy single-token manager rollouts",
    }
