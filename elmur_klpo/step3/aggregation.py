"""Export compact measured artifacts while retaining full histories under runs/."""

import argparse
import json
from pathlib import Path

from elmur_klpo.step3.artifacts import write_json


def compact_evaluation(measured: dict) -> dict:
    """Keep episode measurements and behavior; omit per-timestep strata."""
    behavior = {
        key: value for key, value in measured["behavior"].items() if key != "strata"
    }
    return {**measured, "behavior": behavior}


def aggregate(inputs: list[Path]) -> dict:
    """Gather completed stage outputs without inventing missing comparisons."""
    runs = []
    for root in inputs:
        source = json.loads((root / "results.json").read_text())
        entry = {
            "source": str(root),
            "outcome": source.get("outcome", "not_evaluated"),
            "tasks": {},
        }
        for name, task in source["tasks"].items():
            if task["status"] in {
                "upstream_parity_smoke_passed",
                "reader_preparation_complete_gates_pending",
            }:
                compact = dict(task)
                if "reader" in compact:
                    compact["reader"] = {
                        k: v for k, v in compact["reader"].items() if k != "history"
                    }
                if "parity" in compact:
                    compact["parity"] = {
                        **compact["parity"],
                        "sections": [
                            {
                                k: v
                                for k, v in row.items()
                                if k != "writer_parameters_with_aux_grad"
                            }
                            for row in compact["parity"]["sections"]
                        ],
                    }
                entry["tasks"][name] = compact
                continue
            compact = {
                key: task[key]
                for key in (
                    "config",
                    "stage",
                    "device",
                    "torch_version",
                    "status",
                    "outcome",
                )
            }
            if "error" in task:
                compact["error"] = task["error"]
            if "reader" in task:
                reader = task["reader"]
                gate = reader["readiness"]
                compact["reader"] = {
                    key: reader[key]
                    for key in (
                        "reader_seconds",
                        "environment_interactions",
                        "checkpoint_bytes",
                        "initial_probabilities",
                    )
                }
                compact["reader"]["readiness"] = {
                    **gate,
                    "genuine": compact_evaluation(gate["genuine"]),
                    "ablated": compact_evaluation(gate["ablated"]),
                    "management_robustness": {
                        k: compact_evaluation(v)
                        for k, v in gate["management_robustness"].items()
                    },
                    "decodability": {
                        **gate["decodability"],
                        "measurement": compact_evaluation(
                            gate["decodability"]["measurement"]
                        ),
                    },
                }
                compact["reader"]["curve"] = [
                    {
                        key: row[key]
                        for key in (
                            "iteration",
                            "difficulty",
                            "prediction_loss",
                            "tok2mem_gradient_norm",
                            "native_return",
                            "reader_seconds",
                            "environment_steps",
                        )
                    }
                    | (
                        {"validation": compact_evaluation(row["validation"])}
                        if "validation" in row
                        else {}
                    )
                    for row in reader["history"]
                ]
            if "baselines" in task:
                compact["baselines"] = {
                    key: compact_evaluation(value)
                    for key, value in task["baselines"].items()
                    if key != "fixed"
                }
                compact["baselines"]["fixed"] = {
                    key: compact_evaluation(value)
                    for key, value in task["baselines"]["fixed"].items()
                }
            if "managers" in task:
                compact["managers"] = []
                for manager in task["managers"]:
                    value = {
                        key: item for key, item in manager.items() if key != "history"
                    }
                    value["curve"] = [
                        {
                            "iteration": row["iteration"],
                            "gradient_norm": row["gradient_norm"],
                            "training_return": row["training"]["return"],
                            **(
                                {"validation_return": row["validation"]["return"]}
                                if "validation" in row
                                else {}
                            ),
                        }
                        for row in manager.get("history", [])
                    ]
                    compact["managers"].append(value)
            for key in ("hypotheses", "systems", "robustness", "manager_training"):
                if key in task:
                    compact[key] = task[key]
            entry["tasks"][name] = compact
        runs.append(entry)
    return {
        "runs": runs,
        "note": "Only executed measurements are present; omitted manager/system comparisons were not executed",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write_json(args.output, aggregate(args.inputs))


if __name__ == "__main__":
    main()
