"""Independent reader systems trained with matched curriculum and budgets."""

from pathlib import Path

from elmur_klpo.step3.artifacts import write_json
from elmur_klpo.step3.curriculum import environment_seeds
from elmur_klpo.step3.evaluation import evaluate
from elmur_klpo.step3.preparation import prepare_reader


def fixed_systems(config: dict, output: Path, resume: bool, device: str) -> dict:
    """Train native LRU, uniform paper-retention, and uniform fixed-half systems."""
    systems = {
        "native_lru": ("lru", None),
        "paper_coefficient": ("fixed", config["paper_retention"]),
        "fixed_0.5": ("fixed", 0.5),
    }
    result = {}
    validation = environment_seeds(
        "validation", 0, config["evaluation"]["episodes"], config["data_seed"]
    )
    test = environment_seeds(
        "test", 0, config["evaluation"]["episodes"], config["data_seed"]
    )
    for name, (method, alpha) in systems.items():
        reader, initial, preparation = prepare_reader(
            config,
            output / name,
            resume and (output / name / "reader.pt").exists(),
            device,
            method=method,
            fixed_alpha=alpha,
        )
        valid = evaluate(
            reader,
            config["task"],
            validation,
            initial,
            method=method,
            fixed_alpha=alpha,
        )
        measured = {
            str(seed): evaluate(
                reader,
                config["task"],
                test,
                initial,
                method=method,
                fixed_alpha=alpha,
                policy_seed=seed + 10000,
            )
            for seed in config["manager"]["seeds"]
        }
        result[name] = {
            "preparation": preparation,
            "validation": valid,
            "test_by_manager_seed": measured,
        }
        write_json(output / "fixed_systems.json", result)
        write_json(output / name / "reader_report.json", preparation)
    return result
