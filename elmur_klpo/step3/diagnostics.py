"""Episode-level uncertainty, memory behavior, and seed-level comparisons."""

import math

import numpy as np

from elmur_klpo.memory import MERGE_COEFFICIENTS
from elmur_klpo.step3.rollout import NativeRollout


def mean_interval(values: list[float], seed: int = 1729) -> dict:
    """Episode-bootstrap 95% mean interval, with fixed diagnostic-only RNG."""
    values = np.asarray(values, dtype=float)
    if not len(values):
        raise ValueError("Cannot summarize empty measurements")
    draws = (
        np.random.default_rng(seed)
        .choice(values, (2000, len(values)), replace=True)
        .mean(1)
    )
    return {
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        "n": len(values),
        "ci95": np.quantile(draws, [0.025, 0.975]).tolist(),
        "interval_unit": "episode",
        "interval_method": "percentile bootstrap, 2000 resamples",
    }


def behavior(traces: list[dict], slots: int) -> dict:
    """Joint frequencies, entropy, memory changes, and observable-event strata."""
    n = max(1, len(traces))
    counts = np.zeros((slots, 5))
    selected = np.zeros(slots)
    noops = 0
    for item in traces:
        noops += item["noop"]
        if item["slot"] is not None:
            selected[item["slot"]] += 1
            if item["alpha"] in MERGE_COEFFICIENTS:
                counts[item["slot"], MERGE_COEFFICIENTS.index(item["alpha"])] += 1
    marginal = counts.sum(0) / counts.sum() if counts.sum() else counts.sum(0)
    entropy = -sum(p * math.log(p) for p in marginal if p > 0)
    changes = [item["memory_change_l2"] for item in traces]
    strata = {}
    for key in ("time", "event", "difficulty", "progress_bin", "age_bin"):
        groups = {}
        for item in traces:
            value = item.get(key)
            if key == "progress_bin":
                value = min(3, int(item["progress"] * 4))
            if key == "age_bin":
                value = None if item["age"] is None else min(3, item["age"] // 5)
            groups.setdefault(str(value), []).append(item)
        strata[key] = {
            label: {
                "n": len(rows),
                "mean_retention": float(np.mean(alphas))
                if (alphas := [r["alpha"] for r in rows if r["alpha"] is not None])
                else None,
                "noop_frequency": sum(r["noop"] for r in rows) / len(rows),
            }
            for label, rows in groups.items()
        }
    episode_groups = {}
    for trace in traces:
        episode_groups.setdefault(trace["episode"], []).append(trace)
    episode_distributions = []
    for rows in episode_groups.values():
        alphas = [
            row["alpha"]
            for row in rows
            if row["alpha"] in MERGE_COEFFICIENTS and row["slot"] is not None
        ]
        if alphas:
            episode_distributions.append(
                [alphas.count(alpha) / len(alphas) for alpha in MERGE_COEFFICIENTS]
            )
    difference = None
    if episode_distributions:
        difference = [
            mean_interval([row[index] - initial for row in episode_distributions])
            for index, initial in enumerate((0.1, 0.1, 0.6, 0.1, 0.1))
        ]
    return {
        "decisions": len(traces),
        "joint_frequencies": (counts / n).tolist(),
        "coefficient_distribution_given_write": marginal.tolist(),
        "coefficient_distribution_by_slot": (
            counts / np.maximum(counts.sum(1, keepdims=True), 1)
        ).tolist(),
        "slot_frequencies": (selected / n).tolist(),
        "coefficient_entropy": entropy,
        "noop_frequency": noops / n,
        "noop_count": noops,
        "overwrite_count": sum(t["overwrite"] for t in traces),
        "mean_slot_age": float(np.mean(ages))
        if (ages := [t["age"] for t in traces if t["age"] is not None])
        else None,
        "mean_memory_change_l2": float(np.mean(changes)) if changes else 0.0,
        "unchanged_memory_fraction": sum(c == 0 for c in changes) / n,
        "strata": strata,
        "coefficient_difference_from_initial": difference,
        "coefficient_interval_scope": "Episode-level bootstrap of conditional write frequencies; exploratory, without multiplicity correction; a distribution change alone is not task improvement",
    }


def summarize(rollout: NativeRollout, slots: int) -> dict:
    result = {
        "return": mean_interval(rollout.episode_returns),
        "episode_returns": rollout.episode_returns,
        "success": None
        if rollout.successes is None
        else mean_interval(rollout.successes),
        "accuracy": None
        if rollout.accuracies is None
        else mean_interval(rollout.accuracies),
        "behavior": behavior(rollout.traces, slots),
        "compute": rollout.compute,
    }
    return result


def paired_difference(a: list[float], b: list[float]) -> dict:
    if len(a) != len(b):
        raise ValueError("Paired measurements require equal sample counts")
    return mean_interval((np.asarray(a) - np.asarray(b)).tolist())


def seed_comparison(differences: list[float], margin: float) -> dict:
    """Seed-level t interval, conditional on reader; assumes normal seed effects."""
    n = len(differences)
    mean = float(np.mean(differences))
    std = float(np.std(differences, ddof=1)) if n > 1 else None
    critical_values = (
        12.706204736,
        4.302652729911275,
        3.182446305,
        2.776445105,
        2.570581836,
        2.446911851,
        2.364624252,
        2.306004135,
        2.262157163,
        2.228138852,
    )
    critical = critical_values[min(max(n - 2, 0), len(critical_values) - 1)]
    half = critical * std / math.sqrt(n) if std is not None else None
    ci = [mean - half, mean + half] if half is not None else None
    return {
        "mean_difference": mean,
        "std_across_seeds": std,
        "seeds": n,
        "ci95": ci,
        "interval_unit": "manager_seed",
        "interval_method": f"Student t, df={n - 1}"
        if n <= 11
        else "Conservative Student t, df=10",
        "noninferiority_margin": margin,
        "noninferiority_supported": bool(ci and ci[0] > -margin),
        "improvement_supported": bool(ci and ci[0] > 0),
    }
