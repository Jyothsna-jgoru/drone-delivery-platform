from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from packages.marl.training import collect_episode, load_checkpoint
from packages.simulation.environment import DroneDeliveryEnv


ROOT = Path(__file__).resolve().parents[2]


def evaluate(checkpoint: str | Path, seeds: list[int]) -> dict:
    network, metadata = load_checkpoint(checkpoint)
    metrics = []
    for seed in seeds:
        env = DroneDeliveryEnv(num_drones=3, max_steps=metadata["config"]["max_steps"], seed=seed)
        _, result = collect_episode(env, network, epsilon=0.0, seed=seed)
        metrics.append({"seed": seed, **result})
    keys = [k for k in metrics[0] if k != "seed"]
    summary = {
        key: {"mean": float(np.mean([m[key] for m in metrics])), "std": float(np.std([m[key] for m in metrics]))}
        for key in keys
    }
    result = {
        "run_id": metadata["run_id"],
        "algorithm": metadata["algorithm"],
        "checkpoint": str(checkpoint),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "episodes": len(seeds),
        "seeds": seeds,
        "results": metrics,
        "summary": summary,
    }
    reports = ROOT / "reports"
    reports.mkdir(exist_ok=True)
    output = reports / f"evaluation-{metadata['algorithm']}-{metadata['run_id'][:8]}.json"
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    result["report"] = str(output)
    return result


def compare(qmix_report: dict, idqn_report: dict) -> dict:
    comparison = {"qmix": qmix_report, "idqn": idqn_report, "winner_by_metric": {}}
    lower_is_better = {
        "collision_rate", "near_miss_rate", "safety_overrides", "energy_per_delivery",
        "average_delivery_time", "p95_delivery_time", "failed_missions", "charging_wait_time",
        "communication_loss_events",
    }
    for metric in qmix_report["summary"]:
        q = qmix_report["summary"][metric]["mean"]
        d = idqn_report["summary"][metric]["mean"]
        comparison["winner_by_metric"][metric] = (
            "qmix" if (q < d if metric in lower_is_better else q > d) else "idqn" if q != d else "tie"
        )
    comparison["generated_at"] = datetime.now(timezone.utc).isoformat()
    return comparison


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--seeds", default="101,102,103")
    args = parser.parse_args()
    print(json.dumps(evaluate(args.checkpoint, [int(x) for x in args.seeds.split(",")]), indent=2))


if __name__ == "__main__":
    main()
