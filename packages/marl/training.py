from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np
import torch
import yaml
from torch import nn

from packages.marl.networks import AgentRNN, QMixer
from packages.marl.replay import EpisodeBatch, EpisodeReplayBuffer
from packages.shared.models import Action
from packages.simulation.environment import DroneDeliveryEnv


ROOT = Path(__file__).resolve().parents[2]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_actions(network, observations, hidden, epsilon, rng):
    agent_ids = list(observations)
    obs = torch.tensor(np.stack([observations[a]["observation"] for a in agent_ids]), dtype=torch.float32)
    with torch.no_grad():
        q_values, next_hidden = network(obs, hidden)
    actions = {}
    for index, agent in enumerate(agent_ids):
        mask = observations[agent]["action_mask"].astype(bool)
        valid = np.flatnonzero(mask)
        if rng.random() < epsilon:
            action = int(rng.choice(valid))
        else:
            masked = q_values[index].numpy().copy()
            masked[~mask] = -np.inf
            action = int(masked.argmax())
        actions[agent] = action
    return actions, next_hidden, q_values


def collect_episode(env, network, epsilon: float, seed: int) -> tuple[EpisodeBatch, dict]:
    observations, _ = env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    hidden = network.initial_hidden(env.num_drones)
    rows = []
    total_reward = 0.0
    while env.agents:
        state = env.state()
        actions, hidden, q_values = select_actions(network, observations, hidden, epsilon, rng)
        next_observations, rewards, terminated, truncated, infos = env.step(actions)
        for agent_index, agent in enumerate(env.possible_agents):
            infos[agent]["decision_event"]["policy_q_values"] = {
                Action(i).name: float(q_values[agent_index, i]) for i in range(len(Action))
            }
            infos[agent]["decision_event"]["model_version"] = "training-candidate"
        next_state = env.state()
        rows.append(
            (
                np.stack([observations[a]["observation"] for a in env.possible_agents]),
                state,
                np.array([actions[a] for a in env.possible_agents]),
                np.stack([observations[a]["action_mask"] for a in env.possible_agents]),
                float(sum(rewards.values())),
                np.stack([next_observations[a]["observation"] for a in env.possible_agents]),
                next_state,
                np.stack([next_observations[a]["action_mask"] for a in env.possible_agents]),
                bool(all(terminated.values()) or all(truncated.values())),
            )
        )
        total_reward += sum(rewards.values())
        observations = next_observations
    batch = EpisodeBatch(*[np.asarray([row[i] for row in rows]) for i in range(9)])
    delivered = sum(p["delivered"] for p in env.packages.values())
    delivery_steps = [p["delivered_step"] for p in env.packages.values() if p["delivered"]]
    near_misses = sum(1 for event in env.safety_events if "MINIMUM_SEPARATION" in event["violated_rules"])
    return batch, {
        "reward": total_reward,
        "delivery_success_rate": delivered / len(env.packages),
        "on_time_delivery_rate": sum(p["delivered"] and p["delivered_step"] <= p["deadline"] for p in env.packages.values()) / len(env.packages),
        "collision_rate": 0.0,
        "near_miss_rate": near_misses / max(1, env.step_count * env.num_drones),
        "safety_overrides": len(env.safety_events),
        "energy_per_delivery": sum(d["energy_consumed"] for d in env.drones.values()) / max(1, delivered),
        "average_delivery_time": float(np.mean(delivery_steps)) if delivery_steps else float(env.max_steps),
        "p95_delivery_time": float(np.percentile(delivery_steps, 95)) if delivery_steps else float(env.max_steps),
        "failed_missions": len(env.packages) - delivered,
        "charging_wait_time": 0.0,
        "communication_loss_events": sum(1 for event in env.communication_events if event["dropped"]),
    }


def _transition_samples(episodes: list[EpisodeBatch]):
    fields = []
    for index in range(9):
        fields.append(np.concatenate([getattr(ep, list(EpisodeBatch.__annotations__)[index]) for ep in episodes], axis=0))
    return fields


def train(preset: str = "quick", algorithm: str = "qmix", seed: int = 42, progress=None) -> dict:
    seed_everything(seed)
    config = yaml.safe_load((ROOT / "configs" / "training.yaml").read_text(encoding="utf-8"))
    cfg = {**config["common"], **config[preset]}
    env = DroneDeliveryEnv(num_drones=3, max_steps=cfg["max_steps"], seed=seed)
    obs_dim, action_dim = env.OBS_SIZE, len(Action)
    state_dim = obs_dim * env.num_drones
    network = AgentRNN(obs_dim, action_dim, cfg["hidden_dim"])
    target_network = AgentRNN(obs_dim, action_dim, cfg["hidden_dim"])
    target_network.load_state_dict(network.state_dict())
    mixer = QMixer(env.num_drones, state_dim, cfg["hidden_dim"]) if algorithm == "qmix" else None
    target_mixer = QMixer(env.num_drones, state_dim, cfg["hidden_dim"]) if algorithm == "qmix" else None
    if target_mixer:
        target_mixer.load_state_dict(mixer.state_dict())
    parameters = list(network.parameters()) + (list(mixer.parameters()) if mixer else [])
    optimizer = torch.optim.Adam(parameters, lr=cfg["learning_rate"])
    buffer = EpisodeReplayBuffer(cfg["buffer_size"], seed)
    history = []

    for episode in range(cfg["episodes"]):
        fraction = episode / max(1, cfg["episodes"] - 1)
        epsilon = cfg["epsilon_start"] + fraction * (cfg["epsilon_end"] - cfg["epsilon_start"])
        episode_batch, metrics = collect_episode(env, network, epsilon, seed + episode)
        buffer.add(episode_batch)
        loss_value = None
        if len(buffer) >= cfg["batch_size"]:
            samples = _transition_samples(buffer.sample(cfg["batch_size"]))
            obs, states, actions, masks, rewards, next_obs, next_states, next_masks, terminated = samples
            n = len(obs)
            hidden = network.initial_hidden(n * env.num_drones)
            q, _ = network(torch.tensor(obs.reshape(-1, obs_dim), dtype=torch.float32), hidden)
            chosen = q.gather(1, torch.tensor(actions.reshape(-1, 1), dtype=torch.long)).view(n, env.num_drones)
            with torch.no_grad():
                target_hidden = target_network.initial_hidden(n * env.num_drones)
                target_q, _ = target_network(torch.tensor(next_obs.reshape(-1, obs_dim), dtype=torch.float32), target_hidden)
                mask_tensor = torch.tensor(next_masks.reshape(-1, action_dim), dtype=torch.bool)
                target_q[~mask_tensor] = -1e9
                max_next = target_q.max(dim=1).values.view(n, env.num_drones)
            state_tensor = torch.tensor(states, dtype=torch.float32)
            next_state_tensor = torch.tensor(next_states, dtype=torch.float32)
            if mixer:
                prediction = mixer(chosen, state_tensor).squeeze(-1)
                with torch.no_grad():
                    target_total = target_mixer(max_next, next_state_tensor).squeeze(-1)
            else:
                prediction = chosen.mean(dim=1)
                target_total = max_next.mean(dim=1)
            reward_tensor = torch.tensor(rewards, dtype=torch.float32)
            done_tensor = torch.tensor(terminated, dtype=torch.float32)
            target = reward_tensor + cfg["gamma"] * (1 - done_tensor) * target_total
            loss = nn.functional.mse_loss(prediction, target)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(parameters, cfg["gradient_clip"])
            optimizer.step()
            loss_value = float(loss.detach())
        if (episode + 1) % cfg["target_update_interval"] == 0:
            target_network.load_state_dict(network.state_dict())
            if mixer:
                target_mixer.load_state_dict(mixer.state_dict())
        record = {"episode": episode + 1, "epsilon": epsilon, "loss": loss_value, **metrics}
        history.append(record)
        if progress:
            progress(record)

    run_id = str(uuid4())
    checkpoint_dir = ROOT / "artifacts" / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = checkpoint_dir / f"{algorithm}-{preset}-{run_id[:8]}.pt"
    torch.save(
        {
            "run_id": run_id,
            "algorithm": algorithm,
            "preset": preset,
            "seed": seed,
            "config": cfg,
            "agent": network.state_dict(),
            "mixer": mixer.state_dict() if mixer else None,
            "history": history,
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
        checkpoint,
    )
    result = {"run_id": run_id, "algorithm": algorithm, "seed": seed, "checkpoint": str(checkpoint), "history": history}
    (checkpoint.with_suffix(".json")).write_text(json.dumps(result, indent=2), encoding="utf-8")
    try:
        import mlflow
        with mlflow.start_run(run_name=f"{algorithm}-{preset}-{run_id[:8]}"):
            mlflow.log_params({"algorithm": algorithm, "preset": preset, "seed": seed, **cfg})
            for record in history:
                mlflow.log_metrics({k: v for k, v in record.items() if isinstance(v, (int, float)) and v is not None}, step=record["episode"])
            mlflow.log_artifact(str(checkpoint))
    except Exception:
        pass  # MLflow is optional at runtime; the checkpoint and JSON history remain authoritative.
    return result


def load_checkpoint(path: str | Path):
    data = torch.load(path, map_location="cpu", weights_only=False)
    cfg = data["config"]
    network = AgentRNN(DroneDeliveryEnv.OBS_SIZE, len(Action), cfg["hidden_dim"])
    network.load_state_dict(data["agent"])
    network.eval()
    return network, data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", choices=["quick", "standard", "full"], default="quick")
    parser.add_argument("--algorithm", choices=["qmix", "idqn"], default="qmix")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(train(args.preset, args.algorithm, args.seed), indent=2))


if __name__ == "__main__":
    main()

