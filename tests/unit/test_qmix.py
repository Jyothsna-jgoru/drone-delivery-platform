from pathlib import Path

import numpy as np
import torch

from packages.marl.networks import AgentRNN, QMixer
from packages.marl.replay import EpisodeBatch, EpisodeReplayBuffer
from packages.marl.training import load_checkpoint


def test_network_shapes_and_masking():
    net = AgentRNN(32, 13, 16)
    q, hidden = net(torch.zeros(6,32), net.initial_hidden(6))
    assert q.shape == (6,13)
    assert hidden.shape == (6,16)
    mask = torch.ones_like(q, dtype=torch.bool); mask[:,3] = False
    q[~mask] = -torch.inf
    assert not torch.any(q.argmax(1) == 3)


def test_mixer_is_monotonic_in_agent_values():
    torch.manual_seed(2)
    mixer = QMixer(3, 96, 16)
    state = torch.randn(8,96)
    lower = torch.randn(8,3)
    higher = lower + torch.rand(8,3)
    assert torch.all(mixer(higher,state) >= mixer(lower,state) - 1e-6)


def test_replay_buffer_sampling():
    array = np.zeros((2,3,4))
    episode = EpisodeBatch(array,np.zeros((2,12)),np.zeros((2,3)),array,np.zeros(2),array,np.zeros((2,12)),array,np.zeros(2))
    replay = EpisodeReplayBuffer(3, seed=1)
    replay.add(episode); replay.add(episode)
    assert len(replay.sample(1)) == 1


def test_checkpoint_round_trip(tmp_path: Path):
    network = AgentRNN(32,13,8)
    checkpoint = tmp_path / "model.pt"
    torch.save({"run_id":"x","algorithm":"idqn","config":{"hidden_dim":8},"agent":network.state_dict()},checkpoint)
    loaded, metadata = load_checkpoint(checkpoint)
    for a,b in zip(network.parameters(),loaded.parameters()): assert torch.equal(a,b)
    assert metadata["run_id"] == "x"

