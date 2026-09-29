from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class AgentRNN(nn.Module):
    def __init__(self, observation_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.encoder = nn.Linear(observation_dim, hidden_dim)
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.head = nn.Linear(hidden_dim, action_dim)

    def initial_hidden(self, batch_size: int, device: torch.device | None = None) -> torch.Tensor:
        return torch.zeros(batch_size, self.hidden_dim, device=device)

    def forward(self, observation: torch.Tensor, hidden: torch.Tensor):
        encoded = F.relu(self.encoder(observation))
        next_hidden = self.gru(encoded, hidden)
        return self.head(next_hidden), next_hidden


class QMixer(nn.Module):
    """Monotonic QMIX mixer: hypernetwork weights are forced non-negative."""

    def __init__(self, num_agents: int, state_dim: int, embed_dim: int = 32):
        super().__init__()
        self.num_agents = num_agents
        self.state_dim = state_dim
        self.embed_dim = embed_dim
        self.hyper_w1 = nn.Sequential(nn.Linear(state_dim, 64), nn.ReLU(), nn.Linear(64, num_agents * embed_dim))
        self.hyper_b1 = nn.Linear(state_dim, embed_dim)
        self.hyper_w2 = nn.Sequential(nn.Linear(state_dim, 64), nn.ReLU(), nn.Linear(64, embed_dim))
        self.value = nn.Sequential(nn.Linear(state_dim, embed_dim), nn.ReLU(), nn.Linear(embed_dim, 1))

    def forward(self, agent_qs: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        leading = agent_qs.shape[:-1]
        flat_qs = agent_qs.reshape(-1, 1, self.num_agents)
        flat_states = states.reshape(-1, self.state_dim)
        w1 = F.softplus(self.hyper_w1(flat_states)).view(-1, self.num_agents, self.embed_dim)
        b1 = self.hyper_b1(flat_states).view(-1, 1, self.embed_dim)
        hidden = F.elu(torch.bmm(flat_qs, w1) + b1)
        w2 = F.softplus(self.hyper_w2(flat_states)).view(-1, self.embed_dim, 1)
        value = self.value(flat_states).view(-1, 1, 1)
        total = torch.bmm(hidden, w2) + value
        return total.view(*leading, 1)

