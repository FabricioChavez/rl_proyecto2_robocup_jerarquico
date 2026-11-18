"""Experience replay circular implementado aquí, no en una librería de RL."""

import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, capacity, observation_size):
        self.buffer_size = capacity
        self.pos = 0
        self.full = False
        self.observations = np.empty((capacity, observation_size), dtype=np.float32)
        self.next_observations = np.empty_like(self.observations)
        self.actions = np.empty(capacity, dtype=np.int64)
        self.rewards = np.empty(capacity, dtype=np.float32)
        self.dones = np.empty(capacity, dtype=np.float32)
        self.timeouts = np.empty(capacity, dtype=np.float32)

    def size(self):
        return self.buffer_size if self.full else self.pos

    def agregar(self, obs, action, reward, next_obs, terminated, truncated):
        i = self.pos
        self.observations[i] = obs
        self.next_observations[i] = next_obs
        self.actions[i] = action
        self.rewards[i] = reward
        self.dones[i] = terminated
        self.timeouts[i] = truncated
        self.pos = (i + 1) % self.buffer_size
        if self.pos == 0:
            self.full = True

    def muestrear(self, batch, device):
        indices = np.random.randint(0, self.size(), size=batch)
        return {
            name: torch.as_tensor(getattr(self, name)[indices], device=device)
            for name in (
                "observations",
                "next_observations",
                "actions",
                "rewards",
                "dones",
                "timeouts",
            )
        }
