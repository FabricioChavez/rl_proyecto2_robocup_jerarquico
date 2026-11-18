"""GAE con máscaras separadas: bootstrap por truncación, corte entre episodios."""

import torch


@torch.no_grad()
def gae(rewards, values, next_values, terminated, boundaries, gamma, lam):
    advantages = torch.zeros_like(rewards)
    accumulator = torch.zeros((), dtype=values.dtype, device=values.device)
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * (1 - terminated[t]) * next_values[t] - values[t]
        accumulator = delta + gamma * lam * (1 - boundaries[t]) * accumulator
        advantages[t] = accumulator
    return advantages, advantages + values
