"""PPO propio: razón de probabilidades, clipping, crítico y regularización entrópica."""

import numpy as np
import torch
from torch.nn import functional as F
from modelo.base import Agente
from .redes import ActorCritico


class PPO(Agente):
    def __init__(self, env, args):
        super().__init__(env, args)
        self.policy = ActorCritico(
            env.observation_space.shape[0], env.action_space.n, args.network
        ).to(self.device)
        self.policy.optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=args.lr, eps=args.optimizer_eps or 1e-5
        )

    def actualizar(self, batch):
        self.policy.train()
        advantages = batch["ventajas"]
        if self.args.normalize_advantages is not False and len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (
                advantages.std(unbiased=False) + 1e-8
            )
        metrics = []
        stop = False
        for epoch in range(self.args.epochs):
            for indices in torch.randperm(len(advantages), device=self.device).split(
                self.args.batch
            ):
                distribution, values = self.policy(batch["observaciones"][indices])
                log_probability = distribution.log_prob(batch["acciones"][indices])
                log_ratio = log_probability - batch["log_prob"][indices]
                ratio = log_ratio.exp()
                clipped = ratio.clamp(1 - self.args.clip, 1 + self.args.clip)
                actor_loss = -torch.minimum(
                    ratio * advantages[indices], clipped * advantages[indices]
                ).mean()
                value_loss = F.mse_loss(values, batch["retornos"][indices])
                entropy = distribution.entropy().mean()
                loss = (
                    actor_loss
                    + self.args.value_coef * value_loss
                    - self.args.entropy * entropy
                )
                with torch.no_grad():
                    kl = ((ratio - 1) - log_ratio).mean()
                    fraction = ((ratio - 1).abs() > self.args.clip).float().mean()
                if self.args.target_kl and kl.item() > 1.5 * self.args.target_kl:
                    stop = True
                    break
                self.policy.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.args.max_grad_norm or 0.5
                )
                self.policy.optimizer.step()
                self._n_updates += 1
                metrics.append(
                    dict(
                        perdida_actor=actor_loss.item(),
                        perdida_valor=value_loss.item(),
                        perdida_total=loss.item(),
                        entropia=entropy.item(),
                        kl=kl.item(),
                        fraccion_clip=fraction.item(),
                        norma_gradiente=float(norm),
                    )
                )
            if stop:
                break
        return (
            {key: float(np.mean([row[key] for row in metrics])) for key in metrics[0]}
            if metrics
            else {}
        )
