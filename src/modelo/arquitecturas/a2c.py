"""A2C propio: gradiente de política con ventaja TD/GAE y crítico aprendido."""

import torch
from torch.nn import functional as F
from modelo.base import Agente
from .redes import ActorCritico


class A2C(Agente):
    def __init__(self, env, args):
        super().__init__(env, args)
        self.policy = ActorCritico(
            env.observation_space.shape[0], env.action_space.n, args.network
        ).to(self.device)
        self.policy.optimizer = torch.optim.RMSprop(
            self.policy.parameters(),
            lr=args.lr,
            alpha=args.rms_alpha,
            eps=args.optimizer_eps or 1e-5,
        )

    def actualizar(self, batch):
        self.policy.train()
        distribution, values = self.policy(batch["observaciones"])
        advantages = batch["ventajas"]
        if self.args.normalize_advantages and len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (
                advantages.std(unbiased=False) + 1e-8
            )
        actor_loss = -(distribution.log_prob(batch["acciones"]) * advantages).mean()
        value_loss = F.mse_loss(values, batch["retornos"])
        entropy = distribution.entropy().mean()
        loss = (
            actor_loss + self.args.value_coef * value_loss - self.args.entropy * entropy
        )
        self.policy.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(
            self.policy.parameters(), self.args.max_grad_norm or 0.5
        )
        self.policy.optimizer.step()
        self._n_updates += 1
        return dict(
            perdida_actor=actor_loss.item(),
            perdida_valor=value_loss.item(),
            perdida_total=loss.item(),
            entropia=entropy.item(),
            norma_gradiente=float(norm),
        )
