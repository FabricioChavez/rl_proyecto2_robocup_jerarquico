"""DQN propio: epsilon-greedy, replay, objetivo TD y sincronización de red objetivo."""

import numpy as np
import torch
from torch.nn import functional as F
from modelo.base import Agente
from modelo.replay import ReplayBuffer
from .redes import PoliticaQ


class DQN(Agente):
    def __init__(self, env, args):
        super().__init__(env, args)
        self.policy = PoliticaQ(
            env.observation_space.shape[0], env.action_space.n, args.network
        ).to(self.device)
        self.policy.optimizer = torch.optim.Adam(
            self.policy.q_net.parameters(), lr=args.lr, eps=args.optimizer_eps or 1e-8
        )
        self.replay_buffer = ReplayBuffer(args.buffer, env.observation_space.shape[0])
        self.learning_starts = args.warmup
        self.total_programado = args.steps
        self.epsilon = args.epsilon_initial

    def actuar(self, observacion):
        decay = self.args.epsilon_steps or max(
            1, int(self.total_programado * self.args.exploration_fraction)
        )
        fraction = min(1.0, self.num_timesteps / decay)
        self.epsilon = self.args.epsilon_initial + fraction * (
            self.args.epsilon_final - self.args.epsilon_initial
        )
        action = (
            int(np.random.randint(self.env.action_space.n))
            if np.random.random() < self.epsilon
            else int(self.predict(observacion)[0])
        )
        return action, 0.0, 0.0

    def recordar(self, obs, action, reward, next_obs, terminated, truncated):
        self.replay_buffer.agregar(obs, action, reward, next_obs, terminated, truncated)
        if self._n_calls % self.args.target_update == 0:
            self.policy.q_net_target.load_state_dict(self.policy.q_net.state_dict())

    def actualizar(self):
        batch = self.replay_buffer.muestrear(self.args.batch, self.device)
        with torch.no_grad():
            siguiente = batch["next_observations"]
            if getattr(self.args, "double", False):
                # Double DQN (van Hasselt et al. 2016): reduce la sobreestimación.
                elegida = self.policy.q_net(siguiente).argmax(dim=1, keepdim=True)
                future = self.policy.q_net_target(siguiente).gather(1, elegida).squeeze(1)
            else:
                future = self.policy.q_net_target(siguiente).max(dim=1).values
            target = batch["rewards"] + self.gamma * (1 - batch["dones"]) * future
        predicted = (
            self.policy.q_net(batch["observations"])
            .gather(1, batch["actions"].long().unsqueeze(1))
            .squeeze(1)
        )
        loss = F.smooth_l1_loss(predicted, target)
        self.policy.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(
            self.policy.q_net.parameters(), self.args.max_grad_norm or 10.0
        )
        self.policy.optimizer.step()
        self._n_updates += 1
        return dict(
            perdida_q=loss.item(),
            q_media=predicted.mean().item(),
            objetivo_td=target.mean().item(),
            epsilon=self.epsilon,
            norma_gradiente=float(norm),
        )
