import random
import numpy as np
import torch
from modelo.pesos import dispositivo_elegido


class Agente:
    def __init__(self, env, args):
        self.env = env
        self.args = args
        self.device = torch.device(dispositivo_elegido(args.device))
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(args.seed)
        self.num_timesteps = 0
        self._n_updates = 0
        self._n_calls = 0
        self._episode_num = 0
        self.replay_buffer = None
        self.origen_pesos = "torch_propio"
        self.gamma = args.gamma ** (args.option_steps if args.task == "global" else 1)

    def predict(self, observaciones, deterministic=True):
        return self.policy.predict(observaciones, deterministic)

    @torch.no_grad()
    def actuar(self, observacion):
        tensor = torch.as_tensor(
            observacion, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        distribution, value = self.policy(tensor)
        action = distribution.sample()
        return (
            int(action.item()),
            float(value.item()),
            float(distribution.log_prob(action).item()),
        )

    @torch.no_grad()
    def valor(self, observacion):
        tensor = torch.as_tensor(
            observacion, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        return float(self.policy(tensor)[1].item())
