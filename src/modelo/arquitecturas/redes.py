"""Redes PyTorch propias. Los nombres de capas permiten leer las referencias históricas."""

from copy import deepcopy
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical


def mlp(entrada, capas, activacion):
    modules = []
    for salida in capas:
        modules.extend([nn.Linear(entrada, salida), activacion()])
        entrada = salida
    return nn.Sequential(*modules)


class ExtractorActorCritico(nn.Module):
    def __init__(self, observaciones, capas):
        super().__init__()
        self.policy_net = mlp(observaciones, capas, nn.Tanh)
        self.value_net = mlp(observaciones, capas, nn.Tanh)


class ActorCritico(nn.Module):
    def __init__(self, observaciones, acciones, capas):
        super().__init__()
        self.mlp_extractor = ExtractorActorCritico(observaciones, capas)
        self.action_net = nn.Linear(capas[-1], acciones)
        self.value_net = nn.Linear(capas[-1], 1)
        for module in self.mlp_extractor.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, np.sqrt(2))
                nn.init.zeros_(module.bias)
        nn.init.orthogonal_(self.action_net.weight, 0.01)
        nn.init.zeros_(self.action_net.bias)
        nn.init.orthogonal_(self.value_net.weight, 1.0)
        nn.init.zeros_(self.value_net.bias)

    def forward(self, observaciones):
        actor = self.mlp_extractor.policy_net(observaciones)
        critic = self.mlp_extractor.value_net(observaciones)
        return Categorical(logits=self.action_net(actor)), self.value_net(
            critic
        ).squeeze(-1)

    def get_distribution(self, observaciones):
        return self(observaciones)[0]

    def predict_values(self, observaciones):
        return self(observaciones)[1].unsqueeze(-1)

    def set_training_mode(self, mode):
        self.train(mode)

    @torch.no_grad()
    def predict(self, observaciones, deterministic=True):
        tensor = torch.as_tensor(
            observaciones, dtype=torch.float32, device=next(self.parameters()).device
        )
        single = tensor.ndim == 1
        if single:
            tensor = tensor.unsqueeze(0)
        distribution, _ = self(tensor)
        actions = (
            distribution.logits.argmax(-1) if deterministic else distribution.sample()
        )
        result = actions.cpu().numpy()
        return (result[0] if single else result), None


class RedQ(nn.Module):
    def __init__(self, observaciones, acciones, capas):
        super().__init__()
        self.q_net = nn.Sequential(
            *list(mlp(observaciones, capas, nn.ReLU).children()),
            nn.Linear(capas[-1], acciones),
        )

    def forward(self, x):
        return self.q_net(x)


class PoliticaQ(nn.Module):
    def __init__(self, observaciones, acciones, capas):
        super().__init__()
        self.q_net = RedQ(observaciones, acciones, capas)
        self.q_net_target = deepcopy(self.q_net)
        self.q_net_target.requires_grad_(False)
        self.q_net_target.eval()

    def set_training_mode(self, mode):
        self.q_net.train(mode)
        self.q_net_target.eval()

    @torch.no_grad()
    def predict(self, observaciones, deterministic=True):
        tensor = torch.as_tensor(
            observaciones, dtype=torch.float32, device=next(self.parameters()).device
        )
        single = tensor.ndim == 1
        if single:
            tensor = tensor.unsqueeze(0)
        actions = self.q_net(tensor).argmax(-1).cpu().numpy()
        return (actions[0] if single else actions), None
