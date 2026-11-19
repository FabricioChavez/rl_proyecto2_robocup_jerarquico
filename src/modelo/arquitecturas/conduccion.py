"""Redes de conducción escritas en torch.nn; sin implementaciones externas RL."""
import torch
from torch import nn


def tronco(entradas, capas):
    modules = []
    for ancho in capas:
        modules.extend((nn.Linear(entradas, ancho), nn.ReLU()))
        entradas = ancho
    return nn.Sequential(*modules)


class DuelingQ(nn.Module):
    def __init__(self, estados, capas, acciones=8):
        super().__init__()
        self.features = tronco(estados, capas)
        self.valor = nn.Linear(capas[-1], 1)
        self.ventaja = nn.Linear(capas[-1], acciones)

    def forward(self, estado):
        h = self.features(estado)
        a = self.ventaja(h)
        return self.valor(h)+a-a.mean(dim=-1, keepdim=True)


class Parametros(nn.Module):
    def __init__(self, estados, capas):
        super().__init__()
        self.features = tronco(estados, capas)
        self.output = nn.Linear(capas[-1], 4)
        nn.init.uniform_(self.output.weight, -.003, .003)
        nn.init.zeros_(self.output.bias)

    def forward(self, estado):
        # Salida lineal: límites mediante clamp al actuar e inversión del gradiente
        # al aprender. No usa tanh saturado para los parámetros continuos.
        return self.output(self.features(estado))


class MultiPassQ(nn.Module):
    def __init__(self, estados, capas):
        super().__init__()
        self.network = nn.Sequential(tronco(estados+4, capas), nn.Linear(capas[-1], 3))
        self.register_buffer('mascara', torch.tensor([
            [1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 1.]]))

    def forward(self, estado, parametros):
        n = estado.shape[0]
        observations = estado[:, None, :].expand(-1, 3, -1)
        masked = parametros[:, None, :]*self.mascara[None, :, :]
        inputs = torch.cat((observations, masked), dim=-1).reshape(n*3, -1)
        all_q = self.network(inputs).reshape(n, 3, 3)
        # Solo es válido Q_k en el pase que contiene los parámetros de k.
        return all_q.diagonal(dim1=1, dim2=2)
