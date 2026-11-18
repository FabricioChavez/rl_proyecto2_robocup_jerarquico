"""Checkpoints PyTorch .pth: política, optimizador, pasos y replay cuando existe."""

import os
import random
import numpy as np
import torch
from modelo.arquitecturas.redes import ActorCritico, PoliticaQ


def dispositivo_elegido(device=None):
    value = device or os.getenv("ROBOCUP_DEVICE", "cpu")
    if value.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA solicitada pero no disponible. Usa --dispositivo cpu o habilita NVIDIA en Docker."
        )
    return value


def _cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: _cpu(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_cpu(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_cpu(v) for v in value)
    return value


def leer_checkpoint(path):
    data = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(data, dict) or data.get("formato") not in (1, 2):
        raise ValueError(f"{path}: no es un checkpoint .pth de este proyecto")
    return data


def arquitectura(data):
    state = data["politica"]
    prefix = (
        "q_net.q_net." if data["algoritmo"] == "dqn" else "mlp_extractor.policy_net."
    )
    layers = sorted(
        (int(k[len(prefix) :].split(".")[0]), v)
        for k, v in state.items()
        if k.startswith(prefix) and k.endswith(".weight")
    )
    if not layers:
        raise ValueError("No se reconoce la arquitectura del checkpoint")
    observation = layers[0][1].shape[1]
    hidden = [tensor.shape[0] for _, tensor in layers]
    actions = (
        hidden.pop()
        if data["algoritmo"] == "dqn"
        else state["action_net.weight"].shape[0]
    )
    return int(observation), int(actions), hidden


class PoliticaCargada:
    def __init__(self, data, device):
        observation, actions, hidden = arquitectura(data)
        constructor = PoliticaQ if data["algoritmo"] == "dqn" else ActorCritico
        self.policy = constructor(observation, actions, hidden).to(device)
        self.policy.load_state_dict(data["politica"])
        self.policy.set_training_mode(False)
        self.device = torch.device(device)
        self.num_timesteps = int(data["pasos"])
        self.checkpoint = data

    def predict(self, observation, deterministic=True):
        return self.policy.predict(observation, deterministic=deterministic)


def cargar_politica(algoritmo, path, device=None):
    data = leer_checkpoint(path)
    if data["algoritmo"] != algoritmo:
        raise ValueError(f'El archivo contiene {data["algoritmo"]}, no {algoritmo}')
    return PoliticaCargada(data, dispositivo_elegido(device))


def guardar_checkpoint(model, path, tarea, algoritmo):
    replay = None
    buffer = getattr(model, "replay_buffer", None)
    if buffer is not None and buffer.size():
        count = buffer.buffer_size if buffer.full else buffer.pos
        replay = {
            name: torch.as_tensor(getattr(buffer, name)[:count].copy())
            for name in (
                "observations",
                "next_observations",
                "actions",
                "rewards",
                "dones",
                "timeouts",
            )
        }
        replay.update(pos=int(buffer.pos), full=bool(buffer.full))
    numpy_rng = np.random.get_state()
    data = dict(
        formato=2,
        generador="torch_propio",
        origen_pesos=model.origen_pesos,
        algoritmo=algoritmo,
        tarea=tarea,
        pasos=int(model.num_timesteps),
        llamadas=int(getattr(model, "_n_calls", model.num_timesteps)),
        episodios=int(getattr(model, "_episode_num", 0)),
        actualizaciones=int(model._n_updates),
        politica=_cpu(model.policy.state_dict()),
        optimizador=_cpu(model.policy.optimizer.state_dict()),
        replay=replay,
        rng=dict(
            python=random.getstate(),
            torch=torch.get_rng_state(),
            cuda=(
                torch.cuda.get_rng_state_all()
                if str(model.device).startswith("cuda")
                else []
            ),
            numpy=(
                numpy_rng[0],
                numpy_rng[1].tolist(),
                int(numpy_rng[2]),
                int(numpy_rng[3]),
                float(numpy_rng[4]),
            ),
        ),
    )
    # One PTH per checkpoint; no archive, sidecar config or temporary model files.
    torch.save(data, path)


def restaurar_entrenamiento(model, data):
    model.policy.load_state_dict(data["politica"])
    settings = [
        {k: v for k, v in group.items() if k != "params"}
        for group in model.policy.optimizer.param_groups
    ]
    model.policy.optimizer.load_state_dict(data["optimizador"])
    for group, values in zip(model.policy.optimizer.param_groups, settings):
        group.update(values)
    model.origen_pesos = data.get(
        "origen_pesos", data.get("generador", "sb3_historico")
    )
    model.num_timesteps = int(data["pasos"])
    model._n_updates = int(data["actualizaciones"])
    if hasattr(model, "_n_calls"):
        model._n_calls = int(data.get("llamadas", data["pasos"]))
    model._episode_num = int(data.get("episodios", 0))
    replay = data.get("replay")
    if replay is not None:
        buffer = model.replay_buffer
        count = len(replay["observations"])
        if count > buffer.buffer_size:
            raise ValueError(
                "El replay guardado supera --buffer; aumenta ese argumento"
            )
        for name in (
            "observations",
            "next_observations",
            "actions",
            "rewards",
            "dones",
            "timeouts",
        ):
            getattr(buffer, name)[:count] = replay[name].numpy()
        buffer.full = count == buffer.buffer_size
        buffer.pos = int(replay["pos"]) if buffer.full else count
    elif data["algoritmo"] == "dqn":
        # Historical ZIP runs did not contain replay; collect fresh experience first.
        model.learning_starts = model.num_timesteps + model.learning_starts
    rng = data.get("rng")
    if rng:
        random.setstate(rng["python"])
        torch.set_rng_state(rng["torch"])
        if (
            str(model.device).startswith("cuda")
            and len(rng.get("cuda", [])) == torch.cuda.device_count()
        ):
            torch.cuda.set_rng_state_all(rng["cuda"])
        tag, keys, pos, gauss, cached = rng["numpy"]
        np.random.set_state(
            (tag, np.asarray(keys, dtype=np.uint32), pos, gauss, cached)
        )
