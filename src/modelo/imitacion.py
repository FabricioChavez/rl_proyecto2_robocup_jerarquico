"""Demostraciones humanas/heurísticas, BC y DAgger. Dataset CSV y pesos PTH."""

import numpy as np
import pandas as pd
import torch
from entorno.robocup import SoccerEnv
from entorno.acciones import LABELS, heuristic
from entorno.servidor import local_server
from entorno.pausa import pause_managed_server
from modelo.arquitecturas import crear_modelo
from modelo.pesos import (
    leer_checkpoint,
    arquitectura,
    guardar_checkpoint,
    restaurar_entrenamiento,
)
from rutas import carpeta


def ajustar(model, observaciones, acciones, epochs, lr, batch):
    x = torch.as_tensor(
        np.array(observaciones, dtype=np.float32, copy=True),
        dtype=torch.float32,
        device=model.device,
    )
    y = torch.as_tensor(
        np.array(acciones, dtype=np.int64, copy=True),
        dtype=torch.long,
        device=model.device,
    )
    optimizer = model.policy.optimizer
    for group in optimizer.param_groups:
        group["lr"] = lr
    losses = []
    model.policy.set_training_mode(True)
    for epoch in range(epochs):
        values = []
        for index in torch.randperm(len(x), device=model.device).split(batch):
            loss = -model.policy.get_distribution(x[index]).log_prob(y[index]).mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.policy.parameters(), 0.5)
            optimizer.step()
            model._n_updates += 1
            values.append(loss.item())
        losses.append(dict(epoca=epoch + 1, perdida=float(np.mean(values))))
    model.policy.set_training_mode(False)
    return losses


def etiqueta_humana(env):
    env.render()
    print(" | ".join(f"{i}: {label}" for i, label in enumerate(LABELS[env.task])))
    while True:
        answer = input("Acción experta; q guarda y termina: ").strip()
        if answer == "q":
            raise KeyboardInterrupt
        if answer.isdigit() and env.action_space.contains(int(answer)):
            return int(answer)


def ejecutar(args):
    if args.algo == "dqn":
        raise ValueError("BC/DAgger requieren la política categórica PPO o A2C")
    if args.task not in ("persecucion", "conduccion", "definicion"):
        raise ValueError("Recoge demostraciones por habilidad")
    if not 0 <= args.beta <= 1:
        raise ValueError("--beta debe estar entre 0 y 1")
    columns = [f"observacion_{i}" for i in range(34)]
    data = (
        pd.read_csv(args.dataset)
        if args.dataset.exists()
        else pd.DataFrame(columns=columns + ["accion", "origen", "tarea", "escenario"])
    )
    if len(data) and (
        set(data.tarea) != {args.task} or set(data.escenario) != {args.scenario}
    ):
        raise ValueError("El CSV contiene otra tarea/escenario")
    checkpoint = leer_checkpoint(args.resume) if args.resume else None
    if checkpoint and checkpoint.get("generador") != "torch_propio":
        raise ValueError(
            "BC/DAgger usa pesos propios; los SB3 se conservan solo para inferencia"
        )
    if checkpoint and (
        checkpoint["algoritmo"] != args.algo or checkpoint["tarea"] != args.task
    ):
        raise ValueError(
            "El checkpoint no corresponde a la tarea y algoritmo seleccionados"
        )
    args.network = (
        arquitectura(checkpoint)[2] if checkpoint else (args.network or [128, 128])
    )
    weights = carpeta("modelo_weights", args.task, args.mode, args.seed, args.run)
    weights.mkdir(parents=True, exist_ok=True)
    logs = carpeta("logs", args.task, args.mode, args.seed, args.run)
    logs.mkdir(parents=True, exist_ok=True)
    args.dataset.parent.mkdir(parents=True, exist_ok=True)

    def save_data():
        data.to_csv(args.dataset, index=False)

    with local_server(args.seed, args.port) as (host, port):
        env = SoccerEnv(
            task=args.task,
            host=host,
            port=port,
            scenario=args.scenario,
            gamma=args.gamma,
            horizon=args.horizon,
        )
        model = crear_modelo(env, args)
        if checkpoint:
            restaurar_entrenamiento(model, checkpoint)
        rng = np.random.default_rng(args.seed)
        losses = []
        try:
            if args.mode == "bc":
                if data.empty:
                    raise ValueError("Primero recoge datos con imitar demos")
                losses = ajustar(
                    model,
                    data[columns],
                    data.accion,
                    args.bc_epochs,
                    args.lr,
                    args.batch,
                )
            else:
                if args.mode == "dagger" and not checkpoint:
                    raise ValueError("DAgger requiere --continuar politica_inicial.pth")
                for iteration in range(args.rounds if args.mode == "dagger" else 1):
                    for episode in range(args.episodes):
                        observation, _ = env.reset(
                            seed=args.seed + iteration * 1000 + episode
                        )
                        while True:
                            if args.expert == "humano":
                                with pause_managed_server(port):
                                    label = etiqueta_humana(env)
                            else:
                                label = heuristic(args.task, env.world)
                            row = {k: float(v) for k, v in zip(columns, observation)}
                            row.update(
                                accion=label,
                                origen=args.expert,
                                tarea=args.task,
                                escenario=args.scenario,
                            )
                            data = pd.concat(
                                [data, pd.DataFrame([row])], ignore_index=True
                            )
                            action = (
                                label
                                if args.mode == "demos" or rng.random() < args.beta
                                else int(
                                    model.predict(observation, deterministic=True)[0]
                                )
                            )
                            observation, _, done, trunc, _ = env.step(action)
                            if done or trunc:
                                break
                    save_data()
                    if args.mode == "dagger":
                        current = ajustar(
                            model,
                            data[columns],
                            data.accion,
                            args.bc_epochs,
                            args.lr,
                            args.batch,
                        )
                        losses.extend(
                            dict(ronda=iteration + 1, **row) for row in current
                        )
            if args.mode != "demos":
                guardar_checkpoint(model, weights / "final.pth", args.task, args.algo)
                pd.DataFrame(losses).to_csv(logs / "imitacion.csv", index=False)
        except KeyboardInterrupt:
            print(
                "Se guardan las demostraciones disponibles; la ronda incompleta no se ajusta."
            )
        finally:
            save_data()
            env.close()
