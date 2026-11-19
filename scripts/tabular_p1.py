"""Baseline tabular de la Fase 1 (P1): Q-Learning con Q(s, a) en tabla.

Mismo servidor, tarea, recompensa y CSV que los agentes profundos, para que la
escalera de ablación compare solo el aproximador. El estado continuo se discretiza
como en P1: distancia al balón (6 tramos) x ángulo relativo al balón (12 tramos de 30°)
= 72 estados; acciones = las 4 primitivas originales de persecución
(dash 100, dash 50, girar +35, girar -35).

Uso:
  python scripts/tabular_p1.py --semilla 0 --pasos 30000 --corrida escalera_tabular_s0
"""

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from entorno.acciones import relative  # noqa: E402
from entorno.robocup import SoccerEnv  # noqa: E402
from entorno.servidor import local_server  # noqa: E402
from modelo.entrenamiento import agregar_csv  # noqa: E402

DISTANCIAS = np.array([1.0, 2.0, 4.0, 7.0, 12.0])  # 6 tramos
N_ANGULOS = 12
N_ACCIONES = 4


def estado(env):
    p, b = env.world.players[("Attack", 1)], env.world.ball
    d, ang = relative(p, b)
    i = int(np.searchsorted(DISTANCIAS, d))
    j = int(((ang + 180.0) % 360.0) // (360.0 / N_ANGULOS)) % N_ANGULOS
    return i * N_ANGULOS + j


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--tarea", default="persecucion", choices=["persecucion"])
    a.add_argument("--semilla", type=int, default=0)
    a.add_argument("--pasos", type=int, default=30000)
    a.add_argument("--alfa", type=float, default=0.1)
    a.add_argument("--gamma", type=float, default=0.99)
    a.add_argument("--epsilon-inicial", type=float, default=1.0)
    a.add_argument("--epsilon-final", type=float, default=0.05)
    a.add_argument("--fraccion-exploracion", type=float, default=0.5)
    a.add_argument("--shaping", choices=["none", "pbrs", "naive"], default="pbrs")
    a.add_argument("--fuerza-defensor", type=int, default=50)
    a.add_argument("--corrida")
    a.add_argument("--puerto", type=int, default=8100)
    args = a.parse_args()
    corrida = args.corrida or f"tabular_s{args.semilla}"
    logs = Path("resultados/logs") / args.tarea / "tabular" / corrida
    logs.mkdir(parents=True, exist_ok=True)
    (logs / "episodios.csv").unlink(missing_ok=True)
    rng = np.random.default_rng(args.semilla)
    q = np.zeros((len(DISTANCIAS) + 1) * N_ANGULOS * N_ACCIONES).reshape(-1, N_ACCIONES)
    with local_server(args.semilla, args.puerto) as (host, port):
        env = SoccerEnv(task=args.tarea, host=host, port=port, shaping=args.shaping,
                        gamma=args.gamma, reglas="recuperacion",
                        potencia_defensor=args.fuerza_defensor)
        try:
            env.reset(seed=args.semilla)
            s, pasos, inicio = estado(env), 0, time.monotonic()
            G_shaping, G_base, descuento, decisiones = 0.0, 0.0, 1.0, 0
            while pasos < args.pasos:
                frac = min(1.0, pasos / max(1, args.pasos * args.fraccion_exploracion))
                eps = args.epsilon_inicial + frac * (args.epsilon_final - args.epsilon_inicial)
                accion = int(rng.integers(N_ACCIONES)) if rng.random() < eps else int(q[s].argmax())
                _, r, term, trunc, info = env.step(accion)
                s2 = estado(env)
                objetivo = r + (0.0 if term else args.gamma * q[s2].max())
                q[s, accion] += args.alfa * (objetivo - q[s, accion])
                s, pasos = s2, pasos + 1
                G_shaping += r
                G_base += descuento * info["base_reward"]
                descuento *= args.gamma
                decisiones += 1
                if term or trunc:
                    agregar_csv(logs / "episodios.csv", dict(
                        retorno_shaping=G_shaping, decisiones=decisiones,
                        t=time.monotonic() - inicio, exito=int(info["success"]),
                        gol=int(info["goal"]), base_reward=info["base_reward"],
                        retorno=info.get("base_return", G_base), pasos=pasos))
                    env.reset()
                    s = estado(env)
                    G_shaping, G_base, descuento, decisiones = 0.0, 0.0, 1.0, 0
        finally:
            env.close()
    np.save(logs / "q_tabla.npy", q)
    print(f"Tabular {args.tarea} s{args.semilla}: {pasos} pasos -> {logs}")


if __name__ == "__main__":
    main()
