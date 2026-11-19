"""Diagnóstico de la jerarquía: por qué terminan los episodios y qué habilidad estaba activa.

Juega N episodios con las habilidades de una corrida y un meta-controlador:
  red         la política jerárquica aprendida de esa corrida
  heuristica  regla fija: persecución si el balón está lejos, definición si x > X, si no conducción
Escribe un CSV por episodio con la causa de fin y el uso de cada habilidad.

Uso:
  python scripts/diagnostico_jerarquia.py --corrida mejor --meta red --episodios 50
  python scripts/diagnostico_jerarquia.py --corrida mejor --meta heuristica --episodios 50
"""

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from entorno.acciones import TASKS, relative  # noqa: E402
from entorno.escenarios import InteractiveEnv, apply_scene  # noqa: E402
from entorno.servidor import local_server  # noqa: E402
from modelo.pesos import cargar_politica  # noqa: E402
from rutas import ruta_pesos  # noqa: E402


def meta_heuristica(env, x_tiro, radio, radio_tiro=1.085):
    """Persecución solo con el balón suelto lejos; conducción lo persigue de cerca
    (su drible lo adelanta 2-4 m); definición con el balón al pie y cerca del arco."""
    w = env.world
    p = w.players[("Attack", 1)]
    d = relative(p, w.ball)[0]
    if d > radio:
        return 0
    return 2 if (d <= radio_tiro and p[0] > x_tiro) else 1


def causa(env, info):
    b = env.world.ball
    if info.get("goal"):
        return "gol"
    if abs(b[0]) >= 52.5 or abs(b[1]) > 34:
        return "fuera"
    if env.posesion_rival >= env.ciclos_posesion:
        return "rival_con_balon"
    if info.get("timeout"):
        return "tiempo"
    return "otro"


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--corrida", default="mejor")
    a.add_argument("--algoritmo", default="ppo")
    a.add_argument("--meta", choices=["red", "heuristica"], default="red")
    a.add_argument("--x-tiro", type=float, default=35.0)
    a.add_argument("--radio-persecucion", type=float, default=5.0)
    a.add_argument("--radio-tiro", type=float, default=1.085)
    a.add_argument("--episodios", type=int, default=50)
    a.add_argument("--fuerza-defensor", type=int, default=70)
    a.add_argument("--horizonte", type=int, default=200)
    a.add_argument("--puerto", type=int, default=7100)
    a.add_argument("--semilla-prueba", type=int, default=10000)
    a.add_argument("--salida", type=Path)
    args = a.parse_args()

    skills = [
        cargar_politica(None, ruta_pesos(t, args.algoritmo, 0, corrida=args.corrida), "cpu")
        for t in TASKS
    ]
    meta = None
    if args.meta == "red":
        meta = cargar_politica(
            args.algoritmo, ruta_pesos("global", args.algoritmo, 0, corrida=args.corrida), "cpu"
        )
    salida = args.salida or Path(
        f"resultados/logs/diagnostico_jerarquia/{args.corrida}_{args.meta}_x{args.x_tiro:g}_r{args.radio_persecucion:g}_t{args.radio_tiro:g}_f{args.fuerza_defensor}.csv"
    )
    salida.parent.mkdir(parents=True, exist_ok=True)
    filas, causas = [], Counter()
    with local_server(args.semilla_prueba, args.puerto) as (host, port):
        env = InteractiveEnv(
            task="global", host=host, port=port, scenario="2v1", skills=skills,
            shaping="none", horizon=args.horizonte, reglas="recuperacion",
        )
        env.conditions = {}
        env.X_DEFINICION, env.RADIO_TIRO = args.x_tiro, args.radio_tiro
        try:
            for ep in range(args.episodios):
                env.reset(seed=args.semilla_prueba + ep)
                apply_scene(env, dict(condiciones=dict(
                    defensor="perseguir", potencia_defensor=args.fuerza_defensor,
                    arquero_activo=True)))
                uso, x_max, info, cambios, anterior = Counter(), -99.0, {}, 0, None
                while True:
                    obs = env.observe()
                    opcion = (
                        int(meta.predict(obs, deterministic=True)[0]) if meta
                        else meta_heuristica(env, args.x_tiro, args.radio_persecucion, args.radio_tiro)
                    )
                    uso[TASKS[opcion]] += 1
                    cambios += int(anterior is not None and opcion != anterior)
                    anterior = opcion
                    _, _, done, trunc, info = env.step(opcion)
                    x_max = max(x_max, float(env.world.ball[0]))
                    if done or trunc:
                        break
                c = causa(env, info)
                causas[c] += 1
                filas.append(dict(
                    episodio=ep, causa=c, gol=int(c == "gol"), ciclos=env.t,
                    decisiones=sum(uso.values()), cambios=cambios,
                    persecucion=uso["persecucion"], conduccion=uso["conduccion"],
                    definicion=uso["definicion"], ultima=TASKS[anterior],
                    x_max_balon=round(x_max, 1),
                    x_final_atacante=round(float(env.world.players[("Attack", 1)][0]), 1),
                ))
        finally:
            env.close()
    with salida.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0]))
        w.writeheader()
        w.writerows(filas)
    n = len(filas)
    print(f"{args.corrida} meta={args.meta}: goles {causas['gol']}/{n} "
          f"| causas {dict(causas)} -> {salida}")
    for c in sorted(causas):
        sub = [r for r in filas if r["causa"] == c]
        print(f"  {c:16s} n={len(sub):3d}  última habilidad {Counter(r['ultima'] for r in sub)}"
              f"  x_max balón {np.mean([r['x_max_balon'] for r in sub]):5.1f}"
              f"  uso p/c/d {np.mean([r['persecucion'] for r in sub]):.1f}/"
              f"{np.mean([r['conduccion'] for r in sub]):.1f}/{np.mean([r['definicion'] for r in sub]):.1f}")


if __name__ == "__main__":
    main()
