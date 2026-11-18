"""Evaluación por lotes: misma física y políticas que visualizar, sin monitor ni ventanas."""

import csv
from pathlib import Path
import numpy as np
import pandas as pd
from entorno.escenarios import InteractiveEnv, apply_scene
from entorno.servidor import local_server
from modelo.pesos import cargar_politica
from rutas import carpeta, ruta_pesos, tarea_interna, nombre_tarea


def ejecutar(args):
    if args.episodes <= 0:
        raise ValueError("--episodios debe ser positivo")
    if args.weights and len(args.tasks) * len(args.algorithms) * len(args.seeds) != 1:
        raise ValueError("--pesos requiere una única tarea, algoritmo y semilla")
    summary = []
    for selected in args.tasks:
        task = tarea_interna(selected)
        for algo in args.algorithms:
            for seed in args.seeds:
                path = args.weights or ruta_pesos(task, algo, seed, corrida=args.run)
                model = cargar_politica(algo, path, args.device)
                skills = (
                    [
                        cargar_politica(
                            algo,
                            ruta_pesos(t, algo, seed, corrida=args.run),
                            args.device,
                        )
                        for t in ("persecucion", "conduccion", "definicion")
                    ]
                    if task == "global"
                    else None
                )
                out = (
                    (
                        Path(args.output)
                        / nombre_tarea(task)
                        / algo
                        / (args.run or f"semilla_{seed}")
                    )
                    if args.output
                    else carpeta("logs", task, algo, seed, args.run)
                )
                out = Path(out)
                out.mkdir(parents=True, exist_ok=True)
                for scenario in args.scenarios:
                    destination = out / f"evaluacion_{scenario}.csv"
                    rows = []
                    with (
                        local_server(args.test_seed, args.port) as (host, port),
                        destination.open("w", newline="", encoding="utf-8") as stream,
                    ):
                        writer = csv.DictWriter(
                            stream,
                            fieldnames=[
                                "episodio",
                                "semilla",
                                "escenario",
                                "retorno",
                                "exito",
                                "gol",
                                "duracion",
                                "velocidad_balon",
                                "fuerza_defensor",
                                "gamma",
                            ],
                        )
                        writer.writeheader()
                        env = InteractiveEnv(
                            task=task,
                            host=host,
                            port=port,
                            scenario=scenario,
                            skills=skills,
                            shaping="none",
                            gamma=args.gamma,
                            horizon=args.horizon,
                            option_steps=args.option_steps,
                        )
                        env.conditions = {}
                        try:
                            for ep in range(args.episodes):
                                initial_seed = args.test_seed + ep
                                rng = np.random.default_rng(initial_seed)
                                speed = float(rng.uniform(*args.ball_speed))
                                angle = float(rng.uniform(-np.pi, np.pi))
                                power = int(
                                    rng.integers(
                                        args.defender_power_range[0],
                                        args.defender_power_range[1] + 1,
                                    )
                                )
                                env.reset(seed=initial_seed)
                                apply_scene(
                                    env,
                                    dict(
                                        velocidad_balon=[
                                            speed * np.cos(angle),
                                            speed * np.sin(angle),
                                        ],
                                        condiciones=dict(
                                            defensor=args.defender_behavior,
                                            potencia_defensor=power,
                                            arquero_activo=not args.static_goalkeeper,
                                        ),
                                    ),
                                )
                                while True:
                                    action = int(
                                        model.predict(
                                            env.observe(), deterministic=True
                                        )[0]
                                    )
                                    _, _, done, trunc, info = env.step(action)
                                    if done or trunc:
                                        break
                                rows.append(
                                    dict(
                                        episodio=ep,
                                        semilla=initial_seed,
                                        escenario=scenario,
                                        retorno=env.base_return,
                                        exito=int(info["success"]),
                                        gol=int(info["goal"]),
                                        duracion=env.t,
                                        velocidad_balon=speed,
                                        fuerza_defensor=power,
                                        gamma=args.gamma,
                                    )
                                )
                                writer.writerow(rows[-1])
                                stream.flush()
                                if (ep + 1) % 50 == 0:
                                    print(
                                        f"{selected}/{algo}/s{seed}/{scenario}: {ep+1}/{args.episodes}",
                                        flush=True,
                                    )
                        finally:
                            env.close()
                    frame = pd.DataFrame(rows)
                    summary.append(
                        dict(
                            tarea=selected,
                            algoritmo=algo,
                            semilla=seed,
                            escenario=scenario,
                            episodios=len(frame),
                            retorno=frame.retorno.mean(),
                            dispersion=frame.retorno.std(ddof=0),
                            exito=frame.exito.mean(),
                            goles=frame.gol.mean(),
                            duracion=frame.duracion.mean(),
                            archivo=destination.as_posix(),
                        )
                    )
                    print(
                        f"{selected}/{algo}/s{seed}/{scenario}: retorno={frame.retorno.mean():.3f}, éxito={frame.exito.mean():.1%}, goles={frame.gol.mean():.1%} -> {destination}",
                        flush=True,
                    )
    if not args.no_graphs and not args.output:
        from evaluador.graficas import ejecutar as graficar

        graficar(args)
    return pd.DataFrame(summary)
