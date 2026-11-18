"""Índice navegable y exportación de resultados a informe/."""

from pathlib import Path
from types import SimpleNamespace
import shutil
import os
import pandas as pd
from rutas import recorridos, TAREAS


def indice():
    lines = [
        "# Resultados",
        "",
        "Pesos: `modelo_weights/` · CSV: `logs/` · PNG: `graficas/`. Todos los modelos son `.pth`.",
        "Las corridas sb3_* son historicas. Las nuevas semilla_N usan algoritmos propios en PyTorch.",
        "",
        "| Tarea | Algoritmo | Corrida | Pesos | Registros |",
        "|---|---|---|---|---|",
    ]
    for task, algo, run, path, logs in recorridos():
        weight = path.relative_to("resultados").as_posix()
        log = logs.relative_to("resultados").as_posix()
        lines.append(
            f"| {task} | {algo} | {run} | [final.pth]({weight}) | [CSV]({log}/) |"
        )
    Path("resultados").mkdir(exist_ok=True)
    Path("resultados/README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def estado(args=None):
    rows = []
    for task, algo, run, path, logs in recorridos():
        rows.append(
            dict(
                tarea=task,
                algoritmo=algo,
                corrida=run,
                pesos=path.as_posix(),
                evaluaciones=len(list(logs.glob("evaluacion*.csv"))),
            )
        )
    if rows:
        print(pd.DataFrame(rows).to_string(index=False))
    else:
        print("No hay pesos. Ejecuta python proyecto.py entrenar --help")
    indice()
    return rows


def ejecutar(args):
    from evaluador.graficas import ejecutar as graficar

    summaries = {
        scope: graficar(
            SimpleNamespace(tasks=list(TAREAS), window=20, historical=historical)
        )
        for scope, historical in [("torch_propio", False), ("historico_sb3", True)]
    }
    indice()
    destination = Path(args.destino)
    destination.mkdir(parents=True, exist_ok=True)
    for source, name in [
        ("resultados/graficas", "graficas"),
        ("resultados/logs", "logs"),
    ]:
        if Path(source).exists():
            ignore = (
                (
                    lambda folder, names: (
                        [
                            name
                            for name in names
                            if name.startswith("resumen_evaluaciones")
                        ]
                        if Path(folder) == Path(source)
                        else []
                    )
                )
                if name == "logs"
                else None
            )
            shutil.copytree(
                source, destination / name, dirs_exist_ok=True, ignore=ignore
            )
    (destination / "logs/resumen_evaluaciones.csv").unlink(missing_ok=True)
    tables = destination / "tablas"
    tables.mkdir(exist_ok=True)
    for scope, summary in summaries.items():
        if not summary.empty:
            summary.to_csv(tables / f"resumen_evaluaciones_{scope}.csv", index=False)
    index_link = os.path.relpath(
        Path("resultados/README.md").resolve(), destination.resolve()
    ).replace("\\", "/")
    lines = [
        "# Informe de resultados",
        "",
        "Generado con `python proyecto.py informe`. No inicia entrenamientos ni abre ventanas.",
        "",
        "- [Gráficas](graficas/)",
        "- [Logs y evaluaciones por episodio](logs/)",
        "- [Tablas por procedencia](tablas/)",
        "historico_sb3 y torch_propio separan experimentos anteriores de las implementaciones propias.",
        f"- [Índice de pesos PTH](<{index_link}>)",
        "",
        "Los pesos se conservan en `resultados/modelo_weights/`; no se duplican en este informe.",
        "",
        "Las tasas de éxito de las subtareas no equivalen a la tasa de gol del ataque completo.",
    ]
    (destination / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Informe exportado a {destination}: PNG, CSV y enlaces a los PTH.")
