"""Una sola convención para pesos, registros y gráficas."""

from pathlib import Path

TAREAS = ("persecucion", "conduccion", "definicion", "jerarquico", "plana")
ALGORITMOS = ("ppo", "a2c", "dqn")


def tarea_interna(tarea):
    return "global" if tarea == "jerarquico" else tarea


def nombre_tarea(tarea):
    return "jerarquico" if tarea == "global" else tarea


def carpeta(tipo, tarea, algoritmo, semilla=0, corrida=None):
    return (
        Path("resultados")
        / tipo
        / nombre_tarea(tarea)
        / algoritmo
        / (corrida or f"semilla_{semilla}")
    )


def ruta_pesos(tarea, algoritmo, semilla=0, corrida=None):
    return carpeta("modelo_weights", tarea, algoritmo, semilla, corrida) / "final.pth"


def recorridos():
    for path in sorted(Path("resultados/modelo_weights").glob("*/*/*/final.pth")):
        tarea, algoritmo, corrida = path.parts[-4:-1]
        yield tarea, algoritmo, corrida, path, Path(
            "resultados/logs"
        ) / tarea / algoritmo / corrida
