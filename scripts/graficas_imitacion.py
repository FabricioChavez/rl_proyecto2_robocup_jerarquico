"""Gráficas del Pilar 2B (imitación + ajuste fino con RL) a partir de los CSV existentes.

No entrena ni evalúa nada: solo lee
  resultados/logs/<tarea>/<algoritmo>/<corrida>/evaluacion_2v1.csv  (evaluar)
  resultados/logs/<tarea>/<algoritmo>/<corrida>/episodios.csv       (entrenar)
y escribe en informe/figuras/imitacion/:
  imitacion_exito_evaluacion.png   éxito por fase y variante, IC Wilson 95 %
  imitacion_curvas_ajuste_fino.png éxito durante el ajuste fino con RL
  imitacion_resumen.csv            la misma tabla en números
Las variantes sin CSV se omiten (aparecen al evaluarlas).

Uso:  python scripts/graficas_imitacion.py
"""

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

LOGS = Path("resultados/logs")
SALIDA = Path("informe/figuras/imitacion")

# Paleta categórica validada (orden fijo): imitación, imitación + RL, otra referencia.
AZUL, NARANJA, AGUA = "#2a78d6", "#eb6834", "#1baf7a"
SUPERFICIE, TEXTO, TEXTO_2, REJILLA = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
FAMILIAS = {"imitación": AZUL, "imitación + RL": NARANJA}
ALGORITMOS = {"ppo": AZUL, "a2c": NARANJA, "dqn": AGUA}

# (tarea, etiqueta, algoritmo, corrida, familia, carpeta de salida opcional)
VARIANTES = [
    ("persecucion", "BC humano", "ppo", "humano_bc", "imitación"),
    ("persecucion", "DAgger humano", "ppo", "humano", "imitación"),
    ("persecucion", "Imitación + PPO", "ppo", "sistema_ft", "imitación + RL"),
    ("persecucion", "Imitación + A2C", "a2c", "sistema_ft", "imitación + RL"),
    ("persecucion", "Imitación + DQN", "dqn", "sistema_ft", "imitación + RL"),
    ("conduccion", "BC humano", "ppo", "human_bc", "imitación"),
    ("conduccion", "DAgger humano", "ppo", "human_dagger", "imitación"),
    ("conduccion", "BC experto", "ppo", "expert_bc", "imitación"),
    ("conduccion", "DAgger experto", "ppo", "expert_dagger", "imitación"),
    ("conduccion", "Imitación + PPO", "ppo", "sistema_ft", "imitación + RL"),
    ("conduccion", "Imitación + A2C", "a2c", "sistema_ft", "imitación + RL"),
    ("conduccion", "Imitación + DQN", "dqn", "sistema_ft", "imitación + RL"),
    ("definicion", "BC humano", "ppo", "humano_bc", "imitación"),
    ("definicion", "DAgger humano", "ppo", "humano", "imitación"),
    ("definicion", "Imitación + PPO", "ppo", "sistema_ft", "imitación + RL"),
    ("definicion", "Imitación + A2C", "a2c", "sistema_ft", "imitación + RL"),
    ("definicion", "Imitación + DQN", "dqn", "sistema_ft", "imitación + RL"),
    ("definicion", "+ PPO zona de tiro", "ppo", "zona_tiro", "imitación + RL"),
    ("definicion", "+ DDQN zona de tiro", "dqn", "zona_tiro", "imitación + RL"),
    ("jerarquico", "Coord. DAgger humano", "ppo", "sistema", "imitación"),
    ("jerarquico", "Todo + PPO", "ppo", "sistema_ft", "imitación + RL"),
    ("jerarquico", "Todo + A2C", "a2c", "sistema_ft", "imitación + RL"),
    ("jerarquico", "Todo + DQN", "dqn", "sistema_ft", "imitación + RL"),
    ("jerarquico", "Coord. BC experto", "ppo", "final_bc/jerarquico/ppo/final", "imitación"),
    ("jerarquico", "Coord. DAgger experto", "ppo", "final", "imitación"),
    ("jerarquico", "Coord. DAgger + PPO", "ppo", "final_ft/jerarquico/ppo/final", "imitación + RL"),
]
TITULOS = {
    "persecucion": "Persecución",
    "conduccion": "Conducción",
    "definicion": "Definición (gol)",
    "jerarquico": "Jerárquico (gol)",
}


def wilson(exitos, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = exitos / n
    centro = (p + z * z / (2 * n)) / (1 + z * z / n)
    margen = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centro - margen), min(1.0, centro + margen)


def estilo(ax):
    ax.set_facecolor(SUPERFICIE)
    for lado in ("top", "right", "left"):
        ax.spines[lado].set_visible(False)
    ax.spines["bottom"].set_color(REJILLA)
    ax.tick_params(colors=TEXTO_2, labelsize=9, length=0)
    ax.grid(axis="x", color=REJILLA, linewidth=0.8)
    ax.set_axisbelow(True)


def resumen():
    filas = []
    for tarea, etiqueta, algo, corrida, familia in VARIANTES:
        f = LOGS / tarea / algo / corrida / "evaluacion_2v1.csv"
        if not f.exists():
            continue
        d = pd.read_csv(f)
        columna = "gol" if tarea in ("definicion", "jerarquico") else "exito"
        n, k = len(d), int(d[columna].sum())
        bajo, alto = wilson(k, n)
        fuerza = "/".join(str(int(v)) for v in sorted(d.fuerza_defensor.unique()))
        filas.append(
            dict(tarea=tarea, variante=etiqueta, familia=familia, algoritmo=algo,
                 corrida=corrida, episodios=n, exito=k / n, ic95_bajo=bajo,
                 ic95_alto=alto, fuerza_defensor=fuerza, archivo=f.as_posix())
        )
    return pd.DataFrame(filas)


def figura_evaluacion(tabla):
    tareas = [t for t in TITULOS if t in set(tabla.tarea)]
    if not tareas:
        print("No hay evaluaciones todavía: corre 'python proyecto.py evaluar ...' primero.")
        return
    alturas = [max(2, (tabla.tarea == t).sum()) for t in tareas]
    fig, ejes = plt.subplots(
        len(tareas), 1, figsize=(8, 0.5 * sum(alturas) + 1.2 * len(tareas)),
        gridspec_kw=dict(height_ratios=alturas), facecolor=SUPERFICIE,
    )
    ejes = [ejes] if len(tareas) == 1 else list(ejes)
    for ax, tarea in zip(ejes, tareas):
        t = tabla[tabla.tarea == tarea].iloc[::-1]
        y = range(len(t))
        colores = [FAMILIAS[f] for f in t.familia]
        ax.barh(y, t.exito, height=0.62, color=colores, edgecolor=SUPERFICIE, linewidth=2)
        ax.errorbar(t.exito, y, xerr=[t.exito - t.ic95_bajo, t.ic95_alto - t.exito],
                    fmt="none", ecolor=TEXTO_2, elinewidth=1, capsize=3)
        for yi, (_, r) in zip(y, t.iterrows()):
            ax.text(min(r.ic95_alto + 0.02, 1.02), yi, f"{r.exito:.0%}  (n={r.episodios})",
                    va="center", fontsize=8.5, color=TEXTO)
        ax.set_yticks(list(y), t.variante, color=TEXTO)
        ax.set_xlim(0, 1.18)
        ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], ["0", "20", "40", "60", "80", "100 %"])
        fuerzas = ", ".join(sorted(set(t.fuerza_defensor)))
        ax.set_title(f"{TITULOS[tarea]}  ·  defensor a potencia {fuerzas}",
                     loc="left", fontsize=11, color=TEXTO, fontweight="bold")
        if tarea == "jerarquico":
            ax.axvline(0.8, color=TEXTO_2, linestyle=(0, (4, 3)), linewidth=1)
            ax.text(0.8, len(t) - 0.4, " meta 80 %", fontsize=8, color=TEXTO_2, va="bottom")
        estilo(ax)
    marcas = [plt.Rectangle((0, 0), 1, 1, color=c) for c in FAMILIAS.values()]
    fig.legend(marcas, FAMILIAS.keys(), loc="upper right", frameon=False, fontsize=9,
               ncol=2, bbox_to_anchor=(0.98, 1.0))
    fig.suptitle("Éxito en evaluación independiente (n episodios por barra, IC Wilson 95 %)",
                 x=0.02, ha="left", fontsize=12, color=TEXTO)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    destino = SALIDA / "imitacion_exito_evaluacion.png"
    fig.savefig(destino, dpi=160, facecolor=SUPERFICIE)
    plt.close(fig)
    print(f"Figura: {destino}")


def figura_curvas():
    tareas = [t for t in TITULOS
              if any((LOGS / t / a / "sistema_ft" / "episodios.csv").exists() for a in ALGORITMOS)]
    if not tareas:
        print("No hay logs de ajuste fino (sistema_ft) todavía.")
        return
    fig, ejes = plt.subplots(1, len(tareas), figsize=(3.6 * len(tareas), 3.4),
                             facecolor=SUPERFICIE, sharey=True)
    ejes = [ejes] if len(tareas) == 1 else list(ejes)
    for ax, tarea in zip(ejes, tareas):
        for algo, color in ALGORITMOS.items():
            f = LOGS / tarea / algo / "sistema_ft" / "episodios.csv"
            if not f.exists():
                continue
            d = pd.read_csv(f)
            columna = "gol" if tarea in ("definicion", "jerarquico") else "exito"
            ventana = max(20, len(d) // 25)
            media = d[columna].rolling(ventana, min_periods=ventana // 2).mean()
            x = d.pasos - d.pasos.iloc[0]
            ax.plot(x, media, color=color, linewidth=2, label=algo.upper())
        ax.set_title(TITULOS[tarea], loc="left", fontsize=10.5, color=TEXTO, fontweight="bold")
        ax.set_ylim(0, 1.02)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0], ["0", "25", "50", "75", "100 %"])
        ax.set_xlabel("pasos de ajuste fino" if tarea != "jerarquico" else "decisiones",
                      fontsize=9, color=TEXTO_2)
        estilo(ax)
        ax.grid(axis="y", color=REJILLA, linewidth=0.8)
        ax.grid(axis="x", visible=False)
    ejes[0].legend(frameon=False, fontsize=9, loc="lower right")
    fig.suptitle("Ajuste fino con RL (PBRS) desde los pesos de imitación: éxito móvil",
                 x=0.02, ha="left", fontsize=12, color=TEXTO)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    destino = SALIDA / "imitacion_curvas_ajuste_fino.png"
    fig.savefig(destino, dpi=160, facecolor=SUPERFICIE)
    plt.close(fig)
    print(f"Figura: {destino}")


if __name__ == "__main__":
    SALIDA.mkdir(parents=True, exist_ok=True)
    tabla = resumen()
    if len(tabla):
        tabla.to_csv(SALIDA / "imitacion_resumen.csv", index=False)
        print(tabla[["tarea", "variante", "episodios", "exito", "ic95_bajo", "ic95_alto",
                     "fuerza_defensor"]].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    figura_evaluacion(tabla)
    figura_curvas()
