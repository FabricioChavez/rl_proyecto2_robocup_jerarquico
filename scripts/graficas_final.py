"""Figura compacta de la Vía A: goles del sistema jerárquico y de la política plana
(200 episodios independientes, defensor 50, IC Wilson 95 %). Lee los evaluacion_2v1.csv.

Uso:  python scripts/graficas_final.py
"""

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

L = Path("resultados/logs")
AZUL, NARANJA, AGUA = "#2a78d6", "#eb6834", "#1baf7a"
SUPERFICIE, TEXTO, TEXTO_2, REJILLA = "#fcfcfb", "#0b0b0b", "#52514e", "#e1e0d9"
FAMILIA = {"imitación": AZUL, "imitación + RL": NARANJA, "sin jerarquía": AGUA}
FILAS = [
    ("Plana PPO (200k pasos)", "plana/ppo/plana50", "sin jerarquía"),
    ("Todo + DQN", "jerarquico/dqn/sistema_ft", "imitación + RL"),
    ("Todo + A2C", "jerarquico/a2c/sistema_ft", "imitación + RL"),
    ("Todo + PPO", "jerarquico/ppo/sistema_ft", "imitación + RL"),
    ("Coord. DAgger humano", "jerarquico/ppo/sistema", "imitación"),
    ("Coord. BC experto", "jerarquico/ppo/final_bc/jerarquico/ppo/final", "imitación"),
    ("Coord. DAgger experto", "jerarquico/ppo/final", "imitación"),
    ("Coord. DAgger + PPO", "jerarquico/ppo/final_ft/jerarquico/ppo/final", "imitación + RL"),
]


def wilson(k, n, z=1.96):
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0, c - m), min(1, c + m)


def main():
    datos = []
    for nombre, ruta, familia in FILAS:
        f = L / ruta / "evaluacion_2v1.csv"
        if f.exists():
            d = pd.read_csv(f)
            k, n = int(d.gol.sum()), len(d)
            datos.append((nombre, k / n, *wilson(k, n), n, familia))
    fig, ax = plt.subplots(figsize=(6.4, 3.3), facecolor=SUPERFICIE)
    ax.set_facecolor(SUPERFICIE)
    for i, (nombre, p, lo, hi, n, fam) in enumerate(datos):
        ax.barh(i, p, height=0.62, color=FAMILIA[fam], edgecolor=SUPERFICIE, linewidth=2)
        ax.errorbar(p, i, xerr=[[p - lo], [hi - p]], fmt="none", ecolor=TEXTO_2, elinewidth=1, capsize=3)
        ax.text(hi + 0.015, i, f"{p:.1%}".replace(".", ","), va="center", fontsize=8.5, color=TEXTO)
    ax.set_yticks(range(len(datos)), [d[0] for d in datos], fontsize=8.5, color=TEXTO)
    ax.axvline(0.8, color=TEXTO_2, linestyle=(0, (4, 3)), linewidth=1)
    ax.text(0.805, len(datos) - 0.55, "meta 80 %", fontsize=8, color=TEXTO_2)
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1], ["0", "20", "40", "60", "80", "100 %"])
    for lado in ("top", "right", "left"):
        ax.spines[lado].set_visible(False)
    ax.spines["bottom"].set_color("#c3c2b7")
    ax.tick_params(colors=TEXTO_2, labelsize=8.5, length=0)
    ax.grid(axis="x", color=REJILLA, linewidth=0.7)
    ax.set_axisbelow(True)
    marcas = [plt.Rectangle((0, 0), 1, 1, color=c) for c in FAMILIA.values()]
    ax.legend(marcas, FAMILIA.keys(), frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.45, -0.1), ncol=3)
    ax.set_title("Vía A: goles en 200 episodios de prueba (defensor 50)",
                 loc="left", fontsize=10, color=TEXTO, fontweight="bold")
    fig.tight_layout()
    fig.savefig("informe/figuras/jerarquia_final.png", dpi=200, facecolor=SUPERFICIE)
    for d in datos:
        print(d)


if __name__ == "__main__":
    main()
