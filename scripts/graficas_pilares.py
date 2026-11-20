"""Gráficas y tabla de los Pilares 2 y 3 a partir de los CSV de scripts/escalera_ablacion.sh.

Lee (3 semillas, mismo presupuesto de 30 000 pasos, defensor a potencia 50):
  resultados/logs/persecucion/<algo>/<corrida>_s<k>/episodios.csv        curvas de aprendizaje
  resultados/logs/persecucion/<algo>/<corrida>_s<k>/actualizaciones.csv  pérdidas
  resultados/logs/<tarea>/<algo>/<corrida>_s<k>/evaluacion_2v1.csv       200 episodios de prueba
Escribe en informe/figuras/pilares/:
  pilar3_curvas.png      retorno episódico G0 (sin shaping), media ± 1σ entre semillas
  pilar3_perdidas.png    pérdidas y Q media (DQN frente a DDQN)
  pilar2_shaping.png     PBRS frente a shaping ingenuo y sin shaping (persecución y conducción)
  pilar3_tabla.csv       retorno asintótico, pasos hasta el 80 %, estabilidad y éxito en prueba

Uso:  python scripts/graficas_pilares.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LOGS = Path("resultados/logs")
SALIDA = Path("informe/figuras/pilares")
SEMILLAS = (0, 1, 2)
REJA = np.arange(0, 30001, 1000)

# Colores consistentes por algoritmo en las figuras del informe.
C = ["#8b8b83", "#2a78d6", "#1baf7a", "#eda100", "#eb6834", "#008300"]
SUPERFICIE, TEXTO, TEXTO_2, REJILLA = "#fcfcfb", "#0b0b0b", "#52514e", "#e1e0d9"

# nombre, tarea, algoritmo, prefijo de corrida, color
ESCALERA = [
    ("Q tabular", "persecucion", "tabular", "escalera", C[0]),
    ("DQN", "persecucion", "dqn", "escalera", C[1]),
    ("Double DQN", "persecucion", "dqn", "escalera_ddqn", C[2]),
    ("A2C", "persecucion", "a2c", "escalera", C[3]),
    ("PPO", "persecucion", "ppo", "escalera", C[4]),
    ("BC humano + PPO", "persecucion", "ppo", "escalera_bc", C[5]),
]
SHAPING_P = [
    ("sin shaping", "persecucion", "ppo", "escalera_none", C[0]),
    ("PBRS", "persecucion", "ppo", "escalera", C[1]),
    ("ingenuo", "persecucion", "ppo", "escalera_naive", C[2]),
]
SHAPING_C = [
    ("sin shaping", "conduccion", "ppo", "hack_none", C[0]),
    ("PBRS", "conduccion", "ppo", "hack_pbrs", C[1]),
    ("ingenuo", "conduccion", "ppo", "hack_naive", C[2]),
]


def episodios(tarea, algo, corrida, s):
    f = LOGS / tarea / algo / f"{corrida}_s{s}" / "episodios.csv"
    if not f.exists():
        return None
    d = pd.read_csv(f)
    # Con --continuar los pasos parten del checkpoint: se cuentan desde el inicio de la corrida.
    d["x"] = d.pasos - d.pasos.iloc[0] + d.decisiones.iloc[0]
    return d


def curva(d, columna, ventana=50):
    suave = d[columna].rolling(ventana, min_periods=max(5, ventana // 5)).mean()
    return np.interp(REJA, d.x, suave.bfill())


def semillas(tarea, algo, corrida, columna, ventana=50):
    filas = [episodios(tarea, algo, corrida, s) for s in SEMILLAS]
    filas = [d for d in filas if d is not None and len(d) > 10]
    if not filas:
        return None
    return np.array([curva(d, columna, ventana) for d in filas]), filas


def estilo(ax, titulo):
    ax.set_facecolor(SUPERFICIE)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    for lado in ("left", "bottom"):
        ax.spines[lado].set_color("#c3c2b7")
    ax.tick_params(colors=TEXTO_2, labelsize=8.5, length=0)
    ax.grid(color=REJILLA, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.set_title(titulo, loc="left", fontsize=10, color=TEXTO, fontweight="bold")
    ax.set_xticks([0, 10000, 20000, 30000], ["0", "10k", "20k", "30k"])


def banda(ax, grupo, columna, ventana=50, ylabel=None):
    for nombre, tarea, algo, corrida, color in grupo:
        r = semillas(tarea, algo, corrida, columna, ventana)
        if r is None:
            continue
        m, s = r[0].mean(0), r[0].std(0)
        ax.fill_between(REJA, m - s, m + s, color=color, alpha=0.16, linewidth=0)
        ax.plot(REJA, m, color=color, linewidth=2, label=f"{nombre} (n={len(r[0])})")
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9, color=TEXTO_2)
    ax.set_xlabel("pasos de entorno", fontsize=9, color=TEXTO_2)


def figura_curvas():
    fig, ejes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True, facecolor=SUPERFICIE)
    estilo(ejes[0], "Basados en valor")
    estilo(ejes[1], "Actor-crítico (+ imitación)")
    banda(ejes[0], ESCALERA[:3], "retorno", ylabel="retorno episódico G0 (sin shaping)")
    banda(ejes[1], ESCALERA[3:], "retorno")
    for ax in ejes:
        ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    fig.suptitle("Escalera de ablación en persecución: media ± 1σ sobre 3 semillas, defensor 50",
                 x=0.01, ha="left", fontsize=11.5, color=TEXTO)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(SALIDA / "pilar3_curvas.png", dpi=300, facecolor=SUPERFICIE)
    plt.close(fig)


def actualizaciones(algo, corrida):
    out = []
    for s in SEMILLAS:
        f = LOGS / "persecucion" / algo / f"{corrida}_s{s}" / "actualizaciones.csv"
        if f.exists():
            d = pd.read_csv(f)
            d["x"] = d.pasos - d.pasos.iloc[0]
            out.append(d)
    return out


def media_columna(ds, columna, ventana=10):
    ys = []
    for d in ds:
        v = d[columna].astype(float)
        if v.notna().sum() < 3:
            continue
        v = v.rolling(ventana, min_periods=1).mean()
        ok = v.notna()
        ys.append(np.interp(REJA, d.x[ok], v[ok]))
    return np.array(ys) if ys else None


def figura_perdidas():
    paneles = [
        ("Pérdida TD (Huber)", [("DQN", "dqn", "escalera", C[1]), ("Double DQN", "dqn", "escalera_ddqn", C[2])], "perdida_q"),
        ("Q media de la red online", [("DQN", "dqn", "escalera", C[1]), ("Double DQN", "dqn", "escalera_ddqn", C[2])], "q_media"),
        ("Pérdida del crítico", [("A2C", "a2c", "escalera", C[3]), ("PPO", "ppo", "escalera", C[4]), ("BC + PPO", "ppo", "escalera_bc", C[5])], "perdida_valor"),
        ("Entropía de la política", [("A2C", "a2c", "escalera", C[3]), ("PPO", "ppo", "escalera", C[4]), ("BC + PPO", "ppo", "escalera_bc", C[5])], "entropia"),
    ]
    fig, ejes = plt.subplots(2, 2, figsize=(8, 5.2), facecolor=SUPERFICIE)
    for ax, (titulo, series, columna) in zip(ejes.flat, paneles):
        estilo(ax, titulo)
        for nombre, algo, corrida, color in series:
            ys = media_columna(actualizaciones(algo, corrida), columna)
            if ys is None:
                continue
            m, s = ys.mean(0), ys.std(0)
            ax.fill_between(REJA, m - s, m + s, color=color, alpha=0.16, linewidth=0)
            ax.plot(REJA, m, color=color, linewidth=2, label=nombre)
        ax.set_xlabel("pasos de entorno", fontsize=9, color=TEXTO_2)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Dinámica de optimización (media ± 1σ, 3 semillas)", x=0.01, ha="left",
                 fontsize=11.5, color=TEXTO)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(SALIDA / "pilar3_perdidas.png", dpi=300, facecolor=SUPERFICIE)
    plt.close(fig)


def figura_shaping():
    fig, ejes = plt.subplots(1, 3, figsize=(12, 3.5), facecolor=SUPERFICIE)
    estilo(ejes[0], "Persecución: G0 sin shaping")
    banda(ejes[0], SHAPING_P, "retorno", ylabel="G0")
    estilo(ejes[1], "Conducción: éxito real")
    banda(ejes[1], SHAPING_C, "exito", ventana=40, ylabel="tasa de éxito móvil")
    ejes[1].set_ylim(0, 1.02)
    estilo(ejes[2], "Conducción: retorno que ve el agente")
    banda(ejes[2], SHAPING_C, "retorno_shaping", ventana=40, ylabel="suma de recompensas (sin descuento)")
    for ax in ejes:
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle("PBRS frente a shaping ingenuo (ajuste fino PPO desde el mismo BC del experto en conducción)",
                 x=0.01, ha="left", fontsize=11.5, color=TEXTO)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(SALIDA / "pilar2_shaping.png", dpi=300, facecolor=SUPERFICIE)
    plt.close(fig)


def evaluacion(tarea, algo, corrida, s):
    f = LOGS / tarea / algo / f"{corrida}_s{s}" / "evaluacion_2v1.csv"
    if not f.exists():
        return None
    d = pd.read_csv(f)
    return d.exito.mean(), d.retorno.mean() if "retorno" in d else np.nan


def tabla():
    grupos = ESCALERA + SHAPING_P[0:1] + SHAPING_P[2:] + SHAPING_C
    todas = [semillas(t, a, c, "retorno") for _, t, a, c, _ in ESCALERA]
    todas = np.concatenate([r[0] for r in todas if r is not None])
    lo, hi = np.nanmin(todas[:, 1:]), np.nanmax(todas)
    umbral = lo + 0.8 * (hi - lo)
    filas = []
    for nombre, tarea, algo, corrida, _ in grupos:
        r = semillas(tarea, algo, corrida, "retorno")
        if r is None:
            continue
        curvas, ds = r
        asint = [d.retorno.iloc[int(0.8 * len(d)):].mean() for d in ds]
        exito_ent = [d.exito.iloc[int(0.8 * len(d)):].mean() for d in ds]
        ruido = [d.retorno.iloc[int(0.8 * len(d)):].std() for d in ds]
        cruce = []
        for y in curvas:
            idx = np.nonzero(y >= umbral)[0]
            cruce.append(REJA[idx[0]] if len(idx) else np.nan)
        ev = [evaluacion(tarea, algo, corrida, s) for s in SEMILLAS]
        ev = [e[0] for e in ev if e is not None]
        filas.append(dict(
            tarea=tarea, metodo=nombre, shaping=("none" if "none" in corrida else "naive" if "naive" in corrida else "pbrs"),
            semillas=len(ds),
            G0_asintotico=np.mean(asint), G0_sigma_semillas=np.std(asint),
            G0_sigma_intra=np.mean(ruido),
            pasos_80=np.nanmean(cruce) if tarea == "persecucion" and np.isfinite(cruce).any() else np.nan,
            semillas_que_llegan_80=int(np.sum(~np.isnan(cruce))) if tarea == "persecucion" else np.nan,
            exito_entrenamiento=np.mean(exito_ent),
            exito_prueba=np.mean(ev) if ev else np.nan,
            exito_prueba_sigma=np.std(ev) if ev else np.nan,
            n_prueba=200 * len(ev),
        ))
    t = pd.DataFrame(filas)
    t.to_csv(SALIDA / "pilar3_tabla.csv", index=False)
    print(f"Umbral 80 %: G0 = {umbral:.3f} (mín {lo:.3f}, máx {hi:.3f})")
    print(t.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    SALIDA.mkdir(parents=True, exist_ok=True)
    figura_curvas()
    figura_perdidas()
    figura_shaping()
    tabla()
    print(f"Figuras en {SALIDA}")
