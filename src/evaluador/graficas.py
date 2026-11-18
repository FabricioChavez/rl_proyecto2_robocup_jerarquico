"""Lee solamente CSV de resultados/logs y genera PNG y tablas de evaluación."""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

NOMBRES = {
    "persecucion": "Persecución",
    "conduccion": "Conducción / pase",
    "definicion": "Definición",
    "jerarquico": "Política jerárquica",
    "plana": "Política plana",
}
COLORES = {"ppo": "#2563eb", "a2c": "#d97706", "dqn": "#059669"}


def curva(ax, series, label, color):
    series = [s for s in series if len(s[0]) >= 2 and len(s[1]) >= 2]
    if not series:
        return
    start = max(s[0][0] for s in series)
    end = min(s[0][-1] for s in series)
    if start >= end:
        return
    grid = np.linspace(start, end, 150)
    values = np.array([np.interp(grid, x, y) for x, y in series])
    for x, y in series:
        ax.plot(x, y, color=color, alpha=0.16, lw=0.8)
    mean = values.mean(0)
    ax.plot(grid, mean, color=color, label=f"{label}: {len(series)} semillas", lw=2)
    if len(series) >= 3:
        ax.fill_between(
            grid, mean - values.std(0), mean + values.std(0), color=color, alpha=0.18
        )
    ax.grid(alpha=0.2)
    ax.set_xlabel("Decisiones de entrenamiento")
    ax.legend(fontsize=8)


def ejecutar(args):
    historical = getattr(args, "historical", False) or str(
        getattr(args, "run", "")
    ).startswith("sb3_")
    scope = "historico_sb3" if historical else "torch_propio"
    root = Path("resultados/logs")
    out = Path("resultados/graficas") / scope
    out.mkdir(parents=True, exist_ok=True)
    summaries = []
    figures = 0
    for task in args.tasks:
        for algo in ("ppo", "a2c", "dqn"):
            runs = [
                run
                for run in sorted((root / task / algo).glob("*"))
                if run.name.startswith("sb3_") == historical
                and not run.name.startswith("verificacion_")
            ]
            episodes = []
            updates = []
            for run in runs:
                path = run / "episodios.csv"
                if path.exists():
                    frame = pd.read_csv(path)
                    if not frame.empty:
                        episodes.append(frame)
                path = run / "actualizaciones.csv"
                if path.exists():
                    frame = pd.read_csv(path)
                    if not frame.empty:
                        updates.append(frame)
                for path in sorted(run.glob("evaluacion*.csv")):
                    frame = pd.read_csv(path)
                    if frame.empty:
                        continue
                    scenario = (
                        str(frame.escenario.iloc[0]) if "escenario" in frame else "2v1"
                    )
                    summaries.append(
                        dict(
                            tarea=task,
                            algoritmo=algo,
                            corrida=run.name,
                            escenario=scenario,
                            episodios=len(frame),
                            retorno=frame.retorno.mean(),
                            dispersion_episodica=frame.retorno.std(ddof=0),
                            exito=frame.exito.mean(),
                            goles=frame.gol.mean(),
                            duracion=frame.duracion.mean(),
                            csv=path.as_posix(),
                        )
                    )
            destination = out / task / algo
            destination.mkdir(parents=True, exist_ok=True)
            if episodes:
                fig, axes = plt.subplots(1, 2, figsize=(11, 4))
                for ax, column, title in [
                    (axes[0], "retorno", "Retorno base"),
                    (axes[1], "exito", "Tasa de éxito"),
                ]:
                    series = []
                    for f in episodes:
                        if column in f:
                            data = f[["pasos", column]].dropna().sort_values("pasos")
                            series.append(
                                (
                                    data.pasos.to_numpy(),
                                    data[column]
                                    .rolling(args.window, min_periods=1)
                                    .mean()
                                    .to_numpy(),
                                )
                            )
                    curva(ax, series, algo.upper(), COLORES[algo])
                    ax.set_ylabel(title)
                axes[1].set_ylim(-0.03, 1.03)
                axes[1].yaxis.set_major_formatter(PercentFormatter(1))
                fig.suptitle(
                    f"{NOMBRES[task]} · {algo.upper()} | media y ±1σ entre semillas; ventana {args.window}"
                )
                fig.tight_layout()
                fig.savefig(destination / "aprendizaje.png", dpi=160)
                plt.close(fig)
                figures += 1
            columns = {
                "ppo": [
                    "train/policy_gradient_loss",
                    "train/value_loss",
                    "train/approx_kl",
                ],
                "a2c": ["train/policy_loss", "train/value_loss"],
                "dqn": ["train/loss"],
            }[algo]
            if not historical:
                columns = {
                    "ppo": ["perdida_actor", "perdida_valor", "kl"],
                    "a2c": ["perdida_actor", "perdida_valor"],
                    "dqn": ["perdida_q"],
                }[algo]
            step_column = "time/total_timesteps" if historical else "pasos"
            if updates:
                fig, axes = plt.subplots(
                    1, len(columns), figsize=(5 * len(columns), 3.8), squeeze=False
                )
                for ax, column in zip(axes[0], columns):
                    series = []
                    for f in updates:
                        if column in f and step_column in f:
                            data = (
                                f[[step_column, column]]
                                .dropna()
                                .sort_values(step_column)
                            )
                            series.append(
                                (data.iloc[:, 0].to_numpy(), data.iloc[:, 1].to_numpy())
                            )
                    curva(ax, series, algo.upper(), COLORES[algo])
                    ax.set_ylabel(column.removeprefix("train/"))
                fig.suptitle(
                    f"{NOMBRES[task]} · {algo.upper()} · pérdidas y diagnóstico"
                )
                fig.tight_layout()
                fig.savefig(destination / "perdidas.png", dpi=160)
                plt.close(fig)
                figures += 1
    summary = pd.DataFrame(summaries)
    if not summary.empty:
        summary.to_csv(root / f"resumen_evaluaciones_{scope}.csv", index=False)
        for (task, scenario), group in summary.groupby(["tarea", "escenario"]):
            fig, axes = plt.subplots(2, 2, figsize=(10, 7))
            for ax, column, title in zip(
                axes.flat,
                ["retorno", "exito", "goles", "duracion"],
                ["Retorno base", "Tasa de éxito", "Tasa de gol", "Duración (ciclos)"],
            ):
                stats = (
                    group.groupby("algoritmo")[column]
                    .agg(["mean", "std", "count"])
                    .reindex(["ppo", "a2c", "dqn"])
                    .dropna(subset=["mean"])
                )
                ax.bar(
                    stats.index,
                    stats["mean"],
                    yerr=stats["std"].fillna(0),
                    capsize=4,
                    color=[COLORES[a] for a in stats.index],
                )
                ax.set_title(title)
                ax.grid(axis="y", alpha=0.2)
                if column in ("exito", "goles"):
                    ax.set_ylim(0, 1.1)
                    ax.yaxis.set_major_formatter(PercentFormatter(1))
            fig.suptitle(
                f"{NOMBRES[task]} · {scenario} · evaluación sin ventana\nBarras: media por modelo; dispersión entre semillas"
            )
            fig.tight_layout()
            destination = out / task
            destination.mkdir(parents=True, exist_ok=True)
            fig.savefig(destination / f"evaluacion_{scenario}.png", dpi=160)
            plt.close(fig)
            figures += 1
    print(f"{figures} graficas en {out}; {len(summary)} evaluaciones ({scope})")
    return summary
