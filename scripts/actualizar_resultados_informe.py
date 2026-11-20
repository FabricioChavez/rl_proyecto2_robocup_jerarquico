"""Regenera las figuras de resultados desde CSV; no entrena, evalúa ni compila LaTeX.

Ejecutar desde la raíz: python scripts/actualizar_resultados_informe.py
La trazabilidad de cada imagen queda en informe/figuras/fuentes_resultados.csv.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import graficas_final as final
import graficas_imitacion as imitacion
import graficas_pilares as pilares


LOGS = Path("resultados/logs")
FIGURAS = Path("informe/figuras")
TAREAS = {"persecucion": "Persecución", "conduccion": "Conducción", "definicion": "Definición"}
COLORES = {"dqn": "#2a78d6", "a2c": "#eda100", "ppo": "#eb6834"}


def aprendizaje_subtareas(registrar):
    """Desde cero, ajuste fino y evaluación; las curvas usan retorno base."""
    figura = "imitacion/aprendizaje_subtareas.png"
    fig, ejes = plt.subplots(3, 3, figsize=(10.6, 7.4), layout="constrained",
                             gridspec_kw={"width_ratios": [1.15, 1.15, 0.85]})
    for fila, (tarea, titulo) in enumerate(TAREAS.items()):
        n_semillas = 3 if tarea == "persecucion" else 2
        for indice, (algo, color) in enumerate(COLORES.items()):
            # Se promedian solo pasos cubiertos por todas las semillas disponibles.
            curvas = []
            for semilla in range(n_semillas):
                corrida = f"escalera_s{semilla}" if tarea == "persecucion" else f"semilla_{semilla}"
                archivo = LOGS / tarea / algo / corrida / "episodios.csv"
                datos = pd.read_csv(archivo)
                x = datos.pasos - datos.pasos.iloc[0] + datos.decisiones.iloc[0]
                media = datos.retorno.rolling(50, min_periods=10).mean()
                validos = media.notna()
                curvas.append((x[validos].to_numpy(), media[validos].to_numpy()))
                registrar(figura, archivo, f"desde cero; hasta {n_semillas} semillas; ventana 50; cobertura variable indicada en leyenda; sin extrapolar")
            inicio = max(x[0] for x, _ in curvas)
            fin = min(x[-1] for x, _ in curvas)
            reja = np.linspace(inicio, fin, 250)
            valores = np.array([np.interp(reja, x, y) for x, y in curvas])
            media, sigma = valores.mean(0), valores.std(0)
            cobertura_variable = len(curvas) == 2 and any(x[-1] > fin + 100 for x, _ in curvas)
            etiqueta = f"{algo.upper()} (n=2→1)" if cobertura_variable else algo.upper()
            ejes[fila, 0].plot(reja, media, color=color, label=etiqueta, linewidth=1.3)
            ejes[fila, 0].fill_between(reja, media - sigma, media + sigma, color=color, alpha=0.14)
            # Con dos corridas desiguales, conservar también la cola de la más larga.
            if len(curvas) == 2:
                for x, y in curvas:
                    if x[-1] > fin + 100:
                        cola = x > fin
                        ejes[fila, 0].plot(np.r_[reja[-1], x[cola]], np.r_[media[-1], y[cola]],
                            color=color, linestyle="-", linewidth=1.3)

            archivo = LOGS / tarea / algo / "sistema_ft/episodios.csv"
            datos = pd.read_csv(archivo)
            pasos = datos.pasos - datos.pasos.iloc[0] + datos.decisiones.iloc[0]
            ventana = 200 if tarea == "definicion" else 100
            media = datos.retorno.rolling(ventana, min_periods=ventana // 2).mean()
            ejes[fila, 1].plot(pasos, media, color=color, label=algo.upper(), linewidth=1.3)
            registrar(figura, archivo, f"ajuste fino; una corrida; ventana {ventana}")

            archivo = LOGS / tarea / algo / "sistema_ft/evaluacion_2v1.csv"
            datos = pd.read_csv(archivo)
            if len(datos) != 200 or set(datos.fuerza_defensor) != {50}:
                raise ValueError(f"Condiciones de prueba distintas: {archivo}")
            k = int(datos["gol" if tarea == "definicion" else "exito"].sum())
            tasa = k / len(datos)
            bajo, alto = final.wilson(k, len(datos))
            ejes[fila, 2].bar(indice, 100 * tasa, color=color, width=0.58)
            ejes[fila, 2].errorbar(indice, 100 * tasa,
                yerr=[[100 * (tasa - bajo)], [100 * (alto - tasa)]],
                fmt="none", color="#52514e", capsize=3, linewidth=0.9)
            etiqueta = f"{100 * tasa:.1f}%".replace(".", ",")
            ejes[fila, 2].text(indice, 100 * alto + 2.5, etiqueta, ha="center", fontsize=8)
            registrar(figura, archivo, "prueba del ajuste fino; 200 episodios; defensor 50; Wilson 95 %")
        for columna, ax in enumerate(ejes[fila]):
            ax.tick_params(labelsize=8)
            ax.grid(axis="y", alpha=0.2)
            ax.set_axisbelow(True)
            ax.spines[["top", "right"]].set_visible(False)
            if columna < 2:
                ax.set_ylabel("retorno base G₀", fontsize=8)
                ax.set_ylim(-5, 10)
                ax.set_yticks([-5, 0, 5, 10])
                ubicacion = "upper right" if tarea == "conduccion" and columna == 0 else "lower right"
                ax.legend(frameon=False, ncol=3, fontsize=7, loc=ubicacion)
            else:
                ax.set_ylim(0, 115)
                ax.set_yticks([0, 50, 100])
                ax.set_xticks([0, 1, 2], ["DQN", "A2C", "PPO"])
                ax.set_ylabel("gol (%)" if tarea == "definicion" else "éxito (%)", fontsize=8)
        ejes[fila, 0].set_title(f"{titulo}\nDesde cero ({n_semillas} semillas)", loc="left", fontsize=9)
        ejes[fila, 1].set_title(f"{titulo}\nImitación + RL (1 corrida)", loc="left", fontsize=9)
        ejes[fila, 2].set_title(f"{titulo}\nPrueba: 200 episodios", loc="left", fontsize=9)
        limite = 30000 if tarea == "persecucion" else 10000
        ejes[fila, 0].set_xlim(0, limite)
        ejes[fila, 0].set_xticks([0, limite / 2, limite], ["0", "15k" if limite == 30000 else "5k", "30k" if limite == 30000 else "10k"])
        ejes[fila, 0].set_xlabel("pasos de entrenamiento", fontsize=8)
        ejes[fila, 1].set_xlim(0, 50000)
        ejes[fila, 1].set_xticks([0, 25000, 50000], ["0", "25k", "50k"])
        ejes[fila, 1].set_xlabel("pasos adicionales de RL", fontsize=8)
    imitacion.SALIDA.mkdir(parents=True, exist_ok=True)
    fig.savefig(imitacion.SALIDA / "aprendizaje_subtareas.png", dpi=300)
    plt.close(fig)


def reward_jerarquico(registrar):
    archivo = LOGS / "definicion/ppo/lejano_jer/episodios.csv"
    datos = pd.read_csv(archivo)
    media = datos.retorno.rolling(1000, min_periods=500).mean()
    fig, ax = plt.subplots(figsize=(5, 2.9), layout="constrained")
    ax.plot(np.arange(1, len(datos) + 1), media, color="#eb6834", linewidth=1.3)
    ax.set_title("Retorno durante el entrenamiento jerárquico", loc="left", fontsize=10)
    ax.set_xlabel("episodio de entrenamiento", fontsize=9)
    ax.set_ylabel("retorno base G₀", fontsize=9)
    ax.grid(alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8)
    destino = FIGURAS / "04_jerarquico/reward_y_goles.png"
    destino.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destino, dpi=300)
    plt.close(fig)
    registrar("04_jerarquico/reward_y_goles.png", archivo,
              "retorno base; definición dentro de jerarquía; media móvil de 1000 episodios")


def main():
    fuentes = []

    def registrar(figura, archivo, uso):
        if not archivo.is_file():
            raise FileNotFoundError(archivo)
        fuentes.append(dict(figura=figura, archivo=archivo.as_posix(), uso=uso))

    # Los mismos grupos y cálculos que las figuras de la escalera.
    pilares.SALIDA.mkdir(parents=True, exist_ok=True)
    pilares.figura_curvas()
    pilares.figura_perdidas()
    pilares.figura_shaping()
    pilares.tabla()
    for grupo, figura, columnas in [
        (pilares.ESCALERA, "pilares/pilar3_curvas.png", ["episodios.csv"]),
        (pilares.ESCALERA[1:], "pilares/pilar3_perdidas.png", ["actualizaciones.csv"]),
        (pilares.SHAPING_P + pilares.SHAPING_C, "pilares/pilar2_shaping.png", ["episodios.csv"]),
    ]:
        for _, tarea, algo, corrida, _ in grupo:
            for semilla in pilares.SEMILLAS:
                for nombre in columnas:
                    registrar(figura, LOGS / tarea / algo / f"{corrida}_s{semilla}" / nombre, "entrenamiento; tres semillas")

    aprendizaje_subtareas(registrar)
    reward_jerarquico(registrar)

    tabla = imitacion.resumen()
    tabla.to_csv(imitacion.SALIDA / "imitacion_resumen.csv", index=False)
    for archivo in tabla.archivo:
        registrar("imitacion/imitacion_resumen.csv", Path(archivo), "evaluación; no mezclar condiciones")

    # La comparación completa incluye las variantes posteriores de la exposición.
    final.main()
    resumen = []
    for nombre, ruta, _ in final.FILAS:
        archivo = LOGS / ruta / "evaluacion_2v1.csv"
        datos = pd.read_csv(archivo)
        if len(datos) != 200 or set(datos.fuerza_defensor) != {50}:
            raise ValueError(f"Condiciones de evaluación distintas: {archivo}")
        n, k = len(datos), int(datos.gol.sum())
        bajo, alto = final.wilson(k, n)
        resumen.append(dict(variante=nombre, episodios=n, goles=k, tasa=k / n,
                            ic95_bajo=bajo, ic95_alto=alto, retorno=datos.retorno.mean(),
                            archivo=archivo.as_posix()))
        registrar("top_policies/jerarquia_final.png", archivo, "200 episodios; defensor 50; IC Wilson 95 %")
    pd.DataFrame(resumen).to_csv(FIGURAS / "top_policies/jerarquia_resumen.csv", index=False)
    # Fuentes de las cifras de prueba que acompañan la escalera y el shaping.
    for _, tarea, algo, corrida, _ in pilares.ESCALERA + pilares.SHAPING_P + pilares.SHAPING_C:
        if algo == "tabular":
            continue
        for semilla in pilares.SEMILLAS:
            archivo = LOGS / tarea / algo / f"{corrida}_s{semilla}/evaluacion_2v1.csv"
            datos = pd.read_csv(archivo)
            if len(datos) != 200 or set(datos.fuerza_defensor) != {50}:
                raise ValueError(f"Condiciones de evaluación distintas: {archivo}")
            registrar("pilares/pilar3_tabla.csv", archivo, "200 episodios por semilla")
    pd.DataFrame(fuentes).drop_duplicates().to_csv(FIGURAS / "fuentes_resultados.csv", index=False)
    print(pd.DataFrame(resumen)[["variante", "goles", "tasa", "ic95_bajo", "ic95_alto"]].to_string(index=False))


if __name__ == "__main__":
    main()
