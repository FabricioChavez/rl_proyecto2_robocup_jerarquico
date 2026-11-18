# Exploración opcional

`visualizacion_p2_jeffrey.ipynb` adapta el dibujo y la animación de persecución del
notebook P1 de Jeffrey para ejecutar los modelos P2 en el servidor real. Contiene
las posiciones iniciales, cambios programados y modo paso a paso. Utiliza el
kernel **RoboCup P2 (CUDA)**. No depende de un panel gráfico nuevo.

La ejecución reproducible reside en `src/`, sin depender del orden de celdas.
Para explorar curvas en Jupyter: cargar `informe/evaluaciones.csv` y los archivos
`episodes.monitor.csv` / `progress.csv` de cada `resultados/` con pandas.
`python -m robocup_rl.graficar` genera las figuras sin necesitar notebook.
