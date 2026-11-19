# RoboCup 2D — P2

Una entrada: **`python proyecto.py <comando>`**. Modelos únicamente `.pth`.
Los parámetros se pasan por argumentos; no hay archivos de configuración de modelos.

**PPO, A2C y DQN están implementados directamente en PyTorch. No se importa ni se necesita Stable Baselines3.**
El código de cada actualización está en [PPO](src/modelo/arquitecturas/ppo.py),
[A2C](src/modelo/arquitecturas/a2c.py) y [DQN](src/modelo/arquitecturas/dqn.py).
La recolección de transiciones y el loop están en [entrenamiento.py](src/modelo/entrenamiento.py).
Consulta [la implementación de los métodos](src/modelo/README.md) para seguir las fórmulas hasta el código.

Los 45 modelos anteriores **sí fueron entrenados con SB3**. Se conservan en corridas `sb3_semilla_N`,
como referencia histórica. Las corridas nuevas `semilla_N` se entrenan desde cero con los métodos propios.
Los ensayos cortos de funcionamiento se conservaron fuera de los resultados de la campaña.

## Dónde está cada cosa

```text
proyecto.py                 Entrada única
src/
  modelo/
    arquitecturas/          PPO, A2C y DQN
    entrenamiento.py        Loop de entrenamiento y continuación
    pesos.py                Guardar/cargar checkpoints .pth
    imitacion.py            Demostraciones, BC y DAgger
  entorno/                  Física oficial, UDP y condiciones del escenario
  evaluador/
    evaluar.py              Lotes aleatorios SIN ventana → CSV
    visualizar.py           Ejecutar un .pth en rcssmonitor oficial
    graficas.py             CSV → gráficas PNG
    informe.py              Volcar resultados a informe/
resultados/
  modelo_weights/           Pesos .pth por tarea/algoritmo/semilla
  logs/                     Episodios, actualizaciones y evaluaciones CSV
  graficas/                 Aprendizaje, pérdidas y comparación de algoritmos
informe/                    Volcado para revisar/compartir resultados
GUIA.md                     Instrucciones completas, paso a paso
```

**[Índice de modelos y CSV disponibles](resultados/README.md)** · **[Guía paso a paso](GUIA.md)** · **[Mapa del código](src/README.md)**

## Comandos

Con las dependencias y el servidor instalados, desde la raíz:

```bash
python proyecto.py estado
python proyecto.py entrenar --tarea persecucion --algoritmo ppo --pasos 10000 --gamma 0.99 --lr 0.0003 --dispositivo cuda --corrida prueba
python proyecto.py evaluar --tareas persecucion --algoritmos ppo --semillas 0 --episodios 200 --escenarios 1v1 2v1 --dispositivo cpu
python proyecto.py ver --tarea persecucion --algoritmo ppo --dispositivo cpu
python proyecto.py graficar
python proyecto.py informe
```

El modelo de persecución/PPO/semilla 0 está en:
**[resultados/modelo_weights/persecucion/ppo/semilla_0/final.pth](resultados/modelo_weights/persecucion/ppo/semilla_0/final.pth)**.
Sus datos están en `resultados/logs/persecucion/ppo/semilla_0/`.
Mientras esa corrida está entrenando, el archivo se llama `checkpoint.pth`; `final.pth` aparece al terminar.
`python proyecto.py estado` lista los pesos terminados. Los logs de la campaña propia están en
`resultados/logs/campana/torch_stdout.log` y `torch_stderr.log`.

## Docker

```bash
docker compose build proyecto
docker compose run --rm proyecto graficar
docker compose run --rm proyecto informe
docker compose up visual
```

Para visualizar, abre tú [el monitor oficial en el navegador](http://localhost:6080/vnc.html?autoconnect=true&resize=scale).
La configuración base usa CPU; CUDA usa además `docker-compose.gpu.yml`.
La etapa base y las descargas APT por HTTPS se validaron en Docker; la construcción completa sigue pendiente.
Los loops de entrenamiento/continuación, evaluación sin ventana y carga de PTH sí se comprobaron con el servidor real.
