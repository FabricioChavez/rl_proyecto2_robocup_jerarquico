# Tabla de comandos

Ejecuta los comandos desde la raíz del repositorio, con Docker iniciado.
Sirven en PowerShell y en terminales Linux/macOS, sin alias. La imagen base usa CPU.
En esta PC la imagen CPU ya está construida. En otra PC empieza por la primera tabla.

## Preparar el proyecto en otra PC

| Qué hacer | Comando |
|---|---|
| Clonar el repositorio | `git clone https://github.com/FabricioChavez/rl_proyecto2_robocup_jerarquico.git` |
| Entrar en el clon | `cd rl_proyecto2_robocup_jerarquico` |
| Construir la imagen CPU | `docker build -t robocup-p2:cpu -f docker/Dockerfile .` |
| Consultar pesos disponibles | `docker compose run --rm proyecto estado` |

## Visualizar un modelo

**Para cambiar de modelo:** termina la ejecución anterior con **Ctrl+C**, ejecuta
`docker compose down` y después elige una fila. No tienes que reconstruir la imagen.

Abre o recarga manualmente [el monitor oficial](http://localhost:6080/vnc.html?autoconnect=true&resize=scale).
El modelo se elige con los argumentos de la terminal.

| Modelo que quieres ver | Comando |
|---|---|
| Predeterminado: jerarquía PPO, semilla 0 | `docker compose up visual` |
| Jerarquía PPO | `docker compose run --rm --service-ports visual python proyecto.py ver --tarea jerarquico --algoritmo ppo --semilla 0 --sin-teclado` |
| Jerarquía A2C | `docker compose run --rm --service-ports visual python proyecto.py ver --tarea jerarquico --algoritmo a2c --semilla 0 --sin-teclado` |
| Jerarquía DQN | `docker compose run --rm --service-ports visual python proyecto.py ver --tarea jerarquico --algoritmo dqn --semilla 0 --sin-teclado` |
| Persecución PPO | `docker compose run --rm --service-ports visual python proyecto.py ver --tarea persecucion --algoritmo ppo --semilla 0 --sin-teclado` |
| Conducción PPO | `docker compose run --rm --service-ports visual python proyecto.py ver --tarea conduccion --algoritmo ppo --semilla 0 --sin-teclado` |
| Definición PPO | `docker compose run --rm --service-ports visual python proyecto.py ver --tarea definicion --algoritmo ppo --semilla 0 --sin-teclado` |
| Política plana PPO | `docker compose run --rm --service-ports visual python proyecto.py ver --tarea plana --algoritmo ppo --semilla 0 --sin-teclado` |
| Detener los servicios de Compose | `docker compose down` |

En cualquier fila puedes cambiar `--algoritmo ppo` por `--algoritmo a2c` o
`--algoritmo dqn`. Las cinco tareas × tres algoritmos de **semilla 0** están publicadas.
Para terminar automáticamente, añade `--episodios 10` al comando `ver`.

## Cambiar las condiciones del escenario

Estos argumentos se añaden al comando `ver`; se aplican al iniciar cada episodio.
Para cambiar argumentos, detén la ejecución anterior y vuelve a iniciarla.

| Qué cambia | Argumento de ejemplo |
|---|---|
| Escenario | `--escenario 1v1` o `--escenario 2v1` |
| Posición inicial del balón | `--balon 20 0` |
| Posición inicial del atacante | `--atacante 15 0` |
| Velocidad inicial del balón | `--velocidad-balon 0.5 0` |
| Conducta del defensor | `--conducta-defensor patrulla` (también `perseguir` o `quieto`) |
| Fuerza del defensor | `--fuerza-defensor 60` (0 a 100) |
| Arquero quieto | `--arquero-quieto` |

Ejemplo completo:

```bash
docker compose run --rm --service-ports visual python proyecto.py ver --tarea persecucion --algoritmo dqn --semilla 0 --escenario 2v1 --balon 20 0 --atacante 15 0 --conducta-defensor patrulla --fuerza-defensor 60 --sin-teclado
```

## Evaluar, graficar y exportar

Estos comandos no abren ventanas ni entrenan el modelo.

| Qué hacer | Comando |
|---|---|
| Evaluar jerarquía PPO, 20 episodios por escenario | `docker compose run --rm proyecto evaluar --tareas jerarquico --algoritmos ppo --semillas 0 --escenarios 1v1 2v1 --episodios 20 --dispositivo cpu` |
| Evaluar los 15 modelos propios de semilla 0 con condiciones aleatorias | `docker compose run --rm proyecto evaluar --tareas persecucion conduccion definicion jerarquico plana --algoritmos ppo a2c dqn --semillas 0 --escenarios 1v1 2v1 --episodios 20 --velocidad-balon 0 0.5 --fuerza-defensor 50 100 --dispositivo cpu` |
| Generar gráficas desde los CSV | `docker compose run --rm proyecto graficar` |
| Volcar gráficas y CSV a informe | `docker compose run --rm proyecto informe` |
| Consultar todos los argumentos del evaluador | `docker compose run --rm proyecto evaluar --help` |

Para una evaluación más extensa cambia `--episodios 20` por `--episodios 200`.
Una evaluación reemplaza el CSV de la misma corrida y escenario; `--salida otra_evaluacion`
permite conservarla en otra carpeta. En `evaluar`, velocidad y fuerza son rangos aleatorios;
en `ver`, la velocidad es un vector y la fuerza es un valor fijo.

## Entrenar y continuar en CPU

Usa una `--corrida` nueva para conservar los pesos existentes. `--pasos` cuenta
decisiones; al continuar indica cuántas decisiones **adicionales** ejecutar.

| Qué hacer | Comando |
|---|---|
| Entrenar persecución PPO desde cero | `docker compose run --rm proyecto entrenar --tarea persecucion --algoritmo ppo --pasos 10000 --gamma 0.99 --lr 0.0003 --red 128 128 --dispositivo cpu --corrida prueba_cpu` |
| Añadir 2000 decisiones a esa corrida | `docker compose run --rm proyecto entrenar --tarea persecucion --algoritmo ppo --pasos 2000 --gamma 0.99 --lr 0.0003 --dispositivo cpu --corrida prueba_cpu --continuar resultados/modelo_weights/persecucion/ppo/prueba_cpu/final.pth` |
| Continuar una corrida interrumpida desde su checkpoint | `docker compose run --rm proyecto entrenar --tarea persecucion --algoritmo ppo --pasos 2000 --dispositivo cpu --corrida prueba_cpu --continuar resultados/modelo_weights/persecucion/ppo/prueba_cpu/checkpoint.pth` |
| Consultar hiperparámetros disponibles | `docker compose run --rm proyecto entrenar --help` |

La fila de `checkpoint.pth` se usa si ese archivo existe tras una interrupción.
Al terminar normalmente queda `final.pth` y se retira `checkpoint.pth`.
Para A2C o DQN cambia `--algoritmo` y el nombre de `--corrida`; al continuar deben coincidir con el PTH.

## Entrenar con NVIDIA/CUDA

Requiere una GPU NVIDIA accesible desde Docker. Esta variante no se ha validado
con una construcción completa; las pruebas Docker realizadas corresponden a CPU.

| Qué hacer | Comando |
|---|---|
| Construir la imagen CUDA | `docker compose -f docker-compose.yml -f docker-compose.gpu.yml build proyecto` |
| Entrenar persecución PPO con CUDA | `docker compose -f docker-compose.yml -f docker-compose.gpu.yml run --rm proyecto entrenar --tarea persecucion --algoritmo ppo --pasos 10000 --dispositivo cuda --corrida prueba_cuda` |
| Continuar esa corrida | `docker compose -f docker-compose.yml -f docker-compose.gpu.yml run --rm proyecto entrenar --tarea persecucion --algoritmo ppo --pasos 2000 --dispositivo cuda --corrida prueba_cuda --continuar resultados/modelo_weights/persecucion/ppo/prueba_cuda/final.pth` |

## Dónde quedan las salidas

| Contenido | Ruta dentro del repositorio |
|---|---|
| Pesos | `resultados/modelo_weights/TAREA/ALGORITMO/CORRIDA/final.pth` |
| Checkpoint durante el entrenamiento | `resultados/modelo_weights/TAREA/ALGORITMO/CORRIDA/checkpoint.pth` |
| Episodios, pérdidas y evaluaciones | `resultados/logs/TAREA/ALGORITMO/CORRIDA/` |
| Gráficas de los métodos propios | `resultados/graficas/torch_propio/` |
| Volcado del informe | `informe/` |

Los resultados persisten en la carpeta del repositorio mediante volúmenes de Docker.
Sin `--corrida`, la carpeta se llama `semilla_N`. Las corridas `sb3_*` son históricas.

Para jerarquía con habilidades personalizadas, campaña completa o BC/DAgger, consulta
[GUIA.md](GUIA.md). Los datos humanos BC/DAgger siguen pendientes.
