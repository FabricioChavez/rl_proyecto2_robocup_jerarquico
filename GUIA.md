# Guía de uso — paso a paso

## 1. Ubicarse y elegir cómo ejecutar

Esta copia está en `D:\2026_2_CS\RL\PROYECTO\rl_robocup_jerarquico`.
Todos los comandos se ejecutan desde esa carpeta, no desde `src/`.

La entrada única es `proyecto.py`. Usa `python proyecto.py --help` para ver los comandos,
y por ejemplo `python proyecto.py entrenar --help` para ver todos sus argumentos.

**Docker, en Windows/Linux/macOS:** inicia tú Docker y ejecuta:

```bash
docker compose build proyecto
docker compose run --rm proyecto estado
```

En los ejemplos siguientes, sustituye `python proyecto.py` por
`docker compose run --rm proyecto` para ejecutar dentro del contenedor.
La variante base usa CPU. Para CUDA con NVIDIA se anteponen ambos archivos:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml build proyecto
docker compose -f docker-compose.yml -f docker-compose.gpu.yml run --rm proyecto estado
```

**Python local, alternativa:** las dependencias se instalan con `python -m pip install -e .`.
En esta PC ya existe `.venv`. En PowerShell puedes ejecutar directamente:

```powershell
.\.venv\Scripts\python.exe proyecto.py estado
```

En Linux/macOS se utiliza `.venv/bin/python proyecto.py ...`.
La ejecución local de entrenar/evaluar/ver requiere `rcssserver` instalado; ver también requiere `rcssmonitor`.
Docker incluye ambos. Windows local utiliza los binarios de Ubuntu WSL que ya están instalados en esta PC.

## 2. Qué código hace qué

| Acción | Código que la implementa | Resultado |
|---|---|---|
| Elegir argumentos | `src/cli.py` | Despacha al módulo correspondiente |
| Crear PPO/A2C/DQN | `src/modelo/arquitecturas/` | Red y algoritmo con argumentos del comando |
| Entrenar/continuar | `src/modelo/entrenamiento.py` | PTH + CSV de episodios y pérdidas |
| Guardar/cargar pesos | `src/modelo/pesos.py` | Checkpoint PyTorch, sin ZIP |
| Evaluar sin ventana | `src/evaluador/evaluar.py` | CSV por episodio/escenario |
| Ver una política | `src/evaluador/visualizar.py` | Monitor oficial rcssmonitor |
| Generar gráficas | `src/evaluador/graficas.py` | PNG y tabla consolidada |
| Volcar el informe | `src/evaluador/informe.py` | Copia PNG/CSV e índice a `informe/` |
| BC/DAgger | `src/modelo/imitacion.py` | Demostraciones CSV y pesos PTH |
| Física/acciones/UDP | `src/entorno/` | Comunicación con el servidor oficial |

## 3. Ubicar los pesos y resultados existentes

Los 45 modelos originales se entrenaron con Stable Baselines3: son referencias historicas,
no resultados de los algoritmos propios. Sus carpetas se llaman `sb3_semilla_N`.
Las nuevas corridas `semilla_N` se entrenan desde cero en PyTorch.
Los ensayos cortos de funcionamiento se conservaron fuera de los resultados de la campana.
Tareas: `persecucion`, `conduccion`, `definicion`, `jerarquico`, `plana`.
Algoritmos: `ppo`, `a2c`, `dqn`. Semillas: `0`, `1`, `2`.

```text
resultados/modelo_weights/persecucion/ppo/semilla_0/final.pth        # propio, al finalizar
resultados/modelo_weights/persecucion/ppo/semilla_0/checkpoint.pth   # propio, mientras entrena
resultados/logs/persecucion/ppo/semilla_0/episodios.csv
resultados/logs/persecucion/ppo/semilla_0/actualizaciones.csv
resultados/graficas/torch_propio/persecucion/ppo/aprendizaje.png
resultados/modelo_weights/persecucion/ppo/sb3_semilla_0/final.pth    # historico
resultados/graficas/historico_sb3/persecucion/ppo/aprendizaje.png
```

El índice navegable está en [resultados/README.md](resultados/README.md).
`python proyecto.py estado` muestra los pesos y las evaluaciones disponibles.
Los modelos existentes se entrenaron en 2v1: probarlos en 1v1 mide generalización, no los vuelve a entrenar.

## 4. Entrenar una subtarea nueva

```bash
python proyecto.py entrenar --tarea persecucion --algoritmo ppo --pasos 10000 --gamma 0.99 --lr 0.0003 --red 128 128 --rollout 200 --batch 100 --dispositivo cuda --corrida prueba_ppo
```

La salida queda en:

```text
resultados/modelo_weights/persecucion/ppo/prueba_ppo/final.pth
resultados/logs/persecucion/ppo/prueba_ppo/episodios.csv
resultados/logs/persecucion/ppo/prueba_ppo/actualizaciones.csv
```

Cambia tarea/algoritmo para conducción, definición, A2C o DQN. Usa `--dispositivo cpu` para CPU.
`--corrida` permite conservar los modelos ya existentes. El programa rechaza sobrescribir un `final.pth`
si no se está usando `--continuar`.

PPO requiere pasos múltiplos de rollout y batch que divida rollout. A2C requiere pasos múltiplos de rollout.
Valores predeterminados: PPO rollout 200/batch 100; A2C rollout 100; DQN batch 128 y warmup 500.
Los presupuestos son decisiones, no episodios. En la jerarquía, una decisión mantiene una habilidad
por `--pasos-opcion` ciclos, salvo que termine el episodio.

Todos los hiperparámetros disponibles se consultan con `entrenar --help`: gamma, lr, red, GAE,
entropía, clipping, replay, warmup, exploración, recompensas, shaping y frecuencia del checkpoint.
No se leen configuraciones JSON.

## 5. Continuar desde un checkpoint

```bash
python proyecto.py entrenar --tarea persecucion --algoritmo ppo --pasos 2000 --gamma 0.99 --lr 0.0003 --corrida prueba_ppo --continuar resultados/modelo_weights/persecucion/ppo/prueba_ppo/final.pth --dispositivo cuda
```

Añade 2 000 decisiones al contador anterior. Se conserva el CSV existente de esa corrida.
La arquitectura se infiere de los pesos; si pasas `--red`, debe coincidir.
Vuelve a especificar los hiperparámetros que quieras utilizar: los argumentos controlan el entrenamiento.

El PTH contiene política, optimizador, contador y, en entrenamientos nuevos DQN, replay.
El CSV es el historial para las gráficas: por sí solo no permite reanudar.
El episodio físico se reinicia; no es una reproducción bit a bit del mundo.
La continuacion RL acepta solo PTH de los metodos propios. Los historicos SB3 se conservan para inferencia;
no se usan para iniciar la nueva campana. En BC/DAgger tambien se usan corridas propias.

Mientras entrena se reemplaza un único `checkpoint.pth` cada `--checkpoint-cada` decisiones.
Al terminar, queda `final.pth` y se retira el checkpoint intermedio.
Si interrumpes una corrida, usa su `checkpoint.pth`. `--checkpoint-cada 0` guarda solo al finalizar.

## 6. Entrenar la jerarquía

Primero deben existir las tres habilidades del mismo algoritmo y semilla:

```bash
python proyecto.py entrenar --tarea jerarquico --algoritmo ppo --semilla 0 --pasos 1200 --pasos-opcion 8 --dispositivo cuda --corrida jerarquia_nueva
```

Carga automáticamente persecución, conducción y definición de `semilla_0`.
Para usar otros pesos pasa `--habilidades PERSECUCION.pth CONDUCCION.pth DEFINICION.pth`.
La política plana se entrena con `--tarea plana`. La campaña completa es `python proyecto.py campana --help`.

## 7. Evaluar por lotes, sin abrir ventanas

Prueba rápida:

```bash
python proyecto.py evaluar --tareas persecucion --algoritmos ppo --semillas 0 --escenarios 1v1 2v1 --episodios 20 --dispositivo cpu
```

Lote de todos los modelos y varios escenarios aleatorios:

```bash
python proyecto.py evaluar --tareas persecucion conduccion definicion jerarquico plana --algoritmos ppo a2c dqn --semillas 0 1 2 --escenarios 1v1 2v1 --episodios 200 --velocidad-balon 0 0.5 --fuerza-defensor 50 100 --dispositivo cpu
```

La posición/orientación inicial y las condiciones indicadas se muestrean por episodio.
`--semilla-prueba` controla las condiciones iniciales; el RNG físico del servidor upstream usa el reloj.
200 episodios por modelo/escenario puede tardar bastante; la prueba rápida solo comprueba funcionamiento.
Los parámetros no cambian ni entrenan la red.

Cada CSV registra retorno, éxito, gol, duración, semilla y condiciones muestreadas.
Se escribe episodio a episodio. Al finalizar el lote se generan las gráficas; `--sin-graficas` conserva solo CSV.
Se guarda como `resultados/logs/TAREA/ALGORITMO/CORRIDA/evaluacion_1v1.csv` o `evaluacion_2v1.csv`.
Una evaluación nueva reemplaza el CSV del mismo escenario. `--salida` permite guardarla en otra carpeta.
Para un peso específico usa `--pesos ARCHIVO.pth` junto a una única tarea/algoritmo/semilla.

## 8. Visualizar con el monitor oficial

Docker:

```bash
docker compose up --build visual
```

Abre tú [el monitor](http://localhost:6080/vnc.html?autoconnect=true&resize=scale).
Es rcssmonitor dentro del contenedor, transmitido por noVNC.
Para otra política, detén la sesión anterior y ejecuta:

```bash
docker compose run --rm --service-ports visual python proyecto.py ver --tarea definicion --algoritmo dqn --sin-teclado
```

Ejecución local:

```bash
python proyecto.py ver --tarea definicion --algoritmo dqn --dispositivo cpu
```

Clic derecho → `Drop Ball`; menú `Referee` para modos de juego.
En la terminal local: `p` pausa/reanuda, `n` avanza en pausa, `r` reinicia y `q` termina.
En Linux se confirma la letra con Enter. En el escritorio remoto esas teclas no se envían al agente.
`--balon X Y`, `--atacante X Y`, `--velocidad-balon VX VY`, `--conducta-defensor patrulla`
y `--fuerza-defensor 60` modifican la demostración sin archivos de configuración.
`--pesos ARCHIVO.pth` elige otro modelo. `--grabar` guarda una partida oficial RCG; es opcional.

## 9. Generar gráficas e informe

```bash
python proyecto.py graficar
python proyecto.py informe
```

Equivalente en Docker:

```bash
docker compose run --rm proyecto graficar
docker compose run --rm proyecto informe
```

Las gráficas muestran aprendizaje, pérdidas, retorno de evaluación, éxito, goles y duración.
Estan en `resultados/graficas/torch_propio/`; la tabla es `resultados/logs/resumen_evaluaciones_torch_propio.csv`.
Para las referencias antiguas usa `python proyecto.py graficar --historico`: escribe en `historico_sb3/`
y `resumen_evaluaciones_historico_sb3.csv`. Las curvas de ambas procedencias se mantienen separadas.
El retorno es la recompensa descontada; éxito de persecución/conducción no significa gol.
Las bandas entre semillas se dibujan cuando hay al menos tres corridas.

`informe` regenera gráficas y copia **todos los CSV y PNG** a `informe/logs/`, `informe/tablas/`
y `informe/graficas/`. Crea `informe/README.md` con enlaces. Los PTH se enlazan, no se duplican.
No inicia entrenamiento ni abre ventanas.

## 10. BC y DAgger, cuando recojas demostraciones

```bash
python proyecto.py imitar demos --tarea persecucion --algoritmo ppo --datos datos/demostraciones/persecucion.csv --experto humano --episodios 5
python proyecto.py imitar bc --tarea persecucion --algoritmo ppo --datos datos/demostraciones/persecucion.csv --epocas-bc 30
python proyecto.py imitar dagger --tarea persecucion --algoritmo ppo --datos datos/demostraciones/persecucion.csv --continuar resultados/modelo_weights/persecucion/bc/semilla_0/final.pth --rondas 3 --beta 0.2
```

Los datos humanos siguen pendientes. `--experto heuristico` es solo para ensayos automáticos y queda identificado
como tal en el CSV. BC/DAgger usan PPO/A2C; sus PTH quedan bajo carpetas `bc/` o `dagger/`.
Para ajuste fino con RL, usa `entrenar --continuar` con ese PTH y el algoritmo base correspondiente.

## 11. Implementacion y validacion

PPO, A2C y DQN son implementaciones propias con `torch.nn` y autograd. No requieren Stable Baselines3.
Consulta [las formulas y sus archivos](src/modelo/README.md). GAE, replay, clipping, objetivo TD,
recoleccion de transiciones y loops de optimizacion estan en el repositorio.

`python -m pytest -q` comprueba las formulas y los checkpoints: 14 pruebas.
Los tres metodos completaron ensayos cortos con CUDA contra rcssserver real y se evaluaron sin ventanas.
Estas comprobaciones no demuestran convergencia ni buena tasa de gol.
Los 45 PTH historicos conservan las salidas numericas de los modelos anteriores.
La etapa base Docker y las descargas Debian por HTTPS se validaron. La construccion completa sigue pendiente.

## 12. Campana propia y progreso

La campana nueva entrena desde cero las tres habilidades, la jerarquia y la politica plana
para PPO/A2C/DQN y semillas 0/1/2, con CUDA. Cada habilidad/plana tiene 10000 decisiones;
la jerarquia tiene 1200 opciones de hasta 8 ciclos. No son 10000 episodios.

```bash
python proyecto.py campana --pasos 10000 --pasos-jerarquia 1200 --semillas 0 1 2 --algoritmos ppo a2c dqn --dispositivo cuda
```

En esta PC se inicio en segundo plano. Para consultar su salida en PowerShell:

```powershell
Get-Content resultados/logs/campana/torch_stdout.log -Tail 15
Get-Content resultados/logs/campana/torch_stderr.log -Tail 15
Get-Process -Id (Get-Content resultados/logs/campana/torch.pid)
```

`estado` lista modelos terminados. Una carpeta con `checkpoint.pth` corresponde a una corrida
que empezo y puede estar activa o interrumpida: consulta el proceso y el log para distinguirlo.
No lances otra campana simultanea sobre las mismas carpetas. Si se interrumpe, continua ese
checkpoint con `entrenar --continuar ...` y las decisiones restantes; luego `campana` omite los
`final.pth` existentes. La evaluacion completa se ejecuta aparte con los comandos de la seccion 7.

Para inspeccionar una referencia antigua, pasa `--corrida sb3_semilla_0 --semillas 0` al evaluar,
o `--corrida sb3_semilla_0 --semilla 0` al visualizar. No se seleccionan historicos automaticamente.
