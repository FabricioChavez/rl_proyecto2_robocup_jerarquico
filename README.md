# RoboCup 2D — Proyecto P2, Vía A

Tres habilidades autónomas: **persecución/intercepción**, **conducción/drible/pase** y
**definición**. Cada una permite **DQN, A2C y PPO**, PBRS y mejora posterior con
**BC o DAgger de una experta humana**. El meta-controlador y la política plana
se entrenan por separado. El simulador es el servidor oficial **rcssserver 19**:
no se sustituye su física por una simulación casera.

Repositorio público: <https://github.com/FabricioChavez/rl_proyecto2_robocup_jerarquico>

## 1. Estado y alcance

- Implementados el entorno real, las tres subtareas, DQN/PPO/A2C, jerarquía,
  baseline MC, evaluación, PBRS, BC/DAgger y registro de experimentos.
- El presupuesto inicial es **10 000 pasos por subtarea/algoritmo/semilla**.
  Son pasos, no episodios. La campaña usa semillas 0, 1 y 2.
- Para el meta-controlador se usan **1 200 decisiones de 8 ciclos**: como máximo
  9 600 ciclos físicos. Las opciones terminan antes si finaliza el episodio.
- Las demostraciones humanas quedan pendientes para Emilia. No se generan
  etiquetas humanas artificiales. Los ensayos automáticos de imitación usan
  un controlador **heurístico**, identificado en cada muestra.
- **No se afirma haber alcanzado 80% de goles ni superioridad estadística.**
  Las pruebas cortas verifican ejecución y actualizaciones, no convergencia.
- El estado se obtiene de la percepción global del trainer y se transforma a
  distancias/ángulos relativos: es **observación privilegiada**. No se ha resuelto
  localización ni visión parcial ruidosa. Las patadas, carreras, inercia,
  estamina y paradas del arquero se ejecutan en el servidor real.
- El informe IEEE final y las conclusiones experimentales quedan pendientes.
  La lista completa está en [informe/REQUISITOS.md](informe/REQUISITOS.md).

## 2. Organización (estilo Lab 4, separada por habilidad y algoritmo)

```text
subtareas/
  persecucion/{ppo,a2c,dqn,bc,dagger}/
  conduccion/{ppo,a2c,dqn,bc,dagger}/
  definicion/{ppo,a2c,dqn,bc,dagger}/
global/
  jerarquico/{ppo,a2c,dqn}/
  plana/{ppo,a2c,dqn}/
src/robocup_rl/
  comun/           UDP, acciones, MDP, servidor, pausa para experta humana
  algoritmos.py    configuración compartida de DQN/A2C/PPO
  entrenar.py      entrenamiento con servidor externo
  ejecutar.py      entrenamiento + servidor administrado automáticamente
  campana.py       cola reproducible de experimentos y evaluación
  evaluar.py       episodios independientes, retorno base, goles, IC Wilson
  imitacion.py     captura humana/heurística, BC, DAgger
  baseline.py      MC first-visit con discretización 4x5 inspirada en P1
  comparar.py      bootstrap de diferencias de tasa de gol
  graficar.py      curvas agrupadas por semilla
docker/            imagen CUDA + servidor oficial fijado a un commit
scripts/           instalación del servidor, verificación y contraejemplo PBRS
notebooks/         espacio de exploración opcional
data/humanos/      demostraciones locales (no se publican por defecto)
informe/           teoría, checklist, resultados y figuras
runs/              estado de la cola, logs y pruebas de integración
```

La lógica compartida permanece en `src/`: no hay nueve copias del bucle de
entrenamiento. Cada carpeta de algoritmo contiene su configuración y su espacio
de resultados. `config.json` es descriptivo; los parámetros efectivos se pasan
por CLI y se guardan nuevamente junto al modelo.

## 3. Instalación en esta PC (PowerShell + WSL Ubuntu)

**Ya está preparado localmente** el entorno `.venv` y el binario `rcssserver`
en Ubuntu WSL. Para trabajar abre **PowerShell**, no la consola Python:

```powershell
cd D:\2026_2_CS\RL\PROYECTO\rl_robocup_jerarquico
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

En otra máquina con Windows + NVIDIA + WSL2 Ubuntu:

```powershell
git clone https://github.com/FabricioChavez/rl_proyecto2_robocup_jerarquico.git
cd rl_proyecto2_robocup_jerarquico
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -e . pytest
wsl -d Ubuntu -- bash scripts/setup_server.sh
```

El último comando solicita la contraseña de `sudo` dentro de WSL si es necesaria.
Los comandos administrados usan `wsl -d Ubuntu -u root` y `systemd-run` para
arrancar un **servidor exclusivo por puerto**, cerrándolo al terminar. Si tu
distribución tiene otro nombre, define `$env:ROBOCUP_WSL_DISTRO='Nombre'`.
Si se bloquean paquetes UDP entre Windows y WSL, ejecuta el mismo proyecto en
Linux o usa Docker. No ejecutes dos trabajos en el mismo puerto.

CUDA es obligatoria por defecto: la aplicación falla si no está disponible.
SB3 puede advertir que una MLP pequeña corre más rápido en CPU; aquí se respeta
la elección de CUDA y se verifica que los parámetros estén en la GPU.

## 4. Entrenar una habilidad (un comando)

Desde PowerShell en la raíz:

```powershell
.\.venv\Scripts\python.exe -m robocup_rl.ejecutar --task persecucion --algo ppo --steps 10000 --seed 0 --port 6100
.\.venv\Scripts\python.exe -m robocup_rl.ejecutar --task conduccion --algo a2c --steps 10000 --seed 0 --port 6100
.\.venv\Scripts\python.exe -m robocup_rl.ejecutar --task definicion --algo dqn --steps 10000 --seed 0 --port 6100
```

También: `./entrenar.ps1 -Subtarea persecucion -Algoritmo ppo -Pasos 10000`.
Para probar el escenario sin compañero añade `--scenario 1v1`; el predeterminado
es `2v1`. En ambos hay un defensor y un arquero activos. El compañero de 2v1
tiene un controlador heurístico de apoyo/tiro, y puede recibir pases; no es un
segundo agente neuronal aprendido.

Un entrenamiento individual guarda por defecto en
`runs/<task>/<algo>/<shaping>/<scenario>/seed_<n>/`.
La campaña organizada guarda en las carpetas `resultados/` descritas abajo.
Usa `--output ruta` para elegir un destino. Para repetir sin sobrescribir, usa
otra salida; el programa rechaza reemplazar un `final.zip` existente.

Cada corrida guarda:

- `config.json`: hiperparámetros, GPU, versiones y esquema de observación/acción.
- `episodes.monitor.csv`: retornos, duración y éxito por episodio.
- `progress.csv`: pérdidas, estadísticas y actualizaciones de SB3.
- `checkpoint.zip`: política recuperable, cada 1 000 decisiones.
- `final.zip`: modelo terminado; `status.json`: estado real de la corrida.

PPO exige un múltiplo de 200 pasos y A2C de 100 para respetar el presupuesto
sin excederlo al completar un rollout. DQN empieza a actualizar tras 500 pasos.
Las pruebas DQN deben superar ese umbral.

## 5. Campaña completa y proceso en segundo plano

```powershell
.\.venv\Scripts\python.exe -u -m robocup_rl.campana --steps 10000 --seeds 0 1 2 --algorithms ppo a2c dqn --scenario 2v1 --evaluate 200 --port 6200
```

El orden es: por semilla/algoritmo, entrenar persecución, conducción, definición,
después jerarquía con esas tres políticas congeladas y finalmente política plana.
Al terminar el bloque de entrenamientos, cada modelo se evalúa en 200 episodios independientes. Son **45 corridas**;
10 000 es el límite de cada entrenamiento, no de toda la campaña.
El proceso falla visiblemente ante un error y deja un registro de la causa.
Repetir el comando omite modelos/evaluaciones ya terminados.

En esta PC, la campaña iniciada por Codex se registra en:

```powershell
Get-Content runs\background.json
Get-Content runs\campaign_status.json
Get-Content runs\training.stdout.log -Tail 20
Get-Content runs\training.stderr.log -Tail 10
```

Para detenerla y sus procesos descendientes, utiliza el PID de `background.json`:

```powershell
$run = Get-Content runs\background.json | ConvertFrom-Json
taskkill /PID $run.pid /T /F
wsl -d Ubuntu -u root -- systemctl stop robocup-emilia-6200
```

Si la detienes forzosamente, se conserva el último checkpoint pero no el replay
buffer ni el RNG exacto. Recuperación aproximada de una corrida:

```powershell
.\.venv\Scripts\python.exe -m robocup_rl.ejecutar --task persecucion --algo ppo --resume RUTA\checkpoint.zip --steps 2000 --output runs\recuperacion --port 6100
```

Repite la campaña con `--scenario 1v1` para completar la otra condición exigida.
Para investigar shaping añade `--ablations`: ejecuta `none`, `pbrs` y `naive` en
las tres habilidades con las mismas semillas. La rama humana se agrega después.

## 6. Probar y visualizar

Elige un modelo terminado. Por ejemplo, tras la campaña:

```powershell
$M='subtareas/persecucion/ppo/resultados/pbrs/2v1/seed_0/final.zip'
.\.venv\Scripts\python.exe -m robocup_rl.evaluar --task persecucion --algo ppo --model $M --episodes 3 --render
```

Para entrenar explícitamente el meta-controlador:

```powershell
$P='subtareas/persecucion/ppo/resultados/pbrs/2v1/seed_0/final.zip'
$C='subtareas/conduccion/ppo/resultados/pbrs/2v1/seed_0/final.zip'
$D='subtareas/definicion/ppo/resultados/pbrs/2v1/seed_0/final.zip'
.\.venv\Scripts\python.exe -m robocup_rl.ejecutar --task global --algo ppo --steps 1200 --skills $P $C $D --skill-algo ppo --output global/jerarquico/ppo/resultados/manual
.\.venv\Scripts\python.exe -m robocup_rl.evaluar --task global --algo ppo --model global/jerarquico/ppo/resultados/manual/final.zip --skills $P $C $D --episodes 3 --render
```

El meta-controlador **selecciona una habilidad**, no sus acciones primitivas.
Mantiene la habilidad durante 8 ciclos, vuelve a observar y decide de nuevo.
No sustituye silenciosamente modelos ausentes por heurísticas.
La política plana elige entre las 13 acciones primitivas y se entrena con
`--task plana` sobre los mismos escenarios y recompensa global.

Evaluación final, heurística y comparación:

```powershell
.\.venv\Scripts\python.exe -m robocup_rl.evaluar --task plana --episodes 200 --seed 10000 --output informe/heuristica_2v1.json
.\.venv\Scripts\python.exe -m robocup_rl.comparar RUTA/evaluacion_jerarquica.json informe/heuristica_2v1.json
.\.venv\Scripts\python.exe -m robocup_rl.graficar
```

`evaluation.json` registra cada episodio, tasa de éxito, **tasa de gol separada**,
retorno de recompensa base e intervalo Wilson. Capturar el balón no cuenta como
gol. Comparar también con la evaluación de la política plana. Un resultado
aislado no demuestra superioridad sobre tres semillas. El simulador upstream
19 inicializa su RNG físico por tiempo y no expone `server::random_seed` en la
CLI: las semillas de NumPy/PyTorch y reinicios sí se controlan; no se promete
reproducción bit a bit ni emparejamiento exacto de ruido físico entre políticas.

## 7. BC con tus demostraciones humanas (al volver)

La interacción se hace **paso a paso**: ves el campo y escribes el número de
acción en PowerShell. Mientras eliges, el servidor y su watchdog se congelan;
no hace falta contestar rápido. `q` guarda los datos disponibles y sale.

```powershell
.\.venv\Scripts\python.exe -m robocup_rl.imitacion collect --task persecucion --expert human --episodes 5 --dataset data/humanos/persecucion.npz
.\.venv\Scripts\python.exe -m robocup_rl.imitacion bc --task persecucion --dataset data/humanos/persecucion.npz --epochs 30 --output subtareas/persecucion/bc/resultados/policy
```

BC minimiza entropía cruzada sobre la **misma política PPO/A2C**. Puedes iniciar
desde un modelo RL añadiendo `--model RUTA/final.zip --algo ppo`. Para BC desde
cero omite `--model`. El crítico todavía debe aprender durante el ajuste con RL.
Repite cambiando la subtarea por `conduccion` y `definicion`; cada una tiene su
dataset/esquema de acciones propio. `--scenario` debe coincidir con los datos.

Ajuste fino tras BC:

```powershell
.\.venv\Scripts\python.exe -m robocup_rl.ejecutar --task persecucion --algo ppo --resume subtareas/persecucion/bc/resultados/policy.zip --steps 10000 --output subtareas/persecucion/bc/resultados/finetuning
```

## 8. DAgger con experta humana

```powershell
.\.venv\Scripts\python.exe -m robocup_rl.imitacion dagger --task persecucion --expert human --model subtareas/persecucion/bc/resultados/policy.zip --dataset data/humanos/persecucion.npz --rounds 3 --episodes 5 --beta 0.2 --output subtareas/persecucion/dagger/resultados/policy
```

La aprendiz visita estados con su política y solicita tu etiqueta **antes** de
actuar. En DAgger ejecuta la etiqueta experta con probabilidad beta y la acción
de la aprendiz en los demás casos. Agrega los datos de todas las rondas y vuelve
a ajustar la política. DAgger requiere `--model`; no se anuncia como DAgger una
simple recolección de demostraciones. Para ensayos sin persona puedes usar
`--expert heuristic`, siempre guardado como origen heurístico.

## 9. Baseline P1, PBRS y verificaciones

```powershell
.\.venv\Scripts\python.exe -m robocup_rl.baseline --steps 10000 --seed 0 --output runs/baseline/seed_0
.\.venv\Scripts\python.exe scripts/pbrs_ablation.py
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts/smoke.py
```

El baseline conserva la idea de P1 (MC first-visit, 4 distancias x 5 orientaciones,
4 acciones), pero se **reentrena** en el MDP oficial. No se transfieren cifras de
su simulador cinemático como si fueran resultados de este entorno. La prueba
algebraica PBRS muestra un bucle que recompensa `naive` sin acercarse al objetivo;
las ablaciones reales requieren `campana --ablations`.

La justificación matemática está en [informe/METODOLOGIA.md](informe/METODOLOGIA.md).
El resumen verificable de pruebas está en [informe/VERIFICACION.json](informe/VERIFICACION.json).
`scripts/smoke.py` prueba tres algoritmos por habilidad y tres meta-controladores;
en DQN usa 600 pasos para incluir actualizaciones después del warm-up de 500.

## 10. Docker: un comando alternativo

Con Docker Desktop abierto, backend WSL2 y GPU NVIDIA habilitada:

```powershell
docker compose up --build
```

Incluye el servidor oficial y la campaña CUDA en la misma imagen, con resultados
montados localmente. En esta sesión Docker Desktop estaba detenido; el camino
probado es **PowerShell + WSL**, y la imagen Docker queda sin validación de build.

## Referencias y procedencia

- [Lab 4 de FabricioChavez](https://github.com/FabricioChavez/rl_lab4_final_fabricio):
  organización modular, código común, algoritmos y resultados separados.
- [Proyecto 1 del grupo 02](https://github.com/jeffreymonjacastro/grupo-02-proyecto-rl):
  notebook de persecución MC y protocolo UDP. Su informe contiene plantilla,
  por lo que no se asumen resultados experimentales que no estén acreditados.
- Enunciado P2 del curso DS5345 2026-II, páginas 1–5, y slides S10A/S11A/S13B
  proporcionados localmente; no se republican esos PDFs en este repositorio.
- [Servidor oficial](https://github.com/rcsoccersim/rcssserver), commit
  `ce870013f2c6b31b9e93774abb2822c2f346c287`.
- [Manual oficial del servidor](https://rcsoccersim.readthedocs.io/en/latest/soccerserver.html)
  y [protocolo del entrenador](https://rcsoccersim.readthedocs.io/en/latest/coach.html).
- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/en/master/):
  implementación de DQN, PPO y A2C; se usan sus pérdidas y checkpoints originales.

La implementación Python de este proyecto es nueva; los repositorios anteriores
se mantienen fuera del repositorio publicado como referencias de lectura.
