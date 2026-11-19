# RoboCup 2D jerárquico: RL profundo, PBRS e imitación

Ataque 2v1 en el simulador oficial `rcssserver`. Tres habilidades (persecución, conducción y definición)
y un coordinador que elige entre ellas (Vía A). PPO, A2C, DQN/Double DQN, BC y DAgger están escritos
en PyTorch dentro del repositorio; no se usa Stable Baselines3 ni RLlib. Una sola entrada:
`python proyecto.py <comando>`; pesos solo `.pth`; hiperparámetros por argumentos.

| Entregable | Archivo |
|---|---|
| Informe IEEE (4 págs.) | [informe/informe.pdf](informe/informe.pdf) · fuente [informe/informe.tex](informe/informe.tex) |
| Presentación | [informe/presentacion.pptx](informe/presentacion.pptx) · generador [scripts/presentacion.js](scripts/presentacion.js) |
| Figuras y tablas | [informe/figuras/](informe/figuras/) |

## Resultados (200 episodios independientes, defensor a potencia 50)

| Qué | Resultado |
|---|---|
| **Mejor sistema actual** (`final4b`): tiro lejano con ventaja + definición entrenada dentro de la jerarquía | **65,0 % goles** (IC 95 % 58,2–71,3 %). La meta del 80 % aún no se alcanza |
| Definición con currículo inverso + coordinador DAgger nuevo (`final2`) | 51,5 % (IC 44,6–58,3 %) |
| Sistema del informe (coordinador DAgger + PPO) | 47,0 % (IC 40,2–53,9 %) |
| Primera jerarquía (coordinador DAgger humano) | 30,5 % |
| Política plana PPO, 200k pasos | 0 % |
| Escalera en persecución (3 semillas) | DQN 99,8 % · DDQN 97,0 % · PPO 98,7 % · A2C 72,0 % · BC+PPO 99,8 % |
| PBRS frente a shaping ingenuo (persecución) | 98,7 % frente a 40,5 % (sin shaping: 64,2 %) |
| Reward hacking (conducción) | shaping ingenuo: retorno visto 32 frente a 13, éxito 92,7 % frente a 99,2 % |
| Conducción por imitación | BC humano 13 % → DAgger humano 53,5 % → BC/DAgger experto 99,5/100 % |
| Definición | DAgger humano 58 % → + PPO desde la zona de tiro 86 % |

## 1. Ver las demos en Docker (Windows, Mac o Ubuntu; sin instalar nada más)

Una vez, construir la imagen (también después de cada `git pull` que cambie código):

```bash
docker compose build visual
```

Lanzar una demo y abrir http://localhost:6080/vnc.html?autoconnect=true&resize=scale en el navegador.
El servidor, las políticas y `rcssmonitor` corren dentro del contenedor (no uses el monitor de WSL:
en Windows dibuja la cancha corrupta).

```bash
docker compose run --rm --service-ports --name robocup_demo visual python proyecto.py demo jerarquico
```

**Detenerla:** cierra la ventana del monitor dentro del noVNC, `Ctrl+C` en la terminal, o desde otra terminal:

```bash
docker stop robocup_demo
```

`docker compose down` no detiene contenedores lanzados con `docker compose run`; por eso el `--name`.
Opciones: `--episodios N` al final. Todas usan la condición de prueba: 2v1, reglas de recuperación y
defensor a potencia 50. En Ubuntu, si tu usuario no está en el grupo `docker`, antepon `sudo`.

| Demo | Qué muestra | Comando |
|---|---|---|
| `jerarquico` | Mejor sistema: tiro lejano con ventaja + definición entrenada dentro de la jerarquía (65,0 % goles) | `docker compose run --rm --service-ports --name robocup_demo visual python proyecto.py demo jerarquico` |
| `jerarquico_curriculo` | Definición con currículo inverso + coordinador DAgger nuevo (51,5 %) | `docker compose run --rm --service-ports --name robocup_demo visual python proyecto.py demo jerarquico_curriculo` |
| `jerarquico_informe` | Sistema del informe: coordinador DAgger + ajuste fino PPO (47,0 %) | `docker compose run --rm --service-ports --name robocup_demo visual python proyecto.py demo jerarquico_informe` |
| `jerarquico_dagger` | Coordinador DAgger del experto programado (44,5 %) | `docker compose run --rm --service-ports --name robocup_demo visual python proyecto.py demo jerarquico_dagger` |
| `jerarquico_bc` | Coordinador solo BC del experto programado (41,0 %) | `docker compose run --rm --service-ports --name robocup_demo visual python proyecto.py demo jerarquico_bc` |
| `jerarquico_humano` | Primera versión: coordinador DAgger humano (30,5 %) | `docker compose run --rm --service-ports --name robocup_demo visual python proyecto.py demo jerarquico_humano` |
| `jerarquico_ppo_rl` | Jerarquía con todo ajustado por PPO (26,5 %) | `docker compose run --rm --service-ports --name robocup_demo visual python proyecto.py demo jerarquico_ppo_rl` |
| `jerarquico_a2c_rl` | Jerarquía con todo ajustado por A2C (16,0 %) | `docker compose run --rm --service-ports --name robocup_demo visual python proyecto.py demo jerarquico_a2c_rl` |
| `jerarquico_dqn_rl` | Jerarquía con todo ajustado por Double DQN (1,0 %) | `docker compose run --rm --service-ports --name robocup_demo visual python proyecto.py demo jerarquico_dqn_rl` |
| `plana` | Línea base: política plana PPO sin jerarquía | `docker compose run --rm --service-ports --name robocup_demo visual python proyecto.py demo plana` |
| `persecucion_bc` | BC humano | `docker compose run --rm --service-ports --name robocup_demo visual python proyecto.py demo persecucion_bc` |
| `persecucion_dagger` | DAgger humano | `docker compose run --rm --service-ports --name robocup_demo visual python proyecto.py demo persecucion_dagger` |
| `persecucion_bc_rl` | BC humano + PPO (escalera) | `docker compose run --rm --service-ports --name robocup_demo visual python proyecto.py demo persecucion_bc_rl` |
| `persecucion_ppo` | PPO desde cero | `docker compose run --rm --service-ports --name robocup_demo visual python proyecto.py demo persecucion_ppo` |
| `persecucion_a2c` | A2C desde cero | `docker compose run --rm --service-ports --name robocup_demo visual python proyecto.py demo persecucion_a2c` |
| `persecucion_dqn` | DQN desde cero | `docker compose run --rm --service-ports --name robocup_demo visual python proyecto.py demo persecucion_dqn` |
| `persecucion_ddqn` | Double DQN desde cero | `docker compose run --rm --service-ports --name robocup_demo visual python proyecto.py demo persecucion_ddqn` |
| `persecucion_ppo_ft` | Imitación + ajuste PPO | `docker compose run --rm --service-ports --name robocup_demo visual python proyecto.py demo persecucion_ppo_ft` |
| `persecucion_a2c_ft` | Imitación + ajuste A2C | `docker compose run --rm --service-ports --name robocup_demo visual python proyecto.py demo persecucion_a2c_ft` |
| `persecucion_dqn_ft` | Imitación + ajuste DQN | `docker compose run --rm --service-ports --name robocup_demo visual python proyecto.py demo persecucion_dqn_ft` |
| `conduccion_bc_humano` | BC humano | `docker compose run --rm --service-ports --name robocup_demo visual python proyecto.py demo conduccion_bc_humano` |
| `conduccion_dagger_humano` | DAgger humano | `docker compose run --rm --service-ports --name robocup_demo visual python proyecto.py demo conduccion_dagger_humano` |
| `conduccion_bc` | BC del experto programado | `docker compose run --rm --service-ports --name robocup_demo visual python proyecto.py demo conduccion_bc` |
| `conduccion_dagger` | DAgger del experto programado | `docker compose run --rm --service-ports --name robocup_demo visual python proyecto.py demo conduccion_dagger` |
| `conduccion_bc_rl` | BC experto + PPO con PBRS | `docker compose run --rm --service-ports --name robocup_demo visual python proyecto.py demo conduccion_bc_rl` |
| `conduccion_bc_ingenuo` | BC experto + PPO con shaping ingenuo | `docker compose run --rm --service-ports --name robocup_demo visual python proyecto.py demo conduccion_bc_ingenuo` |
| `conduccion_ppo_ft` | Imitación + ajuste PPO | `docker compose run --rm --service-ports --name robocup_demo visual python proyecto.py demo conduccion_ppo_ft` |
| `conduccion_a2c_ft` | Imitación + ajuste A2C | `docker compose run --rm --service-ports --name robocup_demo visual python proyecto.py demo conduccion_a2c_ft` |
| `conduccion_dqn_ft` | Imitación + ajuste DQN | `docker compose run --rm --service-ports --name robocup_demo visual python proyecto.py demo conduccion_dqn_ft` |
| `definicion_bc` | BC humano | `docker compose run --rm --service-ports --name robocup_demo visual python proyecto.py demo definicion_bc` |
| `definicion_dagger` | DAgger humano | `docker compose run --rm --service-ports --name robocup_demo visual python proyecto.py demo definicion_dagger` |
| `definicion_dagger_rl` | DAgger humano + PPO desde la zona de tiro (habilidad final) | `docker compose run --rm --service-ports --name robocup_demo visual python proyecto.py demo definicion_dagger_rl` |
| `definicion_lejano` | Definición de 23 acciones entrenada dentro de la jerarquía (la de 'jerarquico') | `docker compose run --rm --service-ports --name robocup_demo visual python proyecto.py demo definicion_lejano` |
| `definicion_dqn_zona` | DAgger humano + Double DQN desde la zona de tiro | `docker compose run --rm --service-ports --name robocup_demo visual python proyecto.py demo definicion_dqn_zona` |
| `definicion_ppo_ft` | Imitación + ajuste PPO | `docker compose run --rm --service-ports --name robocup_demo visual python proyecto.py demo definicion_ppo_ft` |
| `definicion_a2c_ft` | Imitación + ajuste A2C | `docker compose run --rm --service-ports --name robocup_demo visual python proyecto.py demo definicion_a2c_ft` |
| `definicion_dqn_ft` | Imitación + ajuste DQN | `docker compose run --rm --service-ports --name robocup_demo visual python proyecto.py demo definicion_dqn_ft` |

**Cualquier otro modelo** (cualquier `.pth` de `resultados/modelo_weights/<tarea>/<algoritmo>/<corrida>/`):

```bash
docker compose run --rm --service-ports --name robocup_demo visual python proyecto.py ver --tarea jerarquico --algoritmo ppo --corrida final4b --x-ventaja 28 --escenario 2v1 --reglas recuperacion --fuerza-defensor 50 --sin-teclado
```

Cambia `--tarea` (persecucion, conduccion, definicion, jerarquico, plana), `--algoritmo` (ppo, a2c, dqn)
y `--corrida`, o pasa `--pesos ruta/al/archivo.pth`. `--x-ventaja 28` solo para las corridas `final4*`
(traspaso a definición desde x > 28 con el defensor superado). El catálogo está en
[src/demos.py](src/demos.py); más ejemplos en [COMANDOS.md](COMANDOS.md).

## 2. Reproducir todo (local, con `rcssserver` y GPU)

Instalación del servidor y del entorno: [GUIA.md](GUIA.md). Cada paso es independiente y escribe
CSV en `resultados/logs/` y pesos en `resultados/modelo_weights/<tarea>/<algoritmo>/<corrida>/`.

**a) Habilidades por imitación.** Las demostraciones humanas se graban con el teclado (ver GUIA).
Conducción usa el experto programado:

```bash
python proyecto.py imitar demos  --tarea persecucion --algoritmo ppo --corrida humano
python proyecto.py imitar bc     --tarea persecucion --algoritmo ppo --corrida humano
python proyecto.py imitar dagger --tarea persecucion --algoritmo ppo --corrida humano --rondas 3
python proyecto.py imitar demos  --tarea conduccion  --algoritmo ppo --corrida expert_dagger --experto programado --episodios 100 --sin-monitor
python proyecto.py imitar bc     --tarea conduccion  --algoritmo ppo --corrida expert_dagger --experto programado
python proyecto.py imitar dagger --tarea conduccion  --algoritmo ppo --corrida expert_dagger --experto programado --rondas 3 --sin-monitor
```

(definición, igual que persecución, con `--tarea definicion`).

**b) Ajuste fino con RL de cada fase** (PPO, A2C y DQN), desde los pesos de imitación:

```powershell
.\scripts\ajuste_fino.ps1 -Algoritmo ppo
.\scripts\ajuste_fino.ps1 -Algoritmo a2c
.\scripts\ajuste_fino.ps1 -Algoritmo dqn
```

**c) Pilares 2 y 3: escalera de ablación y reward hacking** (3 semillas × 8 variantes en
persecución + 3 shapings en conducción, en paralelo; unos 15 minutos con GPU):

```bash
PARALELO=8 bash scripts/escalera_ablacion.sh
bash scripts/evaluar_escalera.sh
python scripts/graficas_pilares.py
```

**d) Sistema jerárquico final** (definición desde la zona de tiro, coordinador BC → DAgger → PPO):

```bash
bash scripts/armar_final.sh
python scripts/diagnostico_jerarquia.py --corrida final --meta red --episodios 200 --fuerza-defensor 50
```

**e) Mejoras posteriores al informe** (definición): currículo inverso (`final2`) y tiro lejano con
ventaja + definición entrenada dentro de la jerarquía congelada (`final4b`, el mejor):

```bash
bash scripts/curriculo_definicion.sh
bash scripts/tiro_lejano.sh
```

`tiro_lejano.sh` amplía definición a 23 acciones (avanzar al arco y tiros al palo lejos del arquero),
entrena la fase 4 (x 25-38 con el defensor superado), reimita el coordinador con `--x-ventaja 28` y
entrena definición con `--coordinador` (el resto del sistema juega congelado). Los sistemas `final4*`
se evalúan y se ven siempre con `--x-ventaja 28`.

**f) Evaluación estándar de todas las variantes, figuras, informe y presentación:**

```bash
bash scripts/evaluar_final.sh
python scripts/graficas_imitacion.py
python scripts/graficas_final.py
cd informe && pdflatex informe.tex && pdflatex informe.tex && cd ..
node scripts/presentacion.js
```

`evaluar_final.sh` aplica un único protocolo: 200 episodios independientes (semillas de prueba 10000+),
política determinista, sin actualizar pesos, 2v1, defensor a potencia 50. Las tasas llevan IC de Wilson al 95 %.
`node scripts/presentacion.js` requiere `pptxgenjs` (`npm install pptxgenjs`).

## Dónde está cada cosa

```text
proyecto.py                 Entrada única (entrenar, imitar, evaluar, ver, demo, graficar, ...)
src/modelo/arquitecturas/   PPO, A2C y DQN/Double DQN en PyTorch
src/modelo/entrenamiento.py Loop de entrenamiento y continuación (--continuar)
src/modelo/imitacion.py     Demostraciones, BC, DAgger y expertos programados
src/entorno/robocup.py      MDP: estado, recompensa, PBRS, opciones con terminación beta(s)
src/evaluador/              Evaluación sin ventana y visualización en rcssmonitor
src/demos.py                Catálogo de demos
scripts/                    Escalera, evaluación final, diagnóstico, gráficas y presentación
resultados/modelo_weights/  Pesos .pth por tarea/algoritmo/corrida
resultados/logs/            Episodios, actualizaciones y evaluaciones (CSV)
informe/                    Informe, presentación y figuras
```

Los modelos `sb3_semilla_N` son referencias históricas entrenadas con SB3 y no forman parte de
los resultados. Más detalle: [GUIA.md](GUIA.md) · [COMANDOS.md](COMANDOS.md) · [docker/README.md](docker/README.md).
