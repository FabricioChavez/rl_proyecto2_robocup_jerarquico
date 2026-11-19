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
| Sistema jerárquico final (coordinador DAgger + PPO) | **47,0 % goles** (IC 95 % 40,2–53,9 %). La meta del 80 % no se alcanza |
| Primera jerarquía (coordinador DAgger humano) | 30,5 % |
| Política plana PPO, 200k pasos | 0 % |
| Escalera en persecución (3 semillas) | DQN 99,8 % · DDQN 97,0 % · PPO 98,7 % · A2C 72,0 % · BC+PPO 99,8 % |
| PBRS frente a shaping ingenuo (persecución) | 98,7 % frente a 40,5 % (sin shaping: 64,2 %) |
| Reward hacking (conducción) | shaping ingenuo: retorno visto 32 frente a 13, éxito 92,7 % frente a 99,2 % |
| Conducción por imitación | BC humano 13 % → DAgger humano 53,5 % → BC/DAgger experto 99,5/100 % |
| Definición | DAgger humano 58 % → + PPO desde la zona de tiro 86 % |

## 1. Ver las demos en Docker (sin instalar nada más)

```bash
docker compose build visual
docker compose run --rm --service-ports visual python proyecto.py demo jerarquico
```

Abre http://localhost:6080/vnc.html?autoconnect=true&resize=scale. El servidor, la política y
`rcssmonitor` corren dentro del contenedor. Cambia `jerarquico` por cualquier nombre de la lista:

```bash
docker compose run --rm visual python proyecto.py demo
```

| Demo | Qué muestra |
|---|---|
| `jerarquico` | Sistema final: coordinador DAgger + PPO (47 %) |
| `jerarquico_dagger`, `jerarquico_bc` | Coordinador solo DAgger (44,5 %) o solo BC (41 %) del experto programado |
| `jerarquico_humano` | Primera jerarquía, coordinador DAgger humano (30,5 %) |
| `jerarquico_ppo_rl`, `jerarquico_a2c_rl`, `jerarquico_dqn_rl` | Todo el sistema ajustado con PPO / A2C / DQN |
| `plana` | Política plana PPO sin jerarquía |
| `persecucion_bc`, `persecucion_dagger` | BC / DAgger humano |
| `persecucion_ppo`, `persecucion_a2c`, `persecucion_dqn`, `persecucion_ddqn` | Desde cero (escalera) |
| `persecucion_bc_rl`, `persecucion_{ppo,a2c,dqn}_ft` | BC/DAgger ajustados con RL |
| `conduccion_bc_humano`, `conduccion_dagger_humano` | BC / DAgger humano |
| `conduccion_bc`, `conduccion_dagger` | BC / DAgger del experto programado |
| `conduccion_bc_rl`, `conduccion_bc_ingenuo` | BC experto + PPO con PBRS / con shaping ingenuo |
| `conduccion_{ppo,a2c,dqn}_ft` | Imitación ajustada con PPO / A2C / DQN |
| `definicion_bc`, `definicion_dagger` | BC / DAgger humano |
| `definicion_dagger_rl`, `definicion_dqn_zona` | DAgger + PPO / Double DQN desde la zona de tiro |
| `definicion_{ppo,a2c,dqn}_ft` | Imitación ajustada con PPO / A2C / DQN |

Opciones: `--episodios N`. Las demos usan la condición de prueba: 2v1, reglas de recuperación y
defensor a potencia 50. El catálogo está en [src/demos.py](src/demos.py).
Para elegir cualquier otro `.pth`: `python proyecto.py ver --tarea ... --algoritmo ... --corrida ...`
(más ejemplos en [COMANDOS.md](COMANDOS.md)).

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

**e) Evaluación estándar de todas las variantes, figuras, informe y presentación:**

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
