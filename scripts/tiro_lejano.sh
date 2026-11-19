#!/usr/bin/env bash
# Tiro lejano con ventaja (defensor 50). Definición tiene 23 acciones: además de los tiros
# fijos, "avanzar al arco" y tiros al palo lejos del arquero. Conducción cede a definición
# desde x > X_VENTAJA si el defensor ya quedó detrás del balón (además de x > 36).
#   1. Fase 4: x 25-38 con el defensor superado (30 % repasa la zona de tiro), desde final2.
#   2. Corrida final4 y coordinador nuevo (experto con traspaso con ventaja: demos -> BC -> DAgger).
#   3. Definición entrenada dentro de esa jerarquía congelada (corrida final4b).
#   4. Evaluación de 200 episodios y diagnóstico con los traspasos grabados.
# Uso:  PY=python bash scripts/tiro_lejano.sh   (idempotente)
set -u
cd "$(dirname "$0")/.."
PY="${PY:-python}"
W=resultados/modelo_weights
L=resultados/logs
XV="${XV:-28}"
P4="${P4:-100000}"; PJ="${PJ:-60000}"
V="--x-ventaja $XV"
E="--semillas 0 --fuerza-defensor 50 50 --episodios 200 --reglas recuperacion --escenarios 2v1 --sin-graficas"
RL="--fuerza-defensor 50 --shaping pbrs --lr 0.0001 --entropia 0.003 --kl-objetivo 0.02 --dispositivo cuda"
paso() { echo; echo "[$(date +%H:%M:%S)] $*"; }

paso "1. Fase 4: lejos con ventaja ($P4 pasos)"
if [ ! -e $W/definicion/ppo/lejano_f4/final.pth ]; then
  "$PY" -X utf8 proyecto.py entrenar --tarea definicion --algoritmo ppo --continuar $W/definicion/ppo/final2/final.pth \
    --corrida lejano_f4 --curriculo-definicion 4 --pasos $P4 $RL --puerto 8910 || exit 1
fi
"$PY" -c "
import pandas as pd
d = pd.read_csv('$L/definicion/ppo/lejano_f4/episodios.csv'); n = len(d) // 5
print('gol por fase, primer 20 %:', d.head(n).groupby('fase').gol.mean().round(3).to_dict())
print('gol por fase, último 20 %:', d.tail(n).groupby('fase').gol.mean().round(3).to_dict())"

paso "2. Corrida final4 y coordinador con traspaso con ventaja (x > $XV)"
for t in persecucion conduccion definicion; do mkdir -p $W/$t/ppo/final4; done
cp $W/persecucion/ppo/final2/final.pth $W/persecucion/ppo/final4/final.pth
cp $W/conduccion/ppo/final2/final.pth $W/conduccion/ppo/final4/final.pth
cp $W/definicion/ppo/lejano_f4/final.pth $W/definicion/ppo/final4/final.pth
D=datos/demostraciones/experto_jerarquico_final4.csv
C="--tarea jerarquico --algoritmo ppo --corrida final4 --experto programado --fuerza-defensor 50 --sin-monitor --dispositivo cuda $V --datos $D"
if [ ! -e $W/jerarquico/ppo/final4/bc.pth ]; then
  "$PY" -X utf8 proyecto.py imitar demos $C --episodios 100 --puerto 8920
  "$PY" -X utf8 proyecto.py imitar bc $C --epocas-bc 200 --puerto 8920
  for r in 1 2 3; do
    for intento in 1 2; do  # el servidor a veces se cae mientras se reentrena la red
      "$PY" -X utf8 proyecto.py imitar dagger $C --rondas 1 --episodios 40 --epocas-bc 200 \
        --puerto $((8930 + 10 * r + intento)) && break
    done
  done
fi
"$PY" -X utf8 proyecto.py evaluar --tareas jerarquico --algoritmos ppo --corrida final4 $E $V --puerto 8970

paso "3. Definición dentro de la jerarquía congelada ($PJ pasos)"
if [ ! -e $W/definicion/ppo/lejano_jer/final.pth ]; then
  "$PY" -X utf8 proyecto.py entrenar --tarea definicion --algoritmo ppo --continuar $W/definicion/ppo/lejano_f4/final.pth \
    --corrida lejano_jer --coordinador $W/jerarquico/ppo/final4/final.pth \
    --habilidades $W/persecucion/ppo/final4/final.pth $W/conduccion/ppo/final4/final.pth $W/definicion/ppo/final4/final.pth \
    --pasos $PJ $RL $V --puerto 8980 || exit 1
fi
for t in persecucion conduccion jerarquico; do mkdir -p $W/$t/ppo/final4b; cp $W/$t/ppo/final4/final.pth $W/$t/ppo/final4b/final.pth; done
mkdir -p $W/definicion/ppo/final4b
cp $W/definicion/ppo/lejano_jer/final.pth $W/definicion/ppo/final4b/final.pth

paso "4. Evaluación (200 episodios, defensor 50)"
"$PY" -X utf8 proyecto.py evaluar --tareas jerarquico --algoritmos ppo --corrida final4b $E $V --puerto 8990
"$PY" -X utf8 proyecto.py evaluar --tareas definicion --algoritmos ppo --corrida lejano_f4 $E --puerto 9000
"$PY" -X utf8 proyecto.py evaluar --tareas definicion --algoritmos ppo --corrida lejano_jer $E --puerto 9010
for c in final4 final4b; do
  "$PY" -X utf8 scripts/diagnostico_jerarquia.py --corrida $c --meta red --x-tiro 36 --x-ventaja $XV --episodios 200 \
    --fuerza-defensor 50 --puerto 9020 --salida $L/traspasos/diagnostico_$c.csv --grabar-traspasos $L/traspasos/estados_$c.csv
done

paso "Resumen"
"$PY" - <<'PY'
import math
import pandas as pd
from pathlib import Path
L = Path("resultados/logs")
def wilson(k, n, z=1.96):
    p = k / n; c = (p + z * z / (2 * n)) / (1 + z * z / n)
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - m, c + m
filas = [
    ("final: sistema del informe", "jerarquico/ppo/final_ft/jerarquico/ppo/final"),
    ("final2: currículo de definición", "jerarquico/ppo/final2"),
    ("final4: fase 4 + traspaso con ventaja", "jerarquico/ppo/final4"),
    ("final4b: + definición dentro de la jerarquía", "jerarquico/ppo/final4b"),
    ("definición sola lejano_f4 (zona de tiro)", "definicion/ppo/lejano_f4"),
    ("definición sola lejano_jer (zona de tiro)", "definicion/ppo/lejano_jer"),
]
for nombre, ruta in filas:
    f = L / ruta / "evaluacion_2v1.csv"
    if f.exists():
        d = pd.read_csv(f); k, n = int(d.gol.sum()), len(d); lo, hi = wilson(k, n)
        print(f"  {nombre:46s} {k / n:6.1%}  IC {lo:.1%}-{hi:.1%}")
for c in ("final4", "final4b"):
    f = L / "traspasos" / f"estados_{c}.csv"
    if f.exists():
        x = pd.read_csv(f).a1_x
        print(f"  traspasos {c}: {len(x)}; x media {x.mean():.1f}; x < 36: {(x < 36).mean():.0%}")
PY
echo "Listo."
