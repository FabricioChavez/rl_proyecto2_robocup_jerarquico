#!/usr/bin/env bash
# Currículo inverso para definición y nuevo sistema jerárquico (defensor a potencia 50).
#   1. Graba estados reales de traspaso a definición jugando el sistema 'final'.
#   2. PPO por fases desde definicion/ppo/zona_tiro (cada fase repasa un 30 % las anteriores):
#        fase 1: x 25-45, y ±15, orientación ±90°, balón al pie
#        fase 2: balón vivo (0,5-1,5 m, hasta 1,5 m/ciclo) y defensor delante/costado/detrás
#        fase 3: 50 % traspasos reales + 50 % fase 2
#   3. Corrida 'final2' = persecución y conducción de 'final' + definición de la fase 3.
#   4. Evalúa con el coordinador viejo, reimita el coordinador (BC -> DAgger -> PPO) y reevalúa.
# Uso:  PY=python bash scripts/curriculo_definicion.sh   (idempotente: salta lo ya hecho)
set -u
cd "$(dirname "$0")/.."
PY="${PY:-python}"
W=resultados/modelo_weights
L=resultados/logs
P1="${P1:-40000}"; P2="${P2:-50000}"; P3="${P3:-60000}"
BANCO=$L/traspasos/estados.csv
E="--semillas 0 --fuerza-defensor 50 50 --episodios 200 --reglas recuperacion --escenarios 2v1 --sin-graficas"
RL="--fuerza-defensor 50 --shaping pbrs --lr 0.0001 --entropia 0.003 --kl-objetivo 0.02 --dispositivo cuda"
paso() { echo; echo "[$(date +%H:%M:%S)] $*"; }

paso "1. Grabar traspasos reales"
if [ ! -s $BANCO ]; then
  "$PY" -X utf8 scripts/diagnostico_jerarquia.py --corrida final --meta red --x-tiro 36 --episodios 300 \
    --fuerza-defensor 50 --puerto 8600 --grabar-traspasos $BANCO \
    --salida $L/traspasos/diagnostico_final.csv
fi

paso "2. Definición por fases"
previo=$W/definicion/ppo/zona_tiro/final.pth
for f in 1 2 3; do
  eval pasos=\$P$f
  destino=$W/definicion/ppo/curriculo_f$f/final.pth
  if [ ! -e $destino ]; then
    "$PY" -X utf8 proyecto.py entrenar --tarea definicion --algoritmo ppo --continuar $previo \
      --corrida curriculo_f$f --curriculo-definicion $f --traspasos $BANCO --pasos $pasos $RL --puerto 861$f \
      || { echo "falló la fase $f"; exit 1; }
  fi
  "$PY" -c "
import pandas as pd
d = pd.read_csv('$L/definicion/ppo/curriculo_f$f/episodios.csv'); u = d.tail(max(1, len(d) // 5))
print('fase $f: gol en el último 20 % de episodios por fase:', u.groupby('fase').gol.mean().round(3).to_dict())"
  previo=$destino
done

paso "3. Corrida final2"
for t in persecucion conduccion definicion; do mkdir -p $W/$t/ppo/final2; done
cp $W/persecucion/ppo/final/final.pth $W/persecucion/ppo/final2/final.pth
cp $W/conduccion/ppo/final/final.pth $W/conduccion/ppo/final2/final.pth
cp $previo $W/definicion/ppo/final2/final.pth

paso "4a. Evaluar: definición sola (zona de tiro) y jerarquía con el coordinador viejo"
"$PY" -X utf8 proyecto.py evaluar --tareas definicion --algoritmos ppo --corrida curriculo_f3 $E --puerto 8620
"$PY" -X utf8 proyecto.py evaluar --tareas jerarquico --algoritmos ppo --corrida final2 $E --puerto 8630   --pesos $W/jerarquico/ppo/final_ft/final.pth --salida $L/jerarquico/ppo/final2_coord_viejo

paso "4b. Reimitar el coordinador sobre las nuevas habilidades"
C="--tarea jerarquico --algoritmo ppo --corrida final2 --experto programado --fuerza-defensor 50 --sin-monitor --dispositivo cuda"
if [ ! -e $W/jerarquico/ppo/final2/bc.pth ]; then
  "$PY" -X utf8 proyecto.py imitar demos $C --episodios 100 --puerto 8640 \
    --datos datos/demostraciones/experto_jerarquico_final2.csv
  "$PY" -X utf8 proyecto.py imitar bc $C --epocas-bc 200 --puerto 8640 \
    --datos datos/demostraciones/experto_jerarquico_final2.csv
  for r in 1 2 3; do
    for intento in 1 2; do  # el servidor a veces se cae mientras se reentrena la red
      "$PY" -X utf8 proyecto.py imitar dagger $C --rondas 1 --episodios 40 --epocas-bc 200 \
        --datos datos/demostraciones/experto_jerarquico_final2.csv --puerto $((8650 + 10 * r + intento)) && break
    done
  done
fi
if [ ! -e $W/jerarquico/ppo/final2_ft/final.pth ]; then
  "$PY" -X utf8 proyecto.py entrenar --tarea jerarquico --algoritmo ppo \
    --continuar $W/jerarquico/ppo/final2/final.pth --corrida final2_ft \
    --habilidades $W/persecucion/ppo/final2/final.pth $W/conduccion/ppo/final2/final.pth $W/definicion/ppo/final2/final.pth \
    --fuerza-defensor 50 --pasos 15000 --lr 0.0001 --entropia 0.001 --kl-objetivo 0.02 \
    --shaping pbrs --reglas recuperacion --dispositivo cuda --puerto 8690
fi

paso "5. Evaluar la nueva jerarquía (200 episodios)"
"$PY" -X utf8 proyecto.py evaluar --tareas jerarquico --algoritmos ppo --corrida final2 $E --puerto 8700
"$PY" -X utf8 proyecto.py evaluar --tareas jerarquico --algoritmos ppo --corrida final2 $E --puerto 8710 \
  --pesos $W/jerarquico/ppo/final2_ft/final.pth --salida $L/jerarquico/ppo/final2_ft
"$PY" -X utf8 scripts/diagnostico_jerarquia.py --corrida final2 --meta red --x-tiro 36 --episodios 200 \
  --fuerza-defensor 50 --puerto 8720 --salida $L/traspasos/diagnostico_final2.csv

paso "Resumen (goles en 200 episodios, defensor 50)"
"$PY" - <<'PY'
import pandas as pd
from pathlib import Path
L = Path("resultados/logs")
filas = [
    ("Antes: sistema final (DAgger + PPO)", "jerarquico/ppo/final_ft/jerarquico/ppo/final"),
    ("Definición currículo, coordinador viejo", "jerarquico/ppo/final2_coord_viejo/jerarquico/ppo/final2"),
    ("Definición currículo, coordinador DAgger nuevo", "jerarquico/ppo/final2"),
    ("Definición currículo, coordinador DAgger + PPO", "jerarquico/ppo/final2_ft/jerarquico/ppo/final2"),
    ("Definición sola (curriculo_f3, zona de tiro)", "definicion/ppo/curriculo_f3"),
]
for nombre, ruta in filas:
    f = L / ruta / "evaluacion_2v1.csv"
    print(f"  {nombre:50s} " + (f"{pd.read_csv(f).gol.mean():6.1%}" if f.exists() else "   (sin CSV)"))
PY
echo "Listo."
