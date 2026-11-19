#!/usr/bin/env bash
# Reconstruye el sistema jerárquico final (Vía A) desde las habilidades ya entrenadas:
#   1. definición: ajuste fino PPO desde la zona de tiro (distribución real de traspasos)
#   2. corrida 'final' = persecución BC humano + conducción DAgger experto + definición zona_tiro
#   3. coordinador: demos del experto programado -> BC -> 3 rondas de DAgger -> PPO
# Todo con defensor a potencia 50. Uso:  PY=python bash scripts/armar_final.sh
set -eu
cd "$(dirname "$0")/.."
PY="${PY:-python}"
W=resultados/modelo_weights
if [ ! -e $W/definicion/ppo/zona_tiro/final.pth ]; then
  "$PY" -X utf8 proyecto.py entrenar --tarea definicion --algoritmo ppo \
    --continuar $W/definicion/ppo/sistema_ft/final.pth --corrida zona_tiro --pasos 60000 \
    --fuerza-defensor 50 --shaping pbrs --lr 0.0001 --entropia 0.001 --kl-objetivo 0.02 \
    --dispositivo cuda --puerto 7800
fi
for t in persecucion conduccion definicion; do mkdir -p $W/$t/ppo/final; done
cp $W/persecucion/ppo/humano/bc.pth $W/persecucion/ppo/final/final.pth
cp $W/conduccion/ppo/expert_dagger/final.pth $W/conduccion/ppo/final/final.pth
cp $W/definicion/ppo/zona_tiro/final.pth $W/definicion/ppo/final/final.pth
C="--tarea jerarquico --algoritmo ppo --corrida final --experto programado --fuerza-defensor 50 --sin-monitor --dispositivo cuda"
"$PY" -X utf8 proyecto.py imitar demos $C --episodios 100 --puerto 7300
"$PY" -X utf8 proyecto.py imitar bc $C --epocas-bc 200 --puerto 7300
# Una ronda por invocación: el servidor puede cerrarse mientras se reentrena la red.
for r in 1 2 3; do
  "$PY" -X utf8 proyecto.py imitar dagger $C --rondas 1 --episodios 40 --epocas-bc 200 --puerto $((7300 + 50 * r))
done
"$PY" -X utf8 proyecto.py entrenar --tarea jerarquico --algoritmo ppo \
  --continuar $W/jerarquico/ppo/final/final.pth --corrida final_ft \
  --habilidades $W/persecucion/ppo/final/final.pth $W/conduccion/ppo/final/final.pth $W/definicion/ppo/final/final.pth \
  --fuerza-defensor 50 --pasos 15000 --lr 0.0001 --entropia 0.001 --kl-objetivo 0.02 \
  --shaping pbrs --reglas recuperacion --dispositivo cuda --puerto 7700
echo "Sistema final listo. Evalúa con: bash scripts/evaluar_final.sh"
