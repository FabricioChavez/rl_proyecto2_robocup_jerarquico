#!/usr/bin/env bash
# Pilar 3 (escalera de ablación, 3 semillas) y Pilar 2 (PBRS frente a shaping ingenuo).
#
#  Escalera en persecución, mismo presupuesto de pasos y defensor a 50:
#    tabular (P1) -> DQN, DDQN (valor) -> A2C, PPO (actor-crítico) -> PPO sin shaping
#    -> PPO + shaping ingenuo -> BC humano + PPO (warm-start + ajuste fino)
#  Reward hacking en conducción: desde el MISMO BC del experto, ajuste fino PPO con
#    shaping none / pbrs / naive.
#
# Es idempotente: salta las corridas que ya tienen final.pth (o episodios.csv en tabular).
# Uso:  bash scripts/escalera_ablacion.sh            (PY=python, 6 procesos en paralelo)
#       PY=python3 PARALELO=4 PASOS=30000 bash scripts/escalera_ablacion.sh
set -u
cd "$(dirname "$0")/.."
PY="${PY:-python}"
PARALELO="${PARALELO:-6}"
PASOS="${PASOS:-30000}"
DISPOSITIVO="${DISPOSITIVO:-cuda}"
W=resultados/modelo_weights
mkdir -p resultados/logs/escalera
puerto=8200
trabajos=()

encolar() {  # nombre, comprobación de hecho, comando...
  local nombre="$1" hecho="$2"; shift 2
  if [ -e "$hecho" ]; then echo "ya hecho: $nombre"; return; fi
  trabajos+=("$nombre|$(printf "%q " "$@")")
}

for s in 0 1 2; do
  comun="--semilla $s --pasos $PASOS --fuerza-defensor 50 --dispositivo $DISPOSITIVO"
  encolar "tabular_s$s" resultados/logs/persecucion/tabular/escalera_s$s/q_tabla.npy \
    "$PY" -X utf8 scripts/tabular_p1.py --semilla $s --pasos $PASOS --corrida escalera_s$s --puerto @P
  encolar "dqn_s$s" $W/persecucion/dqn/escalera_s$s/final.pth \
    "$PY" -X utf8 proyecto.py entrenar --tarea persecucion --algoritmo dqn --reglas recuperacion --corrida escalera_s$s $comun --puerto @P
  encolar "ddqn_s$s" $W/persecucion/dqn/escalera_ddqn_s$s/final.pth \
    "$PY" -X utf8 proyecto.py entrenar --tarea persecucion --algoritmo dqn --double --reglas recuperacion --corrida escalera_ddqn_s$s $comun --puerto @P
  encolar "a2c_s$s" $W/persecucion/a2c/escalera_s$s/final.pth \
    "$PY" -X utf8 proyecto.py entrenar --tarea persecucion --algoritmo a2c --reglas recuperacion --corrida escalera_s$s $comun --puerto @P
  encolar "ppo_s$s" $W/persecucion/ppo/escalera_s$s/final.pth \
    "$PY" -X utf8 proyecto.py entrenar --tarea persecucion --algoritmo ppo --reglas recuperacion --corrida escalera_s$s $comun --puerto @P
  encolar "ppo_none_s$s" $W/persecucion/ppo/escalera_none_s$s/final.pth \
    "$PY" -X utf8 proyecto.py entrenar --tarea persecucion --algoritmo ppo --reglas recuperacion --shaping none --corrida escalera_none_s$s $comun --puerto @P
  encolar "ppo_naive_s$s" $W/persecucion/ppo/escalera_naive_s$s/final.pth \
    "$PY" -X utf8 proyecto.py entrenar --tarea persecucion --algoritmo ppo --reglas recuperacion --shaping naive --corrida escalera_naive_s$s $comun --puerto @P
  encolar "bc_ppo_s$s" $W/persecucion/ppo/escalera_bc_s$s/final.pth \
    "$PY" -X utf8 proyecto.py entrenar --tarea persecucion --algoritmo ppo --continuar $W/persecucion/ppo/humano/bc.pth --corrida escalera_bc_s$s $comun --lr 0.0001 --puerto @P
  for sh in none pbrs naive; do
    encolar "hack_${sh}_s$s" $W/conduccion/ppo/hack_${sh}_s$s/final.pth \
      "$PY" -X utf8 proyecto.py entrenar --tarea conduccion --algoritmo ppo --continuar $W/conduccion/ppo/expert_bc/bc.pth --shaping $sh --corrida hack_${sh}_s$s $comun --lr 0.0001 --puerto @P
  done
done

echo "Corridas pendientes: ${#trabajos[@]} (paralelo $PARALELO)"
activos=0
for t in "${trabajos[@]}"; do
  nombre="${t%%|*}"; comando="${t#*|}"
  comando="${comando//@P/$puerto}"; puerto=$((puerto + 10))
  echo "[$(date +%H:%M:%S)] inicia $nombre"
  bash -c "$comando" > "resultados/logs/escalera/$nombre.log" 2>&1 &
  activos=$((activos + 1))
  if [ "$activos" -ge "$PARALELO" ]; then wait -n; activos=$((activos - 1)); fi
done
wait
echo "[$(date +%H:%M:%S)] escalera terminada. Gráficas: $PY scripts/graficas_pilares.py"
