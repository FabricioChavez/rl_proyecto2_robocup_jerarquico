#!/usr/bin/env bash
# Evalúa con el protocolo estándar (200 episodios independientes, semillas de prueba
# 10000+, política determinista, defensor a potencia 50) cada corrida de la escalera.
set -u
cd "$(dirname "$0")/.."
PY="${PY:-python}"
PARALELO="${PARALELO:-6}"
E="--semillas 0 --fuerza-defensor 50 50 --episodios 200 --reglas recuperacion --escenarios 2v1 --sin-graficas"
puerto=9000; activos=0
lanzar() {  # tarea algoritmo corrida
  local f="resultados/logs/$1/$2/$3/evaluacion_2v1.csv"
  [ -e "$f" ] && return
  "$PY" -X utf8 proyecto.py evaluar --tareas "$1" --algoritmos "$2" --corrida "$3" $E --puerto $puerto \
    > "resultados/logs/escalera/eval_$1_$2_$3.log" 2>&1 &
  puerto=$((puerto + 10)); activos=$((activos + 1))
  if [ "$activos" -ge "$PARALELO" ]; then wait -n; activos=$((activos - 1)); fi
}
for s in 0 1 2; do
  lanzar persecucion dqn escalera_s$s
  lanzar persecucion dqn escalera_ddqn_s$s
  lanzar persecucion a2c escalera_s$s
  for c in escalera escalera_none escalera_naive escalera_bc; do lanzar persecucion ppo ${c}_s$s; done
  for sh in none pbrs naive; do lanzar conduccion ppo hack_${sh}_s$s; done
done
wait
echo "evaluación de la escalera terminada"
