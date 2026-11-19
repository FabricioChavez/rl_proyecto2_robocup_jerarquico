#!/usr/bin/env bash
# Protocolo único de prueba para TODAS las variantes del informe: 200 episodios
# independientes (semillas de prueba 10000+), política determinista, 2v1, reglas de
# recuperación, defensor a potencia 50. Sobrescribe evaluacion_2v1.csv de cada corrida.
set -u
cd "$(dirname "$0")/.."
PY="${PY:-python}"
PARALELO="${PARALELO:-6}"
E="--semillas 0 --fuerza-defensor 50 50 --episodios 200 --reglas recuperacion --escenarios 2v1 --sin-graficas"
puerto=9500; activos=0
mkdir -p resultados/logs/evaluacion_final
lanzar() {  # tarea algoritmo corrida [args extra]
  local t="$1" a="$2" c="$3"; shift 3
  "$PY" -X utf8 proyecto.py evaluar --tareas "$t" --algoritmos "$a" --corrida "$c" $E "$@" --puerto $puerto \
    > "resultados/logs/evaluacion_final/${t}_${a}_${c}.log" 2>&1 &
  puerto=$((puerto + 10)); activos=$((activos + 1))
  if [ "$activos" -ge "$PARALELO" ]; then wait -n; activos=$((activos - 1)); fi
}
for c in humano_bc humano; do lanzar persecucion ppo $c; done
for c in human_bc human_dagger expert_bc expert_dagger; do lanzar conduccion ppo $c; done
for c in humano_bc humano zona_tiro; do lanzar definicion ppo $c; done
lanzar definicion dqn zona_tiro
for a in ppo a2c dqn; do
  for t in persecucion conduccion definicion jerarquico; do lanzar $t $a sistema_ft; done
done
lanzar jerarquico ppo sistema
[ -e resultados/modelo_weights/jerarquico/ppo/final_ft/final.pth ] && lanzar jerarquico ppo final \
  --pesos resultados/modelo_weights/jerarquico/ppo/final_ft/final.pth --salida resultados/logs/jerarquico/ppo/final_ft
[ -e resultados/modelo_weights/plana/ppo/plana50/final.pth ] && lanzar plana ppo plana50
wait
echo "evaluación final terminada"
