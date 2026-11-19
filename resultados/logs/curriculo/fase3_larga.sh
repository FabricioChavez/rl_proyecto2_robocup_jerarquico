set -u
PY="/c/Program Files/Python311/python.exe"; W=resultados/modelo_weights; L=resultados/logs
E="--semillas 0 --fuerza-defensor 50 50 --episodios 200 --reglas recuperacion --escenarios 2v1 --sin-graficas"
echo "[$(date +%H:%M:%S)] Fase 3 larga (200k pasos)"
"$PY" -X utf8 proyecto.py entrenar --tarea definicion --algoritmo ppo --continuar $W/definicion/ppo/curriculo_f3/final.pth \
  --corrida curriculo_f3_largo --curriculo-definicion 3 --traspasos $L/traspasos/estados.csv --pasos 200000 \
  --fuerza-defensor 50 --shaping pbrs --lr 0.0001 --entropia 0.003 --kl-objetivo 0.02 --dispositivo cuda --puerto 8810 || exit 1
"$PY" -c "
import pandas as pd
d = pd.read_csv('$L/definicion/ppo/curriculo_f3_largo/episodios.csv'); n = len(d) // 5
print('gol por fase, primer 20 %:', d.head(n).groupby('fase').gol.mean().round(3).to_dict())
print('gol por fase, último 20 %:', d.tail(n).groupby('fase').gol.mean().round(3).to_dict())"
echo "[$(date +%H:%M:%S)] Corrida final3 (coordinador DAgger de final2)"
for t in persecucion conduccion definicion jerarquico; do mkdir -p $W/$t/ppo/final3; done
cp $W/persecucion/ppo/final2/final.pth $W/persecucion/ppo/final3/final.pth
cp $W/conduccion/ppo/final2/final.pth $W/conduccion/ppo/final3/final.pth
cp $W/definicion/ppo/curriculo_f3_largo/final.pth $W/definicion/ppo/final3/final.pth
cp $W/jerarquico/ppo/final2/final.pth $W/jerarquico/ppo/final3/final.pth
"$PY" -X utf8 proyecto.py evaluar --tareas definicion --algoritmos ppo --corrida curriculo_f3_largo $E --puerto 8820
"$PY" -X utf8 proyecto.py evaluar --tareas jerarquico --algoritmos ppo --corrida final3 $E --puerto 8830
"$PY" -X utf8 scripts/diagnostico_jerarquia.py --corrida final3 --meta red --x-tiro 36 --episodios 200 \
  --fuerza-defensor 50 --puerto 8840 --salida $L/traspasos/diagnostico_final3.csv
echo "Listo."
