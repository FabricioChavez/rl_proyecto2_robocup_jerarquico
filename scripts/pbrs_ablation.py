"""Contraejemplo reproducible de recompensa ingenua por avances positivos."""
import json
from pathlib import Path

gamma=.99
positions=[0,1,0,1,0,1,0]
naive=[max(0,b-a) for a,b in zip(positions,positions[1:])]
potential=[float(p) for p in positions]
pbrs=[gamma*b-a for a,b in zip(potential,potential[1:])]
result=dict(positions=positions,naive_discounted=sum(gamma**t*r for t,r in enumerate(naive)),
            pbrs_discounted=sum(gamma**t*r for t,r in enumerate(pbrs)),
            interpretation='En un bucle que vuelve a potencial cero, PBRS no crea beneficio; premiar solo avance sí.',
            scope='Experimento algebraico; las ablaciones de aprendizaje reales usan campana --ablations.')
out=Path('informe/pbrs_counterexample.json'); out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
