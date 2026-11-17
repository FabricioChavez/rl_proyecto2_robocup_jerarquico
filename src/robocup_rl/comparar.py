"""Bootstrap de diferencia en tasa de gol; no presupone superioridad."""
import argparse
import json
from pathlib import Path
import numpy as np

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('a',type=Path)
    p.add_argument('b',type=Path)
    p.add_argument('--output',type=Path,default=Path('informe/comparacion.json'))
    args=p.parse_args()
    a=json.loads(args.a.read_text()); b=json.loads(args.b.read_text())
    if a['episodes']<200 or b['episodes']<200:
        raise ValueError('El protocolo final requiere al menos 200 episodios por política')
    if a['scenario']!=b['scenario']:
        raise ValueError('Escenarios diferentes no son comparables')
    # Independent resampling: physical server RNG is time-seeded in upstream v19.
    rng=np.random.default_rng(2026)
    x=np.array([r['goal'] for r in a['rows']]); y=np.array([r['goal'] for r in b['rows']])
    differences=np.array([rng.choice(x,len(x)).mean()-rng.choice(y,len(y)).mean() for _ in range(10000)])
    lo,hi=np.quantile(differences,[.025,.975])
    result=dict(goal_difference=float(x.mean()-y.mean()),bootstrap95=[lo,hi],
                superior_at_95=bool(lo>0),meets_goal_target=bool(x.mean()>=.8),
                limitation='Un par de modelos: no sustituye replicación entre al menos 3 semillas')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
