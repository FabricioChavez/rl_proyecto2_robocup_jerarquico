"""MC first-visit, 4x5 estados de P1, reentrenado sobre física oficial."""
import argparse
import json
from pathlib import Path
import numpy as np
from .comun.acciones import relative
from .comun.entorno import SoccerEnv
from .comun.servidor import local_server


def discretize(env):
    distance,angle=relative(env.world.players[('Attack',1)],env.world.ball)
    d=int(np.digitize(distance,[.8,3,8]))
    a=0 if abs(angle)<=15 else (1 if -60<=angle<-15 else (2 if 15<angle<=60 else (3 if angle<-60 else 4)))
    return d,a


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--steps',type=int,default=10000)
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--port',type=int,default=6100)
    p.add_argument('--output',type=Path,default=Path('runs/baseline'))
    args=p.parse_args(); rng=np.random.default_rng(args.seed)
    q=np.zeros((4,5,4)); sums=q.copy(); counts=q.copy()
    rows=[]; consumed=0
    with local_server(args.seed,args.port) as (host,port):
        env=SoccerEnv(task='persecucion',host=host,port=port,shaping='none',horizon=200)
        try:
            while consumed<args.steps:
                env.reset(seed=args.seed+len(rows)); episode=[]
                while True:
                    s=discretize(env)
                    epsilon=max(.05,1-consumed/args.steps)
                    tied=np.flatnonzero(q[s]==q[s].max())
                    action=int(rng.integers(4) if rng.random()<epsilon else rng.choice(tied))
                    _,r,term,trunc,info=env.step(action)
                    consumed+=1; episode.append((s,action,r))
                    if term or trunc or consumed>=args.steps:
                        break
                returns=[]; g=0.
                for _,_,r in reversed(episode):
                    g=r+.99*g; returns.append(g)
                # A budget-interrupted episode is omitted from MC updates.
                if term or trunc:
                    visited=set()
                    for (s,a,_),g in zip(episode,reversed(returns)):
                        key=(*s,a)
                        if key not in visited:
                            visited.add(key); sums[key]+=g; counts[key]+=1; q[key]=sums[key]/counts[key]
                rows.append(dict(steps=consumed,return_=returns[-1],success=int(info['success'])))
        finally:
            env.close()
    args.output.mkdir(parents=True,exist_ok=True)
    np.savez(args.output/'mc.npz',q=q,counts=counts)
    (args.output/'history.json').write_text(json.dumps(rows,indent=2))
