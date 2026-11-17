import argparse
import json
from pathlib import Path
import numpy as np
from .algoritmos import load_model
from .comun.acciones import heuristic
from .comun.entorno import SoccerEnv
from .comun.servidor import local_server


def evaluate(env, model=None, episodes=200, seed=10000, render=False):
    rows=[]
    for episode in range(episodes):
        obs,_=env.reset(seed=seed+episode)
        total=0.
        discount=1.
        while True:
            if model is not None:
                action=int(model.predict(obs,deterministic=True)[0])
            elif env.task == 'plana':
                skill=heuristic('global',env.world)
                task=('persecucion','conduccion','definicion')[skill]
                local=heuristic(task,env.world)
                action=local if skill==0 else (local+1 if skill==1 and local>=3 else ([0,2,3][local] if skill==1 else 8+local))
            else:
                action=heuristic(env.task,env.world)
            obs,_,term,trunc,info=env.step(action)
            total+=discount*info['base_reward']
            discount*=env.gamma**info['primitive_steps']
            if render:
                env.render()
            if term or trunc:
                break
        rows.append(dict(episode=episode,seed=seed+episode,success=int(info['success']),
                         goal=int(info['goal']),base_return=total,steps=env.t))
    p=float(np.mean([r['success'] for r in rows])); n=len(rows)
    z=1.96; centre=(p+z*z/(2*n))/(1+z*z/n)
    radius=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    return dict(backend='rcssserver',episodes=n,success_rate=p,
                goal_rate=float(np.mean([r['goal'] for r in rows])),
                wilson95=[centre-radius,centre+radius],
                base_return_mean=float(np.mean([r['base_return'] for r in rows])),rows=rows)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--task',choices=['persecucion','conduccion','definicion','global','plana'],default='persecucion')
    p.add_argument('--algo',default='ppo',choices=['ppo','dqn','a2c'])
    p.add_argument('--model',type=Path)
    p.add_argument('--skills',nargs=3,type=Path)
    p.add_argument('--episodes',type=int,default=200)
    p.add_argument('--seed',type=int,default=10000)
    p.add_argument('--port',type=int,default=6100)
    p.add_argument('--scenario',choices=['1v1','2v1'],default='2v1')
    p.add_argument('--output',type=Path,default=Path('runs/evaluation.json'))
    p.add_argument('--render',action='store_true')
    args=p.parse_args()
    skills=[load_model(args.algo,path) for path in args.skills] if args.skills else None
    model=load_model(args.algo,args.model) if args.model else None
    with local_server(args.seed,args.port) as (host,port):
        env=SoccerEnv(task=args.task,host=host,port=port,scenario=args.scenario,shaping='none',horizon=200,skills=skills)
        try:
            result=evaluate(env,model,args.episodes,args.seed,args.render)
        finally:
            env.close()
    result.update(task=args.task,algorithm=args.algo if model else 'heuristic',model=str(args.model),scenario=args.scenario)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))
