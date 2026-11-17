"""BC y DAgger sobre la misma política PPO/A2C. No genera datos humanos ficticios."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from .algoritmos import make_model,load_model
from .comun.acciones import LABELS,heuristic
from .comun.entorno import SoccerEnv
from .comun.servidor import local_server
from .comun.pausa import pause_managed_server


def fit(model, observations, actions, epochs=30):
    obs=torch.as_tensor(np.asarray(observations),dtype=torch.float32,device=model.device)
    act=torch.as_tensor(actions,dtype=torch.long,device=model.device)
    optim=torch.optim.Adam(model.policy.parameters(),lr=3e-4)
    history=[]
    model.policy.set_training_mode(True)
    for epoch in range(epochs):
        losses=[]
        for indices in torch.randperm(len(obs),device=model.device).split(128):
            dist=model.policy.get_distribution(obs[indices])
            loss=-dist.log_prob(act[indices]).mean()
            optim.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.policy.parameters(),.5)
            optim.step(); losses.append(loss.item())
        history.append(float(np.mean(losses)))
    model.policy.set_training_mode(False)
    return history


def human_label(env):
    env.render()
    print(' | '.join(f'{i}: {label}' for i,label in enumerate(LABELS[env.task])))
    while True:
        answer=input('Acción experta (q para guardar y salir): ').strip()
        if answer=='q':
            raise KeyboardInterrupt
        if answer.isdigit() and env.action_space.contains(int(answer)):
            return int(answer)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('mode',choices=['collect','bc','dagger'])
    p.add_argument('--task',choices=['persecucion','conduccion','definicion'],required=True)
    p.add_argument('--algo',choices=['ppo','a2c'],default='ppo')
    p.add_argument('--expert',choices=['human','heuristic'],default='human')
    p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--model',type=Path)
    p.add_argument('--output',type=Path,default=Path('runs/imitacion/policy'))
    p.add_argument('--episodes',type=int,default=5)
    p.add_argument('--rounds',type=int,default=3)
    p.add_argument('--epochs',type=int,default=30)
    p.add_argument('--beta',type=float,default=0.2)
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--scenario',choices=['1v1','2v1'],default='2v1')
    p.add_argument('--port',type=int,default=6100)
    args=p.parse_args()
    if not 0<=args.beta<=1:
        p.error('--beta debe estar en [0,1]')
    observations,actions,sources=[],[],[]
    if args.dataset.exists():
        with np.load(args.dataset,allow_pickle=False) as old:
            if str(old['task'])!=args.task or str(old['scenario'])!=args.scenario:
                raise ValueError('Dataset de otra subtarea o escenario')
            observations=list(old['observations']); actions=list(old['actions']); sources=list(old['sources'])
    def save_data():
        args.dataset.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(args.dataset,observations=np.asarray(observations,dtype=np.float32),
                            actions=np.asarray(actions,dtype=np.int64),sources=np.asarray(sources,dtype='U16'),
                            task=args.task,scenario=args.scenario,observation_schema=1,action_schema=1)
    with local_server(args.seed,args.port) as (host,port):
        env=SoccerEnv(task=args.task,host=host,port=port,scenario=args.scenario)
        model=load_model(args.algo,args.model,env=env) if args.model else make_model(args.algo,env,args.seed)
        rng=np.random.default_rng(args.seed)
        try:
            losses=[]
            if args.mode=='bc':
                if not observations:
                    raise ValueError('Primero recoge demostraciones con collect')
                losses.append(fit(model,observations,actions,args.epochs))
            else:
                if args.mode=='dagger' and not args.model:
                    raise ValueError('DAgger requiere --model (política inicial BC o RL)')
                for iteration in range(args.rounds if args.mode=='dagger' else 1):
                    for ep in range(args.episodes):
                        obs,_=env.reset(seed=args.seed+iteration*1000+ep)
                        while True:
                            if args.expert=='human':
                                with pause_managed_server(port):
                                    label=human_label(env)
                            else:
                                label=heuristic(args.task,env.world)
                            observations.append(obs.copy()); actions.append(label); sources.append(args.expert)
                            # Expert labels every visited learner state; execute mixture only in DAgger.
                            execute=label if args.mode=='collect' or rng.random()<args.beta else int(model.predict(obs,deterministic=True)[0])
                            obs,_,term,trunc,_=env.step(execute)
                            if term or trunc:
                                break
                    save_data()
                    if args.mode=='dagger':
                        losses.append(fit(model,observations,actions,args.epochs))
            if args.mode!='collect':
                args.output.parent.mkdir(parents=True,exist_ok=True)
                model.save(args.output)
                args.output.with_suffix('.loss.json').write_text(json.dumps(losses,indent=2))
        except KeyboardInterrupt:
            print('Interrumpido: guardando etiquetas recogidas; la ronda incompleta no se ajusta.')
        finally:
            save_data()
            env.close()
