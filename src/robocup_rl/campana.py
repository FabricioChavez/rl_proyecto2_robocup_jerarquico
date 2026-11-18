"""Campaña secuencial CUDA; se detiene ante errores, sin entrenamientos ocultos fallidos."""
import argparse
import os
import json
import subprocess
import sys
from pathlib import Path
from .comun.acciones import TASKS


def location(task,algo,seed,shaping='pbrs',scenario='2v1'):
    root=Path('global' if task in ('global','plana') else 'subtareas')
    name='jerarquico' if task=='global' else task
    return root/name/algo/'resultados'/shaping/scenario/f'seed_{seed}'


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--steps',type=int,default=10000)
    p.add_argument('--seeds',nargs='+',type=int,default=[0,1,2])
    p.add_argument('--algorithms',nargs='+',choices=['ppo','a2c','dqn'],default=['ppo','a2c','dqn'])
    p.add_argument('--scenario',choices=['1v1','2v1'],default='2v1')
    p.add_argument('--port',type=int,default=6200)
    p.add_argument('--ablations',action='store_true')
    p.add_argument('--evaluate',type=int,default=200)
    p.add_argument('--device',default=os.getenv('ROBOCUP_DEVICE','cuda'))
    args=p.parse_args()
    Path('runs').mkdir(exist_ok=True)
    def run(module,*arguments):
        command=[sys.executable,'-u','-m',module,*map(str,arguments)]
        print(' '.join(command),flush=True)
        subprocess.run(command,check=True)
    jobs=[]
    for seed in args.seeds:
        for algo in args.algorithms:
            for task in (*TASKS,'global','plana'):
                for shaping in (['none','pbrs','naive'] if args.ablations and task in TASKS else ['pbrs']):
                    output=location(task,algo,seed,shaping,args.scenario)
                    jobs.append(dict(task=task,algo=algo,seed=seed,output=str(output),shaping=shaping))
    Path('runs/plan.json').write_text(json.dumps(jobs,indent=2))
    for index,job in enumerate(jobs):
        task,algo,seed=job['task'],job['algo'],job['seed']
        output=Path(job['output'])
        Path('runs/campaign_status.json').write_text(json.dumps(dict(state='running',index=index,total=len(jobs),current=job),indent=2))
        common=['--task',task,'--algo',algo,'--seed',seed,'--port',args.port,'--scenario',args.scenario,'--device',args.device]
        skills=[str(location(t,algo,seed,'pbrs',args.scenario)/'final.zip') for t in TASKS]
        extra=['--skills',*skills,'--skill-algo',algo] if task=='global' else []
        try:
            if not (output/'final.zip').exists():
                steps=max(200,(args.steps//8//200)*200) if task=='global' else args.steps
                run('robocup_rl.ejecutar',*common,'--steps',steps,'--shaping',job['shaping'],'--output',output,*extra)
        except BaseException as exc:
            Path('runs/campaign_status.json').write_text(json.dumps(dict(state='failed',stage='training',index=index,current=job,error=repr(exc)),indent=2))
            raise
    for index,job in enumerate(jobs):
        task,algo,seed=job['task'],job['algo'],job['seed']
        output=Path(job['output'])
        skills=[str(location(t,algo,seed,'pbrs',args.scenario)/'final.zip') for t in TASKS]
        Path('runs/campaign_status.json').write_text(json.dumps(dict(state='running',stage='evaluation',index=index,total=len(jobs),current=job),indent=2))
        try:
            if args.evaluate and not (output/'evaluation.json').exists():
                eval_extra=['--skills',*skills] if task=='global' else []
                run('robocup_rl.evaluar','--task',task,'--algo',algo,'--model',output/'final.zip',
                    '--port',args.port,'--seed',10000,'--scenario',args.scenario,'--episodes',args.evaluate,
                    '--output',output/'evaluation.json','--device',args.device,*eval_extra)
        except BaseException as exc:
            Path('runs/campaign_status.json').write_text(json.dumps(dict(state='failed',stage='evaluation',index=index,current=job,error=repr(exc)),indent=2))
            raise
    Path('runs/campaign_status.json').write_text(json.dumps(dict(state='complete',jobs=len(jobs)),indent=2))
    run('robocup_rl.graficar')
