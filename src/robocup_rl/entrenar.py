import argparse
import hashlib
import json
import os
import platform
from pathlib import Path
import torch
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from .algoritmos import make_model, load_model
from .comun.entorno import SoccerEnv
from .comun.acciones import TASKS


class Checkpoints(BaseCallback):
    def __init__(self, output, interval=1000):
        super().__init__()
        self.output, self.interval = output, interval

    def _on_step(self):
        if self.num_timesteps % self.interval == 0:
            self.model.save(self.output/'checkpoint')
            status = dict(state='running', steps=self.num_timesteps, device=str(self.model.device))
            (self.output/'status.json').write_text(json.dumps(status,indent=2))
            print(json.dumps(status),flush=True)
        return True


def parser():
    p = argparse.ArgumentParser()
    p.add_argument('--task', choices=[*TASKS,'global','plana'], default='persecucion')
    p.add_argument('--algo', choices=['ppo','a2c','dqn'], default='ppo')
    p.add_argument('--steps', type=int, default=10000)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--host', default=os.getenv('SERVER_HOST','127.0.0.1'))
    p.add_argument('--port', type=int, default=6000)
    p.add_argument('--scenario', choices=['1v1','2v1'], default='2v1')
    p.add_argument('--shaping', choices=['none','pbrs','naive'], default='pbrs')
    p.add_argument('--device', default='cuda')
    p.add_argument('--output', type=Path)
    p.add_argument('--skills', nargs=3, type=Path, metavar=('PURSUIT','DRIBBLE','SHOOT'))
    p.add_argument('--skill-algo', choices=['ppo','a2c','dqn'], default='ppo')
    p.add_argument('--resume', type=Path)
    p.add_argument('--horizon', type=int, default=200)
    return p


def train(args):
    if args.steps <= 0 or (args.algo in ('ppo','a2c') and args.steps % (200 if args.algo == 'ppo' else 100)):
        raise ValueError('Pasos positivos, múltiplo de 200 (PPO) o 100 (A2C), para respetar presupuesto exacto')
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA solicitada pero no disponible; no se hace fallback silencioso')
    torch.set_num_threads(2)
    output = args.output or Path(f'runs/{args.task}/{args.algo}/{args.shaping}/{args.scenario}/seed_{args.seed}')
    output.mkdir(parents=True,exist_ok=True)
    if (output/'final.zip').exists() and not args.resume:
        raise FileExistsError(f'{output}/final.zip ya existe: utiliza otra salida o --resume')
    metadata = vars(args).copy()
    digest=hashlib.sha256()
    for source in sorted(Path(__file__).parent.rglob('*.py')):
        digest.update(str(source.relative_to(Path(__file__).parent)).replace('\\','/').encode())
        digest.update(source.read_bytes())
    metadata.update(backend='rcssserver', torch=torch.__version__, gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                    python=platform.python_version(), privileged_state=True, action_schema=1, observation_schema=1,
                    source_sha256=digest.hexdigest())
    (output/'config.json').write_text(json.dumps(metadata,default=str,indent=2),encoding='utf-8')
    skills = [load_model(args.skill_algo,p,args.device) for p in args.skills] if args.skills else None
    raw = SoccerEnv(task=args.task,host=args.host,port=args.port,scenario=args.scenario,shaping=args.shaping,
                    horizon=args.horizon,skills=skills)
    env = Monitor(raw,str(output/'episodes'),info_keywords=('success','goal','base_reward','base_return'))
    try:
        model = load_model(args.algo,args.resume,args.device,env) if args.resume else make_model(args.algo,raw,args.seed,args.device)
        model.set_env(env)
        model.set_random_seed(args.seed)
        model.set_logger(configure(str(output),['csv']))
        print(f'TRAIN {args.task}/{args.algo} device={model.device} steps={args.steps} output={output}',flush=True)
        if args.device.startswith('cuda'):
            assert next(model.policy.parameters()).is_cuda
        # Finite-horizon termination means there is no time-limit bootstrap.
        model.learn(total_timesteps=args.steps,callback=Checkpoints(output),reset_num_timesteps=not bool(args.resume),log_interval=1)
        model.save(output/'final')
        (output/'status.json').write_text(json.dumps(dict(state='complete',steps=model.num_timesteps,device=str(model.device)),indent=2))
        return output
    except BaseException as exc:
        (output/'status.json').write_text(json.dumps(dict(state='failed',error=repr(exc)),indent=2))
        raise
    finally:
        env.close()


if __name__ == '__main__':
    train(parser().parse_args())
