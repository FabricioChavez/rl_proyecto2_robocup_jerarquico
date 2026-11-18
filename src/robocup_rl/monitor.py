"""Ejecuta políticas P2 en el monitor OFICIAL rcssmonitor, sin panel propio."""
import argparse
import contextlib
from datetime import datetime
import json
import os
from pathlib import Path
import platform
import subprocess
import time
import uuid

from .algoritmos import load_model,checkpoint_path
from .campana import location
from .comun.acciones import TASKS,LABELS,approach,relative
from .comun.entorno import SoccerEnv
from .comun.servidor import local_server
from .comun.pausa import pause_managed_server


class InteractiveEnv(SoccerEnv):
    def opponents(self):
        mate,goalie,defender=super().opponents()
        config=self.conditions
        rival=self.world.players[('Rival',2)]
        behavior=config.get('defensor','perseguir')
        power=int(config.get('potencia_defensor',100))
        if behavior=='quieto':
            defender='(turn 0)'
        elif behavior=='patrulla':
            import math
            defender=approach(rival,[23,12*math.sin(self.t/15)],power)
        elif relative(rival,self.world.ball)[0]>=.8:
            defender=approach(rival,self.world.ball,power)
        if not config.get('arquero_activo',True):
            goalie='(turn 0)'
        return mate,goalie,defender


def apply_scene(env,scene):
    env.conditions.update(scene.get('condiciones',{}))
    objects={'balon':'b','atacante':'p Attack 1','companero':'p Attack 2','defensor':'p Rival 2','arquero':'p Rival 1'}
    for name,point in scene.get('posiciones',{}).items():
        x,y=map(float,point[:2])
        if abs(x)>51.5 or abs(y)>33:
            raise ValueError('Posición fuera del campo')
        if name=='balon':
            vx,vy=scene.get('velocidad_balon',[0,0])
            env.server.move('b',x,y,vx=float(vx),vy=float(vy))
        else:
            env.server.move(objects[name],x,y,angle=180 if name in ('defensor','arquero') else 0)
    if 'velocidad_balon' in scene and 'balon' not in scene.get('posiciones',{}):
        b=env.world.ball; vx,vy=scene['velocidad_balon']
        env.server.move('b',float(b[0]),float(b[1]),vx=float(vx),vy=float(vy))
    env.server.world=env.server.look()
    env.last_ball=env.world.ball[:2].copy()


@contextlib.contextmanager
def official_monitor(port,output,offscreen=False,replay=None):
    windows=platform.system()=='Windows'
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    distro=os.getenv('ROBOCUP_WSL_DISTRO','Ubuntu')
    command=['rcssmonitor']
    if replay is None:
        command+=['--connect','--server-host','127.0.0.1','--server-port',str(port),
                  '--geometry','1200x800','--show-stamina','on','--show-player-number','on',
                  '--show-status-bar','on']
    else:
        path=Path(replay).resolve().as_posix()
        if windows:
            path=subprocess.check_output(['wsl','-d',os.getenv('ROBOCUP_WSL_DISTRO','Ubuntu'),'--','wslpath','-a',path],text=True).strip()
        command.append(path)
    if windows:
        pid_path=f'/tmp/robocup-monitor-{uuid.uuid4().hex}.pid'
        if offscreen:command=['env','QT_QPA_PLATFORM=offscreen',*command]
        command=['wsl','-d',distro,'--exec','sh','-c',
                 'echo $$ > "$1"; shift; exec "$@"',
                 'monitor-launcher',pid_path,*command]
    elif offscreen:
        command=['env','QT_QPA_PLATFORM=offscreen',*command]
    with (output/'monitor.log').open('ab') as log:
        process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,
                                 **({'creationflags':subprocess.CREATE_NO_WINDOW} if windows else {}))
        try:
            time.sleep(.8)
            if process.poll() is not None:
                raise RuntimeError(f'rcssmonitor terminó. Consulta {output / "monitor.log"}')
            yield process
        finally:
            if process.poll() is None:
                if windows:
                    subprocess.run(['wsl','-d',distro,'--exec','sh','-c',
                                    'if [ -r "$1" ]; then read -r pid < "$1"; kill -TERM "$pid" 2>/dev/null || true; fi',
                                    'monitor-cleanup',pid_path],capture_output=True,timeout=10)
                process.terminate()
                try:process.wait(timeout=5)
                except subprocess.TimeoutExpired:process.kill(); process.wait()


def keyboard_command(enabled):
    if not enabled:return None
    if platform.system()=='Windows':
        import msvcrt
        return msvcrt.getwch().lower() if msvcrt.kbhit() else None
    import select,sys
    return sys.stdin.readline().strip().lower() if select.select([sys.stdin],[],[],0)[0] else None


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task',choices=[*TASKS,'global','plana'],default='global')
    p.add_argument('--algo',choices=['ppo','a2c','dqn'],default='ppo')
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--device',default=os.getenv('ROBOCUP_DEVICE','cuda'))
    p.add_argument('--scenario',choices=['1v1','2v1'],default='2v1')
    p.add_argument('--port',type=int,default=6500)
    p.add_argument('--fps',type=float,default=10)
    p.add_argument('--episodes',type=int,default=0,help='0: repetir hasta cerrar el monitor o pulsar q')
    p.add_argument('--scene',type=Path,help='JSON con inicial y cambios programados por ciclo')
    p.add_argument('--output',type=Path)
    p.add_argument('--offscreen',action='store_true',help=argparse.SUPPRESS)
    p.add_argument('--no-keyboard',action='store_true',help=argparse.SUPPRESS)
    p.add_argument('--replay',type=Path,help='Abrir un archivo .rcg ya grabado')
    args=p.parse_args()
    if not 1<=args.fps<=30 or args.episodes<0:
        p.error('fps entre 1 y 30; episodes >=0')
    output=args.output or Path('runs/monitor')/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    output.mkdir(parents=True,exist_ok=True)
    if args.replay:
        with official_monitor(args.port,output,args.offscreen,args.replay) as gui:
            gui.wait()
        return
    model=load_model(args.algo,checkpoint_path(args.task,args.algo,args.seed,args.scenario),args.device)
    skills=[load_model(args.algo,checkpoint_path(t,args.algo,args.seed,args.scenario),args.device) for t in TASKS] if args.task=='global' else None
    scene=json.loads(args.scene.read_text(encoding='utf-8')) if args.scene else {}
    (output/'config.json').write_text(json.dumps(dict(arguments=vars(args),scene=scene,scope='interactive_not_benchmark'),default=str,indent=2),encoding='utf-8')
    if not args.no_keyboard:
        print('Monitor oficial. Teclas en ESTA terminal: p pausa/reanuda, n un ciclo estando en pausa, r reinicia, q sale.',flush=True)
    else:
        print('Monitor oficial. Controles del árbitro disponibles en la ventana; termina la sesión cerrando el monitor.',flush=True)
    print('En el monitor: clic derecho -> Drop Ball / modos del árbitro. Menú Referee para PlayOn.',flush=True)
    print(f'Grabación oficial RCG y datos: {output}',flush=True)
    with local_server(args.seed,args.port,game_log_dir=output) as (host,port):
        env=InteractiveEnv(task=args.task,host=host,port=port,scenario=args.scenario,skills=skills,shaping='none')
        env.conditions={}
        try:
            with official_monitor(args.port,output,args.offscreen) as gui, (output/'acciones.jsonl').open('w',encoding='utf-8') as log:
                episode=0; paused=False; quit_requested=False
                while gui.poll() is None and (args.episodes==0 or episode<args.episodes):
                    env.conditions={}
                    env.reset(seed=args.seed+episode)
                    apply_scene(env,scene.get('inicial',{}))
                    option=0; remaining=0; episode+=1
                    print(f'Episodio {episode} | {args.task}/{args.algo} | {model.device}',flush=True)
                    while gui.poll() is None:
                        command=keyboard_command(not args.no_keyboard)
                        if command=='q':quit_requested=True; break
                        if command=='r':break
                        if command=='p':paused=not paused
                        if paused:
                            with pause_managed_server(args.port):
                                while gui.poll() is None:
                                    command=keyboard_command(not args.no_keyboard)
                                    if command in ('p','n','r','q'):break
                                    time.sleep(.05)
                            if command=='q':quit_requested=True; break
                            if command=='r':break
                            if command=='p':paused=False
                            if gui.poll() is not None:break
                        change=scene.get('cambios',{}).get(str(env.t))
                        if change:apply_scene(env,change)
                        # Apply external monitor interventions before observing the policy.
                        env.server.world=env.server.look()
                        env.last_ball=env.world.ball[:2].copy()
                        obs=env.observe()
                        if args.task=='global':
                            if remaining<=0:
                                option=int(model.predict(obs,deterministic=True)[0]); remaining=8
                            active=TASKS[option]; policy=skills[option]; remaining-=1
                        else:active=args.task; policy=model
                        action=int(policy.predict(obs,deterministic=True)[0])
                        _,_,term,trunc,info=env._cycle(action,active)
                        log.write(json.dumps(dict(episode=episode,step=env.t,skill=active,action=LABELS[active][action],ball=env.world.ball.tolist()))+'\n'); log.flush()
                        time.sleep(1/args.fps)
                        if term or trunc:break
                    if quit_requested:break
                    # Keep publishing frames while awaiting the next reset, so the
                    # official monitor remains live and the synchronous watchdog is fed.
                    deadline=time.monotonic()+2
                    while gui.poll() is None and time.monotonic()<deadline:
                        env.server.tick(['(turn 0)']*4)
                        time.sleep(.1)
        except KeyboardInterrupt:
            pass
        finally:
            env.close()


if __name__=='__main__':
    main()
