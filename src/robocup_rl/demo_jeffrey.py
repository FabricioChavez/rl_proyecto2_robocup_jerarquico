"""Adaptador de políticas P2 a la animación Matplotlib del notebook de P1."""
import json
import math
from pathlib import Path
import time
import numpy as np
import matplotlib.pyplot as plt

from .algoritmos import load_model
from .campana import location
from .comun.acciones import TASKS, LABELS, approach, relative
from .comun.entorno import SoccerEnv
from .comun.servidor import local_server
from .comun.pausa import pause_managed_server
from .visual_jeffrey import draw_pitch


class DemoEnv(SoccerEnv):
    def opponents(self):
        mate,goalie,defender=super().opponents()
        cfg=self.conditions
        rival=self.world.players[('Rival',2)]
        if cfg.get('defensor')=='quieto':
            defender='(turn 0)'
        elif cfg.get('defensor')=='patrulla':
            defender=approach(rival,[23,12*math.sin(self.t/15)],cfg.get('potencia_defensor',100))
        elif relative(rival,self.world.ball)[0]>=.8:
            defender=approach(rival,self.world.ball,cfg.get('potencia_defensor',100))
        if not cfg.get('arquero_activo',True):
            goalie='(turn 0)'
        return mate,goalie,defender


def aplicar_escenario(env,settings):
    """Intervenciones explícitas del entrenador; no son goles ni datos de entrenamiento."""
    objects={'balon':'b','atacante':'p Emilia 1','companero':'p Emilia 2',
             'defensor':'p Rival 2','arquero':'p Rival 1'}
    if 'condiciones' in settings:
        env.conditions.update(settings['condiciones'])
    for name,position in settings.get('posiciones',{}).items():
        x,y=map(float,position[:2])
        if abs(x)>51.5 or abs(y)>33:
            raise ValueError('Las posiciones deben estar dentro del campo')
        if name=='balon':
            vx,vy=settings.get('velocidad_balon',[0.,0.])
            env.server.move(objects[name],x,y,vx=float(vx),vy=float(vy))
        else:
            env.server.move(objects[name],x,y,angle=180 if name in ('defensor','arquero') else 0)
    if 'velocidad_balon' in settings and 'balon' not in settings.get('posiciones',{}):
        ball=env.world.ball
        vx,vy=settings['velocidad_balon']
        env.server.move('b',float(ball[0]),float(ball[1]),vx=float(vx),vy=float(vy))
    env.server.world=env.server.look()
    env.last_ball=env.world.ball[:2].copy()


def dibujar(env,history,title,zona=None):
    fig,ax=plt.subplots(figsize=(11,6.8))
    draw_pitch(ax,title)
    ax.set_aspect('equal')
    if history:
        points=np.asarray(history)
        ax.plot(points[:,0],points[:,1],'-',color='#00e676',linewidth=2,label='Trayectoria atacante')
    for key,p in env.world.players.items():
        team,number=key
        label=('Atacante' if number==1 else 'Compañero') if team=='Emilia' else ('Arquero' if number==1 else 'Defensor')
        ax.plot(*p[:2],'o',markersize=12,color=('#1976d2' if team=='Emilia' else '#ef5350'),label=label)
        ax.text(p[0],p[1]+1.7,label,color='white',fontsize=8,ha='center')
    player=env.world.players[('Emilia',1)]
    angle=np.radians(player[4])
    ax.arrow(*player[:2],2*np.cos(angle),2*np.sin(angle),head_width=.9,color='yellow')
    ball=env.world.ball
    ax.plot(*ball[:2],'wo',markersize=9,markeredgecolor='black',label='Balón')
    ax.arrow(*ball[:2],ball[2]*3,ball[3]*3,head_width=.6,color='white',length_includes_head=True)
    if zona:
        x,y,r=zona
        ax.add_patch(plt.Circle((x,y),r,fill=False,edgecolor='yellow',linestyle='--',linewidth=2,label='Zona de prueba'))
    ax.legend(loc='upper left',fontsize=8,ncol=2)
    plt.tight_layout()
    return fig


def animar_politica(tarea='persecucion',algoritmo='ppo',semilla=0,escenario='2v1',
                    inicial=None,cambios=None,zona=None,pasos=80,delay=.06,
                    cada=1,paso_a_paso=False,puerto=6500,salida='runs/demo_jeffrey'):
    """Mismo flujo de animación de P1, con estado UDP real y pesos P2 congelados."""
    from IPython.display import display,clear_output
    if not 1<=pasos<=200 or cada<1 or delay<0:
        raise ValueError('Pasos entre 1 y 200; cada >=1; delay >=0')
    if zona and (len(zona)!=3 or zona[2]<=0):
        raise ValueError('Zona: [x,y,radio_positivo]')
    model_path=location(tarea,algoritmo,semilla,scenario=escenario)/'final.zip'
    model=load_model(algoritmo,model_path)
    skills=[load_model(algoritmo,location(t,algoritmo,semilla,scenario=escenario)/'final.zip') for t in TASKS] if tarea=='global' else None
    history=[]; rows=[]; option=0; remaining=0
    changes=cambios or {}
    output=Path(salida); output.mkdir(parents=True,exist_ok=True)
    config=dict(tarea=tarea,algoritmo=algoritmo,semilla=semilla,escenario=escenario,
                inicial=inicial,cambios=changes,zona=zona,model=str(model_path),scope='demostracion_intervenida_no_benchmark')
    (output/'config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    with local_server(semilla,puerto) as (host,port):
        env=DemoEnv(task=tarea,host=host,port=port,scenario=escenario,shaping='none',skills=skills)
        env.conditions={}
        try:
            env.reset(seed=semilla)
            aplicar_escenario(env,inicial or {})
            for step in range(pasos):
                modification=changes.get(step,changes.get(str(step)))
                if modification:
                    aplicar_escenario(env,modification)
                obs=env.observe()
                if tarea=='global':
                    if remaining<=0:
                        option=int(model.predict(obs,deterministic=True)[0]); remaining=8
                    active=TASKS[option]; policy=skills[option]; remaining-=1
                else:
                    active=tarea; policy=model
                action=int(policy.predict(obs,deterministic=True)[0])
                _,_,terminated,truncated,info=env._cycle(action,active)
                history.append(env.world.players[('Emilia',1)][:2].tolist())
                hit=bool(zona and np.linalg.norm(env.world.ball[:2]-zona[:2])<=zona[2])
                rows.append(dict(paso=step+1,habilidad=active,accion=LABELS[active][action],
                                 balon=env.world.ball.tolist(),zona=hit,goal=bool(info['goal']),success=bool(info['success'])))
                if step%cada==0 or step==pasos-1 or terminated or truncated:
                    with pause_managed_server(puerto):
                        title=f'{tarea} / {algoritmo.upper()} | Paso {step+1} | {active}: {LABELS[active][action]}'
                        fig=dibujar(env,history,title,zona)
                        clear_output(wait=True); display(fig)
                        fig.savefig(output/'ultimo_frame.png',dpi=120)
                        plt.close(fig)
                        if paso_a_paso and not (terminated or truncated):
                            if input('Enter: siguiente paso | q: terminar ').strip().lower()=='q':
                                break
                        elif delay:
                            time.sleep(delay)
                if terminated or truncated:
                    break
        finally:
            env.close()
            (output/'trayectoria.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    return dict(pasos=len(rows),exito_tarea=rows[-1]['success'] if rows else False,
                gol=rows[-1]['goal'] if rows else False,zona_visitada=any(r['zona'] for r in rows),
                salida=str(output),tipo='demostración; no evaluación de rendimiento')
