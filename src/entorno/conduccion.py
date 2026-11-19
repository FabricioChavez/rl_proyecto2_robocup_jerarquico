"""Conducción v2: misma física para menú discreto y acciones parametrizadas.

El éxito exige avanzar la distancia indicada y recuperar control del balón.
Un toque permite una ventana corta de recuperación; no se permite lanzar el balón
lejos y dar por conseguida la tarea. No se pasan balones al compañero.
"""
import numpy as np
from gymnasium import spaces
from .robocup import SoccerEnv
from .acciones import relative, approach

# Parámetros normalizados: [potencia_dash, angulo_turn, potencia_kick, angulo_kick].
# Potencias físicas [0,100]; ángulos relativos al cuerpo [-180,180].
DISCRETAS = (
    (0, (-.2, 0, 0, 0)),   # dash 40
    (0, (.6, 0, 0, 0)),    # dash 80
    (1, (0, -35/180, 0, 0)),
    (1, (0, 35/180, 0, 0)),
    (2, (0, 0, -.8, 0)),   # kick 10, recto
    (2, (0, 0, -.5, 0)),   # kick 25, recto
    (2, (0, 0, -.5, -20/180)),
    (2, (0, 0, -.5, 20/180)),
)

ESTADOS_CONDUCCION = 39


def comando_parametrizado(tipo, parametros):
    p = np.asarray(parametros, dtype=np.float32)
    if p.shape != (4,) or not np.isfinite(p).all() or tipo not in (0, 1, 2):
        raise ValueError('Acción parametrizada inválida')
    p = np.clip(p, -1, 1)
    if tipo == 0:
        return f'(dash {50*(float(p[0])+1):.5f})'
    if tipo == 1:
        return f'(turn {180*float(p[1]):.5f})'
    return f'(kick {50*(float(p[2])+1):.5f} {180*float(p[3]):.5f})'


class ConduccionEnv(SoccerEnv):
    VERSION = 'conduccion_v3_recuperacion'

    def __init__(self, host, port, args):
        self.metros = args.metros
        self.radio = args.radio_control
        self.max_distancia = args.distancia_maxima
        self.gracia = args.gracia_control
        self.gracia_recuperacion = args.gracia_recuperacion
        self.distancia_recuperacion = args.distancia_recuperacion
        self.limite_posesion_rival = args.posesion_rival_maxima
        self.costo_sin_control = args.costo_sin_control
        self.defensor = args.defensor
        self.arquero = args.arquero
        self.premio = args.premio
        self.perdida = args.perdida
        self.intercepcion = args.intercepcion
        self.costo = args.costo
        self.usar_pbrs = args.pbrs
        self.escala = args.escala_pbrs
        self.sin_control = 0
        self.recuperando = False
        self.posesion_rival = 0
        self.inicio_x = 0.
        super().__init__(task='conduccion', host=host, port=port,
                         scenario='1v1', shaping='none', gamma=args.gamma,
                         horizon=args.horizonte)
        self.observation_space = spaces.Box(-np.inf, np.inf, (ESTADOS_CONDUCCION,), dtype=np.float32)
        self.action_space = spaces.Tuple((spaces.Discrete(3), spaces.Box(-1., 1., (4,), dtype=np.float32)))

    def observe(self):
        base = super().observe()
        return np.concatenate((base, np.asarray([
            (self.world.ball[0]-self.inicio_x)/self.metros,
            self.sin_control/max(1, self.gracia_recuperacion if self.recuperando else self.gracia),
            float(self.recuperando),
            self.posesion_rival/max(1, self.limite_posesion_rival),
            self.defensor/100.,
        ], dtype=np.float32)))

    def reset(self, *, seed=None, options=None):
        self.sin_control = 0
        self.recuperando = False
        self.posesion_rival = 0
        super().reset(seed=seed)
        self.server.move('p Attack 2', -50, 30)
        if not self.defensor:
            self.server.move('p Rival 2', -50, -30)
        if not self.arquero:
            self.server.move('p Rival 1', -50, 25)
        self.server.world = self.server.look()
        self.inicio_x = float(self.world.ball[0])
        self.last_ball = self.world.ball[:2].copy()
        self.base_return = 0.
        return self.observe(), {}

    def opponents(self):
        _, goalie, defender = super().opponents()
        if not self.arquero:
            goalie = '(turn 0)'
        if not self.defensor:
            defender = '(turn 0)'
        else:
            rival = self.world.players[('Rival', 2)]
            if relative(rival, self.world.ball)[0] >= .8:
                defender = approach(rival, self.world.ball, self.defensor)
        return '(turn 0)', goalie, defender

    def potencial(self):
        distance = relative(self.world.players[('Attack', 1)], self.world.ball)[0]
        return (self.world.ball[0]-self.inicio_x)/5 - distance

    def step(self, action):
        tipo, params = action
        phi = self.potencial()
        previous_return = self.base_return
        # Reutiliza protocolo, física y avance de un ciclo. La terminación y
        # recompensa de la antigua task se sustituyen por el contrato v2.
        _, _, _, _, old = super()._cycle(
            0, 'conduccion', command_override=comando_parametrizado(tipo, params))
        ball = self.world.ball
        distance = relative(self.world.players[('Attack', 1)], ball)[0]
        intercepted = any(
            enabled and relative(self.world.players[key], ball)[0] < 1.085
            and relative(self.world.players[key], ball)[0] < distance
            for enabled, key in ((self.defensor > 0, ('Rival', 2)),
                                 (self.arquero, ('Rival', 1))))
        # Contacto rival no equivale a una pérdida definitiva. El agente sigue
        # actuando para interceptar y recuperar usando dash/turn/kick.
        controlled = distance <= self.radio and not intercepted
        self.sin_control = 0 if controlled else self.sin_control+1
        self.posesion_rival = self.posesion_rival+1 if intercepted else 0
        recovered = self.recuperando and controlled
        if controlled:
            self.recuperando = False
        elif intercepted:
            self.recuperando = True
        out = abs(ball[0]) >= 52.5 or abs(ball[1]) > 34
        grace = self.gracia_recuperacion if self.recuperando else self.gracia
        maximum_distance = self.distancia_recuperacion if self.recuperando else self.max_distancia
        lost = distance > maximum_distance or self.sin_control > grace
        rival_control = self.posesion_rival >= self.limite_posesion_rival
        success = bool(ball[0]-self.inicio_x >= self.metros and controlled
                       and not intercepted and not out)
        reason = ('exito' if success else 'fuera' if out else
                  'posesion_rival' if rival_control else
                  'recuperacion_agotada' if lost and self.recuperando else
                  'perdida_control' if lost else
                  'tiempo' if self.t >= self.horizon else '')
        done = bool(reason)
        base = (self.premio if success else
                self.intercepcion if done and (rival_control or self.recuperando) else
                self.perdida if done else self.costo)
        if not done and not controlled:
            base += self.costo_sin_control
        shaping = self.escala*(self.gamma*(0. if done else self.potencial())-phi) if self.usar_pbrs else 0.
        self.base_return = previous_return+self.gamma**(self.t-1)*base
        info = dict(success=success, goal=old['goal'], reason=reason,
                    avance=float(ball[0]-self.inicio_x), distancia=distance,
                    base_reward=float(base), shaping=float(shaping), ciclos=self.t,
                    recuperando=self.recuperando, recuperado=bool(recovered),
                    posesion_rival=self.posesion_rival)
        return self.observe(), float(base+shaping), done, False, info
