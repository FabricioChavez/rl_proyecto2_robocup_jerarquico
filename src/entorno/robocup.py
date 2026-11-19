"""MDP continuo con física oficial. El trainer proporciona estado privilegiado."""

import csv
from collections import deque

import gymnasium as gym
import numpy as np
from .cliente_udp import Server
from .acciones import N_ACTIONS, TASKS, command, approach, relative, kick, heuristic


class SoccerEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        task="persecucion",
        host="127.0.0.1",
        port=6000,
        scenario="2v1",
        shaping="pbrs",
        gamma=0.99,
        horizon=200,
        skills=None,
        option_steps=8,
        success_reward=10.0,
        failure_reward=-5.0,
        step_reward=-0.01,
        shaping_scale=1.0,
        reglas="clasicas",
        ciclos_posesion=12,
        potencia_defensor=100,
        curriculo=None,
        traspasos=None,
    ):
        # clasicas: el episodio termina al primer toque rival y el defensor despeja.
        # recuperacion: el defensor conduce el balón; solo se pierde si el rival lo
        # conserva `ciclos_posesion` ciclos seguidos; conducción gana al superar la
        # línea del defensor con el balón controlado (o recibido por el compañero).
        if reglas not in ("clasicas", "recuperacion"):
            raise ValueError("reglas debe ser 'clasicas' o 'recuperacion'")
        self.reglas, self.ciclos_posesion = reglas, ciclos_posesion
        if not 0 <= potencia_defensor <= 100:
            raise ValueError("potencia_defensor debe estar entre 0 y 100")
        self.potencia_defensor = potencia_defensor  # dash del defensor al perseguir
        self.posesion_rival = 0
        self.task, self.scenario = task, scenario
        self.shaping, self.gamma, self.horizon = shaping, gamma, horizon
        self.option_steps, self.skills = option_steps, skills
        self.success_reward, self.failure_reward, self.step_reward = (
            success_reward,
            failure_reward,
            step_reward,
        )
        self.shaping_scale = shaping_scale
        if task == "global" and (skills is None or len(skills) != 3):
            raise ValueError(
                "La jerarquía requiere tres subpolíticas entrenadas explícitas"
            )
        self.action_space = gym.spaces.Discrete(N_ACTIONS[task])
        self.observation_space = gym.spaces.Box(
            -np.inf, np.inf, (34,), dtype=np.float32
        )
        # Currículo de definición (solo reglas de recuperación): None = zona de tiro fija;
        # 0-3 = fase fija; "auto" = empieza en 1 y avanza por tasa de gol.
        self.curriculo = curriculo
        self.fase = 1 if curriculo == "auto" else (int(curriculo) if curriculo is not None else None)
        self.fase_episodio = self.fase
        self.resultados_fase = deque(maxlen=self.VENTANA_FASE)
        self.traspasos = self.cargar_traspasos(traspasos) if traspasos else []
        if self.fase == 3 and not self.traspasos:
            raise ValueError("La fase 3 del currículo necesita --traspasos con estados grabados")
        self.server = Server(host, port)
        self.closed = False

    @property
    def world(self):
        return self.server.world

    def observe(self):
        w = self.world
        p = w.players[("Attack", 1)]
        features = []
        for target in (
            w.ball,
            [52.5, -7.01],
            [52.5, 7.01],
            w.players[("Rival", 2)],
            w.players[("Attack", 2)],
            w.players[("Rival", 1)],
        ):
            dist, angle = relative(p, target)
            features.extend(
                [dist / 105, np.sin(np.radians(angle)), np.cos(np.radians(angle))]
            )
        features.extend(
            [
                p[0] / 52.5,
                p[1] / 34,
                p[2],
                p[3],
                np.sin(np.radians(p[4])),
                np.cos(np.radians(p[4])),
                w.ball[2] / 3,
                w.ball[3] / 3,
                self.server.stamina / 8000,
                self.t / self.horizon,
            ]
        )
        for key in [("Rival", 2), ("Rival", 1), ("Attack", 2)]:
            features.extend(w.players[key][2:4])
        return np.asarray(features, dtype=np.float32)

    def potential(self):
        p, ball = self.world.players[("Attack", 1)], self.world.ball
        if self.task == "persecucion":
            return -relative(p, ball)[0] / 10
        if self.task == "conduccion":
            return (ball[0] - 52.5) / 20 - relative(p, ball)[0] / 20
        return -float(np.linalg.norm(ball[:2] - [52.5, 0])) / 20

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0
        self.posesion_rival = 0
        self.base_return = 0.0
        self.server.reset(self.task, self.np_random, self.scenario)
        if self.reglas == "recuperacion" and self.task == "conduccion":
            # La misma situación que encuentra la jerarquía al pasar de persecución a
            # conducción: balón recién capturado en x de -6 a 0 (donde lo deja el
            # reset jerárquico) y el defensor en su puesto, x = 23. Se sortea de 16 a
            # 23 porque mientras persecución captura, el defensor ya viene avanzando.
            y = float(self.world.players[("Attack", 1)][1])
            x = float(self.np_random.uniform(-6, 0))
            self.server.move("p Attack 1", x - 0.5, y, float(self.np_random.uniform(-45, 45)))
            self.server.move("b", x, y)
            self.server.move(
                "p Attack 2", x + 6.5, y - 9 if self.scenario == "2v1" else -30
            )
            self.server.move("p Rival 2", float(self.np_random.uniform(16, 23)), y + 3, 180)
            self.server.world = self.server.look()
        if self.reglas == "recuperacion" and self.task == "definicion":
            self.fase_episodio = self.sortear_fase()
            self.reset_definicion(self.fase_episodio)
            self.server.world = self.server.look()
        self.last_ball = self.world.ball[:2].copy()
        return self.observe(), {"backend": "rcssserver", "scenario": self.scenario}

    # ------------------------------------------------------------ currículo de definición
    VENTANA_FASE = 200  # episodios de la fase actual para decidir si avanza
    UMBRAL_FASE = {1: 0.70, 2: 0.65}  # tasa de gol para pasar a la fase siguiente
    REPASO = 0.30  # fracción de episodios que repasa fases anteriores
    CAMPOS_TRASPASO = [
        f"{o}_{c}" for o in ("a1", "a2", "r1", "r2") for c in ("x", "y", "vx", "vy", "ang")
    ] + ["b_x", "b_y", "b_vx", "b_vy"]

    @staticmethod
    def cargar_traspasos(ruta):
        with open(ruta, newline="", encoding="utf-8") as f:
            return [{k: float(v) for k, v in fila.items() if k in SoccerEnv.CAMPOS_TRASPASO}
                    for fila in csv.DictReader(f)]

    def estado_traspaso(self):
        """Posición, velocidad y ángulo de los 4 jugadores y el balón (una fila del banco)."""
        w, fila = self.world, {}
        for nombre, clave in (("a1", ("Attack", 1)), ("a2", ("Attack", 2)),
                              ("r1", ("Rival", 1)), ("r2", ("Rival", 2))):
            p = w.players[clave]
            fila.update({f"{nombre}_x": p[0], f"{nombre}_y": p[1], f"{nombre}_vx": p[2],
                         f"{nombre}_vy": p[3], f"{nombre}_ang": p[4]})
        b = w.ball
        fila.update(b_x=b[0], b_y=b[1], b_vx=b[2], b_vy=b[3])
        return {k: round(float(v), 4) for k, v in fila.items()}

    def sortear_fase(self):
        if self.fase is None:
            return 0
        if self.fase == 4:
            return 0 if self.np_random.random() < self.REPASO else 4
        if self.fase > 0 and self.np_random.random() < self.REPASO:
            return int(self.np_random.integers(0, self.fase))
        return self.fase

    def registrar_fase(self, gol):
        """Avance automático: si la fase actual supera su umbral, pasa a la siguiente."""
        if self.curriculo != "auto" or self.fase_episodio != self.fase or self.fase >= 3:
            return
        self.resultados_fase.append(bool(gol))
        tope = 3 if self.traspasos else 2
        if (self.fase < tope and len(self.resultados_fase) == self.VENTANA_FASE
                and np.mean(self.resultados_fase) >= self.UMBRAL_FASE[self.fase]):
            print(f"  Currículo: fase {self.fase} superada ({np.mean(self.resultados_fase):.0%} de gol)"
                  f" -> fase {self.fase + 1}", flush=True)
            self.fase += 1
            self.resultados_fase.clear()

    def colocar(self, x, y, rumbo, balon_dist=0.5, balon_v=(0.0, 0.0)):
        """Balón en (x, y) y el atacante detrás, mirando a `rumbo`, a `balon_dist` m."""
        r = np.radians(rumbo)
        bx, by = float(np.clip(x, -50, 51)), float(np.clip(y, -32, 32))
        self.server.move("p Attack 1", bx - balon_dist * np.cos(r), by - balon_dist * np.sin(r), rumbo)
        self.server.move("b", bx, by, 0, *balon_v)
        self.server.move(
            "p Attack 2", min(bx + 6, 50),
            float(np.clip(by - 9, -30, 30)) if self.scenario == "2v1" else -30,
        )
        return bx, by

    def reset_definicion(self, fase):
        r = self.np_random
        if fase == 3 and self.traspasos and r.random() < 0.5:
            # Traspaso real grabado en la jerarquía: todo con su velocidad y ángulo.
            t = self.traspasos[int(r.integers(len(self.traspasos)))]
            for nombre, obj in (("a1", "p Attack 1"), ("a2", "p Attack 2"),
                                ("r1", "p Rival 1"), ("r2", "p Rival 2")):
                self.server.move(obj, t[f"{nombre}_x"], t[f"{nombre}_y"], t[f"{nombre}_ang"],
                                 t[f"{nombre}_vx"], t[f"{nombre}_vy"])
            self.server.move("b", t["b_x"], t["b_y"], 0, t["b_vx"], t["b_vy"])
            return
        if fase == 0:
            # Zona de tiro: balón al pie en x 32-42, y ±9, cuerpo aprox. hacia el arco y
            # el defensor de 2 a 6 m detrás.
            x, y = float(r.uniform(32, 42)), float(r.uniform(-9, 9))
            rumbo = np.degrees(np.arctan2(-y, 52.5 - x)) + float(r.uniform(-30, 30))
            self.server.move("p Attack 1", x - 0.5, y, rumbo)
            self.server.move("b", x, y)
            self.server.move(
                "p Attack 2", x + 6, float(np.clip(y - 9, -30, 30)) if self.scenario == "2v1" else -30
            )
            self.server.move("p Rival 2", x - float(r.uniform(2, 6)), y + float(r.uniform(-2, 2)), 0)
            return
        if fase == 4:
            # Lejos con ventaja: x 25-38, y ±12, mirando aprox. al arco, balón al pie o
            # rodando hacia delante (como lo suelta conducción) y el defensor ya superado,
            # 2-8 m por detrás del balón.
            x, y = float(r.uniform(25, 38)), float(r.uniform(-12, 12))
            al_arco = np.degrees(np.arctan2(-y, 52.5 - x))
            rumbo = al_arco + float(r.uniform(-40, 40))
            v = float(r.uniform(0, 0.8))
            bx, by = self.colocar(x, y, rumbo, float(r.uniform(0.4, 0.9)),
                                  (v * np.cos(np.radians(rumbo)), v * np.sin(np.radians(rumbo))))
            self.server.move("p Rival 2", bx - float(r.uniform(2, 8)), by + float(r.uniform(-3, 3)), 0)
            return
        # Fases 1-2 (y la otra mitad de la 3): más lejos y peor orientado.
        x, y = float(r.uniform(25, 45)), float(r.uniform(-15, 15))
        al_arco = np.degrees(np.arctan2(-y, 52.5 - x))
        rumbo = al_arco + float(r.uniform(-90, 90))
        if fase == 1:
            bx, by = self.colocar(x, y, rumbo)
            self.server.move("p Rival 2", bx - float(r.uniform(2, 6)), by + float(r.uniform(-2, 2)), 0)
            return
        # Fase 2: balón vivo (a 0,5-1,5 m, rodando hasta 1,5 m/ciclo) y defensor molestando.
        direccion = np.radians(rumbo + float(r.uniform(-60, 60)))
        v = float(r.uniform(0, 1.5))
        bx, by = self.colocar(x, y, rumbo, float(r.uniform(0.5, 1.5)),
                              (v * np.cos(direccion), v * np.sin(direccion)))
        lado = r.random()
        if lado < 0.3:  # delante: entre el balón y el arco
            a, d = np.radians(al_arco), float(r.uniform(3, 7))
            self.server.move("p Rival 2", bx + d * np.cos(a),
                             by + d * np.sin(a) + float(r.uniform(-2, 2)), 180)
        elif lado < 0.6:  # al costado
            self.server.move("p Rival 2", bx + float(r.uniform(-1, 1)),
                             by + float(r.choice([-1, 1])) * float(r.uniform(2, 4)), 0)
        else:  # detrás
            self.server.move("p Rival 2", bx - float(r.uniform(2, 6)), by + float(r.uniform(-2, 2)), 0)

    def opponents(self):
        w = self.world
        d, g = w.players[("Rival", 2)], w.players[("Rival", 1)]
        # Defender and goalkeeper execute real server actions every cycle.
        dc = (
            kick(d, [-52.5, 0], 20 if self.reglas == "recuperacion" else 100)
            if relative(d, w.ball)[0] < 0.8
            else approach(d, w.ball, self.potencia_defensor)
        )
        gd, ga = relative(g, w.ball)
        gc = (
            f"(catch {ga:.3f})"
            if gd < 1.7 and w.ball[0] > 36 and abs(w.ball[1]) < 20
            else approach(g, [49, np.clip(w.ball[1], -6, 6)], 70)
        )
        mate = w.players[("Attack", 2)]
        if self.scenario == "2v1":
            mc = (
                kick(mate, [52.5, -5 if g[1] > 0 else 5], 100)
                if relative(mate, w.ball)[0] < 0.8
                else approach(
                    mate, [min(w.ball[0] + 5, 44), -10 if w.ball[1] > 0 else 10], 70
                )
            )
        else:
            mc = "(turn 0)"
        return mc, gc, dc

    X_DEFINICION = 36.0  # desde aquí, con el balón al pie, se cede a definición
    RADIO_TIRO = 1.085  # distancia al balón con la que se considera "al pie" para tirar
    # Traspaso con ventaja (--x-ventaja): desde esta x, con el balón al pie y el defensor
    # ya superado (por detrás del balón), también se cede a definición. None = desactivado.
    X_VENTAJA = None
    MARGEN_VENTAJA = 1.0  # metros que el defensor debe quedar detrás del balón

    def zona_traspaso(self):
        """Balón al pie y (x > X_DEFINICION, o x > X_VENTAJA con el defensor superado)."""
        p, b = self.world.players[("Attack", 1)], self.world.ball
        if relative(p, b)[0] > self.RADIO_TIRO:
            return False
        if p[0] > self.X_DEFINICION:
            return True
        return (self.X_VENTAJA is not None and p[0] > self.X_VENTAJA
                and self.world.players[("Rival", 2)][0] < b[0] - self.MARGEN_VENTAJA)

    def terminar_opcion(self, opcion):
        """beta(s) de cada habilidad: devuelve el control al meta-controlador."""
        p, b = self.world.players[("Attack", 1)], self.world.ball
        d = relative(p, b)[0]
        if opcion == 0:  # persecución: balón capturado
            return d <= 1.085
        if opcion == 1:  # conducción: zona de tiro con el balón al pie, o balón perdido
            return self.zona_traspaso() or d > 6.0
        return d > 3.0  # definición: ya disparó (o perdió el balón)

    def robo_posible(self):
        """Disputa por física oficial: el balón está a tu alcance de patada (<= 1,085 m).
        En rcssserver cualquier jugador puede patear ese balón, aunque lo conduzca el rival."""
        if self.reglas != "recuperacion":
            return False
        return relative(self.world.players[("Attack", 1)], self.world.ball)[0] <= 1.085

    def _cycle(self, action, task, command_override=None):
        previous_phi = self.potential()
        previous_x = self.world.ball[0]
        chosen = (
            command(task, action, self.world, self.reglas)
            if command_override is None
            else command_override
        )
        self.server.tick([chosen, *self.opponents()])
        self.t += 1
        w = self.world
        p, b = w.players[("Attack", 1)], w.ball
        # Crossing interpolation prevents missed goals at high kick velocity.
        crossed = self.last_ball[0] < 52.5 <= b[0]
        y_cross = (
            self.last_ball[1]
            + (b[1] - self.last_ball[1])
            * (52.5 - self.last_ball[0])
            / max(b[0] - self.last_ball[0], 1e-9)
            if crossed
            else 100
        )
        goal = bool(crossed and abs(y_cross) < 7.01)
        captured = relative(p, b)[0] < 0.8
        mate_captured = (
            self.scenario == "2v1" and relative(w.players[("Attack", 2)], b)[0] < 0.8
        )
        rival = min(
            relative(w.players[("Rival", 2)], b)[0],
            relative(w.players[("Rival", 1)], b)[0],
        )
        if self.reglas == "recuperacion":
            attacker = relative(p, b)[0]
            self.posesion_rival = (
                self.posesion_rival + 1 if rival < 0.8 and rival < attacker else 0
            )
            lost = self.posesion_rival >= self.ciclos_posesion
            beyond = b[0] > w.players[("Rival", 2)][0] + 2.0
            conducted = bool(beyond and (attacker <= 1.085 or mate_captured))
        else:
            lost = bool(rival < 0.65)
            conducted = bool(b[0] > 34 and (captured or mate_captured))
        success = (
            captured
            if self.task == "persecucion"
            else (conducted if self.task == "conduccion" else goal)
        )
        out = bool(abs(b[0]) >= 52.5 or abs(b[1]) > 34)
        terminated = bool(success or out or (lost and self.task != "persecucion"))
        timeout = bool(self.t >= self.horizon and not terminated)
        terminated = terminated or timeout
        truncated = False
        # Finite-horizon MDP: remaining time is observed; boundary potential is zero.
        boundary = terminated or truncated
        base = (
            self.success_reward
            if success
            else (self.failure_reward if boundary else self.step_reward)
        )
        self.base_return += self.gamma ** (self.t - 1) * base
        shaping = 0.0
        if self.shaping == "pbrs":
            shaping = self.shaping_scale * (
                self.gamma * (0.0 if boundary else self.potential()) - previous_phi
            )
        elif self.shaping == "naive":
            shaping = max(0.0, b[0] - previous_x)
        self.last_ball = b[:2].copy()
        info = dict(
            success=success,
            goal=goal,
            base_reward=base,
            shaping=shaping,
            cycle=w.cycle,
            backend="rcssserver",
            scenario=self.scenario,
            primitive_steps=1,
            timeout=timeout,
            base_return=self.base_return,
        )
        if self.curriculo is not None and self.task == "definicion":
            info["fase"] = self.fase_episodio
            if terminated:
                self.registrar_fase(goal)
        return self.observe(), base + shaping, terminated, truncated, info

    def step(self, action):
        action = int(action)
        if not self.action_space.contains(action):
            raise ValueError(action)
        if self.task != "global":
            return self._cycle(action, self.task)
        # Opciones (Sutton, Precup y Singh 1999): cada habilidad dura como máximo
        # option_steps ciclos. Con reglas "recuperacion" además termina en cuanto se
        # cumple su condición beta(s) (terminar_opcion), y el meta-controlador vuelve a
        # decidir: semi-MDP con descuento gamma**k, k = ciclos realmente ejecutados.
        total, base_total = 0.0, 0.0
        for j in range(self.option_steps):
            skill = self.skills[action]
            low_action = int(skill.predict(self.observe(), deterministic=True)[0])
            obs, reward, term, trunc, info = self._cycle(low_action, TASKS[action])
            total += self.gamma**j * reward
            base_total += self.gamma**j * info["base_reward"]
            if term or trunc:
                break
            if self.reglas == "recuperacion" and self.terminar_opcion(action):
                break
        info.update(option=TASKS[action], primitive_steps=j + 1, base_reward=base_total)
        return obs, total, term, trunc, info

    def render(self):
        import matplotlib.pyplot as plt

        w = self.world
        plt.clf()
        ax = plt.gca()
        ax.set_facecolor("#287d38")
        ax.set(
            xlim=(-55, 55),
            ylim=(-36, 36),
            title=f"{self.task} | ciclo {w.cycle} | {self.scenario}",
        )
        ax.plot(
            [-52.5, 52.5, 52.5, -52.5, -52.5], [-34, -34, 34, 34, -34], color="white"
        )
        ax.plot([52.5, 52.5], [-7.01, 7.01], color="yellow", linewidth=5)
        for (team, number), p in w.players.items():
            ax.plot(
                *p[:2], "o", color="cyan" if team == "Attack" else "red", markersize=10
            )
            ax.text(p[0], p[1] + 1, f"{team} {number}", fontsize=8)
        ax.plot(*w.ball[:2], "wo", markersize=6)
        plt.pause(0.03)

    def close(self):
        if not self.closed:
            self.server.close()
            self.closed = True
