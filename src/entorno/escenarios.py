"""Mismas condiciones de escenario para evaluación sin ventana y monitor oficial."""

from entorno.robocup import SoccerEnv
from entorno.acciones import approach, relative


class InteractiveEnv(SoccerEnv):
    def opponents(self):
        mate, goalie, defender = super().opponents()
        config = self.conditions
        rival = self.world.players[("Rival", 2)]
        behavior = config.get("defensor", "perseguir")
        power = int(config.get("potencia_defensor", 100))
        if behavior == "quieto":
            defender = "(turn 0)"
        elif behavior == "patrulla":
            import math

            defender = approach(rival, [23, 12 * math.sin(self.t / 15)], power)
        elif relative(rival, self.world.ball)[0] >= 0.8:
            defender = approach(rival, self.world.ball, power)
        if not config.get("arquero_activo", True):
            goalie = "(turn 0)"
        return mate, goalie, defender


def apply_scene(env, scene):
    env.conditions.update(scene.get("condiciones", {}))
    objects = {
        "balon": "b",
        "atacante": "p Attack 1",
        "companero": "p Attack 2",
        "defensor": "p Rival 2",
        "arquero": "p Rival 1",
    }
    for name, point in scene.get("posiciones", {}).items():
        x, y = map(float, point[:2])
        if abs(x) > 51.5 or abs(y) > 33:
            raise ValueError("Posición fuera del campo")
        if name == "balon":
            vx, vy = scene.get("velocidad_balon", [0, 0])
            env.server.move("b", x, y, vx=float(vx), vy=float(vy))
        else:
            env.server.move(
                objects[name], x, y, angle=180 if name in ("defensor", "arquero") else 0
            )
    if "velocidad_balon" in scene and "balon" not in scene.get("posiciones", {}):
        b = env.world.ball
        vx, vy = scene["velocidad_balon"]
        env.server.move("b", float(b[0]), float(b[1]), vx=float(vx), vy=float(vy))
    env.server.world = env.server.look()
    env.last_ball = env.world.ball[:2].copy()
