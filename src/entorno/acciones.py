import math
import numpy as np

TASKS = ("persecucion", "conduccion", "definicion")
N_ACTIONS = {
    "persecucion": 12,
    "conduccion": 16,
    "definicion": 20,
    "global": 3,
    "plana": 13,
}
LABELS = {
    "persecucion": ["dash100", "dash50", "turn+35", "turn-35"],
    "conduccion": [
        "dash100",
        "turn+35",
        "turn-35",
        "drible centro",
        "drible arriba",
        "drible abajo",
        "pase",
        "robar (disputa; tackle en clasicas)",
    ],
    # 0-4: tiros originales; 5-7: moverse; 8-11: tiros intermedios (mira cada 1,5 m).
    "definicion": [
        "tiro -6",
        "tiro -3",
        "tiro centro",
        "tiro +3",
        "tiro +6",
        "dash100",
        "turn+35",
        "turn-35",
        "tiro -4.5",
        "tiro -1.5",
        "tiro +1.5",
        "tiro +4.5",
    ],
    "global": list(TASKS),
}
# Correr en 8 direcciones del CAMPO sin girar el cuerpo: (dash 100 dir) de
# rcssserver >= 13. De lado rinde ~40 % y hacia atrás ~70 % (side/back_dash_rate).
DIRECCIONES_CAMPO = (0, -45, -90, -135, 180, 135, 90, 45)
NOMBRES_DIRECCION = (
    "derecha", "arriba-derecha", "arriba", "arriba-izquierda",
    "izquierda", "abajo-izquierda", "abajo", "abajo-derecha",
)
BASE_CORRER = {"persecucion": 4, "conduccion": 8, "definicion": 12}
for _task, _base in BASE_CORRER.items():
    assert len(LABELS[_task]) == _base
    LABELS[_task] += [f"correr {nombre}" for nombre in NOMBRES_DIRECCION]
LABELS["plana"] = (
    LABELS["persecucion"][:4] + LABELS["conduccion"][3:7] + LABELS["definicion"][:5]
)
# Índices nuevos se añaden al FINAL: los modelos antiguos (7 y 5 salidas) siguen válidos.
OBJETIVOS_TIRO = {0: -6, 1: -3, 2: 0, 3: 3, 4: 6, 8: -4.5, 9: -1.5, 10: 1.5, 11: 4.5}


def wrap(angle):
    return (angle + 180) % 360 - 180


def relative(player, target):
    d = np.asarray(target[:2]) - player[:2]
    return float(np.linalg.norm(d)), wrap(
        math.degrees(math.atan2(d[1], d[0])) - player[4]
    )


def approach(player, target, power=100):
    _, angle = relative(player, target)
    return f"(turn {angle:.3f})" if abs(angle) > 15 else f"(dash {power})"


def kick(player, target, power):
    return f"(kick {power} {relative(player, target)[1]:.3f})"


def correr_campo(player, angulo):
    return f"(dash 100 {wrap(angulo - player[4]):.3f})"


def command(task, action, world, reglas="clasicas"):
    p = world.players[("Attack", 1)]
    if task in BASE_CORRER and action >= BASE_CORRER[task]:
        return correr_campo(p, DIRECCIONES_CAMPO[action - BASE_CORRER[task]])
    if task == "plana":
        if action < 4:
            return command("persecucion", action, world)
        if action < 8:
            return command("conduccion", action - 1, world, reglas)
        return command("definicion", action - 8, world)
    if task == "persecucion":
        return ["(dash 100)", "(dash 50)", "(turn 35)", "(turn -35)"][action]
    if task == "conduccion":
        if action < 3:
            return ["(dash 100)", "(turn 35)", "(turn -35)"][action]
        if action == 7:
            if reglas == "recuperacion":
                # Disputa con física oficial: patada de potencia 60 hacia adelante y
                # 35 grados AL LADO CONTRARIO del defensor. Solo tiene efecto con el
                # balón a <= 1,085 m. Con 30 recto, el toque del defensor (20 hacia
                # atrás, el mismo ciclo) lo anulaba y el balón quedaba entre los dos.
                rival = world.players[("Rival", 2)]
                lado = -1.0 if rival[1] >= world.ball[1] else 1.0
                angulo = math.radians(35 * lado)
                b = world.ball
                return kick(p, [b[0] + 10 * math.cos(angulo), b[1] + 10 * math.sin(angulo)], 60)
            # Clásicas: tackle oficial (congela 10 ciclos aunque falle).
            return f"(tackle {relative(p, [52.5, 0])[1]:.3f})"
        if action == 6:
            return kick(p, world.players[("Attack", 2)], 45)
        if reglas == "recuperacion":
            # Drible de juego: toque de potencia 30 (0,8 m/ciclo, el jugador lo
            # alcanza corriendo) y arriba/abajo a 45 grados, para poder esquivar.
            # Con potencia 12 y destino (52.5, ±12) el balón iba más lento que el
            # defensor y casi recto: no había forma de superarlo.
            dx, dy = [(1, 0), (1, -1), (1, 1)][action - 3]
            return kick(p, [p[0] + 5 * dx, p[1] + 5 * dy] if dy else [52.5, 0], 30)
        return kick(p, [52.5, [0, -12, 12][action - 3]], 12)
    if action in (5, 6, 7):
        return ["(dash 100)", "(turn 35)", "(turn -35)"][action - 5]
    return kick(p, [52.5, OBJETIVOS_TIRO[action]], 100)


def heuristic(task, world):
    p = world.players[("Attack", 1)]
    distance, angle = relative(p, world.ball)
    if task == "global":
        return 0 if distance > 0.8 else (2 if p[0] > 35 else 1)
    if task == "persecucion":
        return (
            (2 if angle > 0 else 3) if abs(angle) > 17.5 else (1 if distance < 2 else 0)
        )
    if task == "conduccion":
        if distance > 0.8:
            return (1 if angle > 0 else 2) if abs(angle) > 17.5 else 0
        defender = world.players[("Rival", 2)]
        return 4 if defender[1] > p[1] else 5
    goalie = world.players[("Rival", 1)]
    return 0 if goalie[1] >= 0 else 4
