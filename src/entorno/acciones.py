import math
import numpy as np

TASKS = ("persecucion", "conduccion", "definicion")
N_ACTIONS = {
    "persecucion": 4,
    "conduccion": 7,
    "definicion": 5,
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
    ],
    "definicion": ["tiro -6", "tiro -3", "tiro centro", "tiro +3", "tiro +6"],
    "global": list(TASKS),
}
LABELS["plana"] = (
    LABELS["persecucion"] + LABELS["conduccion"][3:] + LABELS["definicion"]
)


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


def command(task, action, world):
    p = world.players[("Attack", 1)]
    if task == "plana":
        if action < 4:
            return command("persecucion", action, world)
        if action < 8:
            return command("conduccion", action - 1, world)
        return command("definicion", action - 8, world)
    if task == "persecucion":
        return ["(dash 100)", "(dash 50)", "(turn 35)", "(turn -35)"][action]
    if task == "conduccion":
        if action < 3:
            return ["(dash 100)", "(turn 35)", "(turn -35)"][action]
        target = (
            world.players[("Attack", 2)]
            if action == 6
            else [52.5, [0, -12, 12][action - 3]]
        )
        return kick(p, target, 45 if action == 6 else 12)
    return kick(p, [52.5, [-6, -3, 0, 3, 6][action]], 100)


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
