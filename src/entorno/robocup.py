"""MDP continuo con física oficial. El trainer proporciona estado privilegiado."""

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
    ):
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
        self.base_return = 0.0
        self.server.reset(self.task, self.np_random, self.scenario)
        self.last_ball = self.world.ball[:2].copy()
        return self.observe(), {"backend": "rcssserver", "scenario": self.scenario}

    def opponents(self):
        w = self.world
        d, g = w.players[("Rival", 2)], w.players[("Rival", 1)]
        # Defender and goalkeeper execute real server actions every cycle.
        dc = (
            kick(d, [-52.5, 0], 100)
            if relative(d, w.ball)[0] < 0.8
            else approach(d, w.ball)
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

    def _cycle(self, action, task):
        previous_phi = self.potential()
        previous_x = self.world.ball[0]
        self.server.tick([command(task, action, self.world), *self.opponents()])
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
        success = (
            captured
            if self.task == "persecucion"
            else (
                bool(b[0] > 34 and (captured or mate_captured))
                if self.task == "conduccion"
                else goal
            )
        )
        out = bool(abs(b[0]) >= 52.5 or abs(b[1]) > 34)
        lost = bool(
            relative(w.players[("Rival", 2)], b)[0] < 0.65
            or relative(w.players[("Rival", 1)], b)[0] < 0.65
        )
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
        return self.observe(), base + shaping, terminated, truncated, info

    def step(self, action):
        action = int(action)
        if not self.action_space.contains(action):
            raise ValueError(action)
        if self.task != "global":
            return self._cycle(action, self.task)
        # Fixed decision interval gives an ordinary discounted MDP with gamma_meta=gamma**k.
        # A skill's capture/kick completion does not alter interval length.
        total, base_total = 0.0, 0.0
        for j in range(self.option_steps):
            skill = self.skills[action]
            low_action = int(skill.predict(self.observe(), deterministic=True)[0])
            obs, reward, term, trunc, info = self._cycle(low_action, TASKS[action])
            total += self.gamma**j * reward
            base_total += self.gamma**j * info["base_reward"]
            if term or trunc:
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
