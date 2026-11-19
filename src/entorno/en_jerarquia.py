"""Definición entrenada DENTRO de la jerarquía (corrige el cambio de distribución del traspaso).

El partido completo corre con el coordinador y las habilidades de persecución y conducción
congelados. Cuando el coordinador elige definición, el control pasa a la red que se entrena:
cada ciclo de definición es un paso del MDP de entrenamiento. Al terminar la opción (beta(s)
o `option_steps` ciclos) la jerarquía congelada sigue jugando hasta que el coordinador vuelve
a elegir definición o termina el episodio; la recompensa de ese tramo (descontada por ciclo)
se acredita al último paso de definición. Así un tiro que acaba en gol del compañero o un
rechace recuperado cuenta, y un tiro atajado que deja el balón al rival se castiga.

Los episodios en los que el coordinador nunca elige definición no generan pasos y se descartan.
"""

import gymnasium as gym

from .acciones import N_ACTIONS

DEFINICION = 2


class DefinicionEnJerarquia(gym.Env):
    def __init__(self, env, coordinador, max_descartes=1000):
        if env.task != "global":
            raise ValueError("DefinicionEnJerarquia envuelve un SoccerEnv con task='global'")
        self.env, self.coordinador, self.max_descartes = env, coordinador, max_descartes
        self.action_space = gym.spaces.Discrete(N_ACTIONS["definicion"])
        self.observation_space = env.observation_space
        self.ciclos_opcion = 0
        self.descartados = 0

    def jugar_congelado(self, obs):
        """Juega la jerarquía congelada hasta que el coordinador elige definición o termina."""
        total, ciclos, info = 0.0, 0, {}
        while True:
            opcion = int(self.coordinador.predict(obs, deterministic=True)[0])
            if opcion == DEFINICION:
                return obs, total, False, info, ciclos
            obs, r, term, trunc, info = self.env.step(opcion)
            total += self.env.gamma**ciclos * r
            ciclos += info["primitive_steps"]
            if term or trunc:
                return obs, total, True, info, ciclos

    def reset(self, *, seed=None, options=None):
        for intento in range(1, self.max_descartes + 1):
            obs, info = self.env.reset(seed=seed)
            seed = None
            obs, _, fin, _, _ = self.jugar_congelado(obs)
            if not fin:
                self.ciclos_opcion = 0
                return obs, info
            self.descartados += 1
            if intento % 50 == 0:
                print(f"  Aviso: {intento} episodios seguidos sin elegir definición", flush=True)
        raise RuntimeError(f"El coordinador no eligió definición en {self.max_descartes} episodios")

    def step(self, action):
        obs, r, term, trunc, info = self.env._cycle(int(action), "definicion")
        self.ciclos_opcion += 1
        if term or trunc:
            return obs, r, term, trunc, info
        if self.ciclos_opcion < self.env.option_steps and not self.env.terminar_opcion(DEFINICION):
            return obs, r, False, False, info
        # Fin de la opción: el resto del sistema juega hasta la próxima definición.
        obs, extra, fin, info_resto, ciclos = self.jugar_congelado(obs)
        self.ciclos_opcion = 0
        r += self.env.gamma * extra
        if fin:
            info = {**info_resto, "primitive_steps": 1 + ciclos}
            return obs, r, True, False, info
        info["primitive_steps"] = 1 + ciclos
        return obs, r, False, False, info

    def close(self):
        self.env.close()
