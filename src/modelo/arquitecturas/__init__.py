"""Algoritmos implementados en este directorio usando únicamente PyTorch."""


def crear_modelo(env, args):
    from .ppo import PPO
    from .a2c import A2C
    from .dqn import DQN

    return {"ppo": PPO, "a2c": A2C, "dqn": DQN}[args.algo](env, args)
