from stable_baselines3 import PPO, A2C, DQN

ALGORITHMS = {'ppo': PPO, 'a2c': A2C, 'dqn': DQN}


def make_model(algo, env, seed, device='cuda'):
    shared = dict(policy='MlpPolicy', env=env, seed=seed, device=device,
                  learning_rate=3e-4, gamma=0.99 ** (env.option_steps if env.task == 'global' else 1),
                  policy_kwargs=dict(net_arch=[128,128]), verbose=0)
    if algo == 'ppo':
        return PPO(**shared, n_steps=200, batch_size=100, n_epochs=5, gae_lambda=0.95,
                   clip_range=0.2, ent_coef=0.01)
    if algo == 'a2c':
        return A2C(**shared, n_steps=100, gae_lambda=0.95, ent_coef=0.01)
    return DQN(**shared, buffer_size=50000, learning_starts=500, batch_size=128,
               train_freq=4, gradient_steps=1, target_update_interval=500,
               exploration_fraction=0.5, exploration_final_eps=0.05)


def load_model(algo, path, device='cuda', env=None):
    return ALGORITHMS[algo].load(path, device=device, env=env)
