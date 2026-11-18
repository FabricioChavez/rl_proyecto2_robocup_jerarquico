import gymnasium as gym
import pytest
import torch
from cli import parser
from modelo.arquitecturas import crear_modelo
from modelo.ventajas import gae


def agente(algorithm):
    args = parser().parse_args(
        [
            "entrenar",
            "--algoritmo",
            algorithm,
            "--red",
            "8",
            "--rollout",
            "8",
            "--batch",
            "2",
            "--epocas",
            "1",
            "--dispositivo",
            "cpu",
        ]
    )
    args.normalize_advantages = False
    args.value_coef = 0.0
    args.entropy = 0.0
    args.gamma = 0.9
    env = gym.make("CartPole-v1")
    return crear_modelo(env, args)


def test_gae_timeout_bootstrap_and_episode_boundary():
    advantages, returns = gae(
        *[
            torch.tensor(x)
            for x in (
                [1.0, 2.0, 3.0],
                [0.5, 0.7, 0.9],
                [0.7, 5.0, 123.0],
                [0.0, 0.0, 1.0],
                [0.0, 1.0, 1.0],
            )
        ],
        0.9,
        0.8,
    )
    torch.testing.assert_close(advantages, torch.tensor([5.306, 5.8, 2.1]))
    torch.testing.assert_close(returns, torch.tensor([5.806, 6.5, 3.0]))


def test_dqn_bellman_target_huber_and_frozen_target():
    model = agente("dqn")
    for parameter in model.policy.parameters():
        parameter.data.zero_()
    list(model.policy.q_net_target.parameters())[-1].data.copy_(
        torch.tensor([2.0, 4.0])
    )
    before = {
        key: value.clone()
        for key, value in model.policy.q_net_target.state_dict().items()
    }
    model.replay_buffer.muestrear = lambda *unused: dict(
        observations=torch.zeros(2, 4),
        next_observations=torch.zeros(2, 4),
        actions=torch.tensor([0, 1]),
        rewards=torch.ones(2),
        dones=torch.tensor([1.0, 0.0]),
        timeouts=torch.tensor([0.0, 1.0]),
    )
    metrics = model.actualizar()
    assert metrics["objetivo_td"] == pytest.approx(2.8)
    assert metrics["perdida_q"] == pytest.approx(2.3)
    torch.testing.assert_close(before, model.policy.q_net_target.state_dict())
    assert list(model.policy.q_net.parameters())[-1].abs().sum() > 0
    model.env.close()


def test_ppo_clipping_saturates_positive_advantage():
    model = agente("ppo")
    observations = torch.zeros(2, 4)
    actions = torch.zeros(2, dtype=torch.long)
    with torch.no_grad():
        distribution, values = model.policy(observations)
        old_log_prob = distribution.log_prob(actions) - torch.log(torch.tensor(1.5))
    before = {key: value.clone() for key, value in model.policy.state_dict().items()}
    metrics = model.actualizar(
        dict(
            observaciones=observations,
            acciones=actions,
            ventajas=torch.ones(2),
            retornos=values,
            log_prob=old_log_prob,
            valores=values,
        )
    )
    assert metrics["perdida_actor"] == pytest.approx(-1.2)
    assert metrics["fraccion_clip"] == 1.0
    torch.testing.assert_close(before, model.policy.state_dict())
    model.env.close()


def test_a2c_gradient_favors_action_with_positive_advantage():
    model = agente("a2c")
    observations = torch.zeros(2, 4)
    before = model.policy.get_distribution(observations).probs.detach().clone()
    model.actualizar(
        dict(
            observaciones=observations,
            acciones=torch.tensor([0, 1]),
            ventajas=torch.tensor([1.0, -1.0]),
            retornos=torch.zeros(2),
        )
    )
    after = model.policy.get_distribution(observations).probs.detach()
    assert torch.all(after[:, 0] > before[:, 0])
    model.env.close()
