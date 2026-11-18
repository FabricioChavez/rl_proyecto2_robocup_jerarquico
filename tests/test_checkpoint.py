import gymnasium as gym
import numpy as np
import pytest
import torch
from cli import parser
from modelo.arquitecturas import crear_modelo
from modelo.entrenamiento import aprender
from modelo.pesos import (
    guardar_checkpoint,
    leer_checkpoint,
    cargar_politica,
    restaurar_entrenamiento,
)


@pytest.mark.parametrize("algorithm", ["ppo", "a2c", "dqn"])
def test_pth_roundtrip_preserves_actions_optimizer_replay_and_can_continue(
    tmp_path, algorithm
):
    torch.set_num_threads(1)
    args = parser().parse_args(
        [
            "entrenar",
            "--algoritmo",
            algorithm,
            "--red",
            "8",
            "--pasos",
            "16",
            "--rollout",
            "8",
            "--batch",
            "8",
            "--epocas",
            "1",
            "--buffer",
            "32",
            "--warmup",
            "0",
            "--train-freq",
            "1",
            "--target-update",
            "4",
            "--checkpoint-cada",
            "0",
            "--semilla",
            "5",
            "--log-cada",
            "8",
            "--dispositivo",
            "cpu",
        ]
    )

    def create():
        return crear_modelo(gym.make("CartPole-v1"), args)

    original = create()
    aprender(original, original.env, args, tmp_path, tmp_path)
    path = tmp_path / "checkpoint.pth"
    guardar_checkpoint(original, path, "persecucion", algorithm)
    data = leer_checkpoint(path)
    assert data["generador"] == data["origen_pesos"] == "torch_propio"
    assert data["actualizaciones"] > 0
    inference = cargar_politica(algorithm, path, "cpu")
    observation = np.array([0.02, 0.01, -0.03, 0.04], dtype=np.float32)
    np.testing.assert_array_equal(
        inference.predict(observation)[0],
        original.predict(observation, deterministic=True)[0],
    )
    resumed = create()
    restaurar_entrenamiento(resumed, data)
    assert resumed.num_timesteps == 16
    torch.testing.assert_close(
        resumed.policy.state_dict(), original.policy.state_dict()
    )
    torch.testing.assert_close(
        resumed.policy.optimizer.state_dict(), original.policy.optimizer.state_dict()
    )
    if algorithm == "dqn":
        assert resumed._n_calls == original._n_calls
        assert resumed.replay_buffer.size() == original.replay_buffer.size()
        np.testing.assert_array_equal(
            resumed.replay_buffer.observations[:16],
            original.replay_buffer.observations[:16],
        )
    args.steps = 8
    aprender(resumed, resumed.env, args, tmp_path, tmp_path)
    assert resumed.num_timesteps == 24
    original.env.close()
    resumed.env.close()
