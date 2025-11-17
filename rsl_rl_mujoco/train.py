import torch
import yaml
import gymnasium as gym
from rsl_rl.runners import OnPolicyRunner

from rsl_rl_mujoco.env_wrapper import GymMujocoWrapper
from rsl_rl_mujoco.env_wrapper_Fix import GymMujocoWrapperFix

from pathlib import Path


def train(cfg_path: str = str(Path(__file__).resolve().parent / "configs" / "default.yaml")):
    # Load configuration
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    # Create environment
    env_id = cfg["env"]["id"]
    num_envs = cfg["env"].get("num_envs", 4)
    envs = [gym.make(env_id) for _ in range(num_envs)]

    # visualize the first environment
    # envs = [
    # gym.make(env_id, render_mode="human" if i == 0 else None)
    # for i in range(num_envs)
    # ]

    # env = GymMujocoWrapper(
    env = GymMujocoWrapperFix(
        envs,
        clip_actions=cfg["env"].get("clip_actions", 1.0),
        is_finite_horizon=cfg["env"].get("is_finite_horizon", True),
        device=cfg.get("device", "cpu"),
    )

    # Initialize runner
    runner = OnPolicyRunner(
        env,
        cfg["train"],
        log_dir=cfg.get("log_dir", "./logs"),
        device=cfg.get("device", "cpu"),
    )

    # Train
    runner.learn(num_learning_iterations=cfg["train"]["num_learning_iterations"])


if __name__ == "__main__":
    train()
