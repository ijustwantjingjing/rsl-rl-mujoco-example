import torch
import yaml
import gymnasium as gym
from rsl_rl.runners import OnPolicyRunner

from rsl_rl_mujoco.env_wrapper import GymMujocoWrapper
from rsl_rl_mujoco.env_wrapper_Fix import GymMujocoWrapperFix

from pathlib import Path


def train(
    # cfg_path: str = str(Path(__file__).resolve().parent / "configs" / "default.yaml"),
    cfg_path: str = str(Path(__file__).resolve().parent / "configs" / "Unitree-g1.yaml"),
    checkpoint_path: str | None = None,
    load_optimizer: bool = True,
    extra_iterations: int | None = None,
):
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    env_id = cfg["env"]["id"]
    num_envs = cfg["env"].get("num_envs", 4)

    
    visualize=cfg["train"].get("visualize", False)
    if visualize:
        # visualize the first environment
        envs = [
            gym.make(env_id, render_mode="human" if i == 0 else None)
            for i in range(num_envs)
        ]
    else:
        envs = [gym.make(env_id) for _ in range(num_envs)]

    print("[Wrapper] action space low:", envs[0].action_space.low)
    print("[Wrapper] action space high:", envs[0].action_space.high)

    env = GymMujocoWrapperFix(
        envs,
        # clip_actions=cfg["env"].get("clip_actions", 1.0),
        clip_actions=100.0,
        is_finite_horizon=cfg["env"].get("is_finite_horizon", True),
        device=cfg.get("device", "cpu"),
    )

    runner = OnPolicyRunner(
        env,
        cfg["train"],
        log_dir=cfg.get("log_dir", "./logs"),
        device=cfg.get("device", "cpu"),
    )

    checkpoint_path=cfg["train"].get("checkpoint_path", None)
    # === 核心：从已有 policy 继续训练 ===
    if checkpoint_path is not None:
        print(f"Loading checkpoint from {checkpoint_path}")
        runner.load(checkpoint_path, load_optimizer=load_optimizer)

    # 想多训几轮，可以用 extra_iterations；否则用原来的配置
    if extra_iterations is None:
        num_iters = cfg["train"]["num_learning_iterations"]
    else:
        num_iters = extra_iterations

    runner.learn(num_learning_iterations=num_iters)


if __name__ == "__main__":
    train()
