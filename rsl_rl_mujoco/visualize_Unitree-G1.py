#!/usr/bin/env python3
import os
import time
import yaml
import torch
import argparse
import gymnasium as gym
from pathlib import Path

# ======= 关键：导入 ActorCriticRecurrent =======
from rsl_rl.modules import ActorCriticRecurrent, EmpiricalNormalization

from rsl_rl_mujoco.env_wrapper_Fix import GymMujocoWrapperFix


class PolicyVisualizer:
    def __init__(self, cfg_path):
        self.load_config(cfg_path)
        self.setup_environment()
        self.load_policy()

    def load_config(self, cfg_path):
        with open(cfg_path, 'r') as f:
            self.cfg = yaml.safe_load(f)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def setup_environment(self):
        env_kwargs = {
            "render_mode": self.cfg["env"]["render_mode"],
        }

        if self.cfg["visualization"]["record_video"]:
            os.makedirs(self.cfg["visualization"]["video_dir"], exist_ok=True)
            env_kwargs.update({
                "render_mode": "rgb_array",
                "width": self.cfg["env"]["width"],
                "height": self.cfg["env"]["height"],
            })

        base_env = gym.make(self.cfg["env"]["id"], **env_kwargs)

        # vector wrapper for RSL-RL
        self.env = GymMujocoWrapperFix(
            [base_env],
            clip_actions=self.cfg["env"].get("clip_actions", 1.0),
            is_finite_horizon=self.cfg["env"].get("is_finite_horizon", True),
            device=self.cfg.get("device", "cpu"),
        )

    def load_policy(self):
        policy_cfg = self.cfg["policy"]

        # ======= 关键：构建 ActorCriticRecurrent =======
        self.policy = ActorCriticRecurrent(
            num_actions=self.env.num_actions,
            num_actor_obs=self.env.num_obs,
            num_critic_obs=self.env.num_obs,
            actor_hidden_dims=policy_cfg["actor_hidden_dims"],
            critic_hidden_dims=policy_cfg["critic_hidden_dims"],
            activation=policy_cfg.get("activation", "elu"),
            rnn_type=policy_cfg.get("rnn_type", "lstm"),
            rnn_hidden_size=policy_cfg.get("rnn_hidden_size", 64),
            rnn_num_layers=policy_cfg.get("rnn_num_layers", 1),
        ).to(self.device)

        checkpoint = torch.load(policy_cfg["checkpoint_path"], map_location=self.device)
        self.policy.load_state_dict(checkpoint["model_state_dict"])

        # load normalizer
        self.obs_normalizer = EmpiricalNormalization(
            shape=[self.env.num_obs], until=1e8
        ).to(self.device)

        self.obs_normalizer.load_state_dict(checkpoint["obs_norm_state_dict"])
        self.obs_normalizer.eval()

        self.policy.eval()
        print(f"Loaded recurrent policy from {policy_cfg['checkpoint_path']}")

    def run(self):
        print(f"Starting visualization for {self.cfg['visualization']['num_episodes']} episodes...")

        for episode in range(self.cfg["visualization"]["num_episodes"]):

            # ======= 关键：RNN 每个 episode 要 reset =======
            if hasattr(self.policy, "reset"):
                self.policy.reset()

            obs, _ = self.env.reset()
            episode_reward = 0.0
            done = False

            while not done:

                # cast to tensor
                if not torch.is_tensor(obs):
                    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
                else:
                    obs_t = obs.to(self.device)

                # normalize obs
                obs_norm = self.obs_normalizer(obs_t)

                # get actions
                with torch.no_grad():
                    actions = self.policy.act_inference(obs_norm)

                # obs, reward, done, _ = self.env.step(actions)
                obs, reward, done, _ = self.env.step(actions)

                if torch.is_tensor(reward):
                    episode_reward += reward.item()
                else:
                    episode_reward += float(reward)

                time.sleep(1.0 / (self.env.max_episode_length * self.cfg["visualization"]["speedup"]))

            print(f"Episode {episode + 1}: Reward = {episode_reward:.1f}")

        self.env.close()


def main():
    parser = argparse.ArgumentParser(description="RSL-RL Policy Visualizer (RNN)")
    parser.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).resolve().parent / "configs" / "visualize_g1.yaml")
    )
    args = parser.parse_args()

    vis = PolicyVisualizer(args.config)
    vis.run()


if __name__ == "__main__":
    main()
