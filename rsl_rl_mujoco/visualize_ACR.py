#!/usr/bin/env python3
import os
import time
import yaml
import torch
import argparse
import gymnasium as gym
from pathlib import Path

from rsl_rl.modules import ActorCritic
from rsl_rl.modules import ActorCriticRecurrent

from rsl_rl_mujoco.env_wrapper import GymMujocoWrapper

from pathlib import Path

class PolicyVisualizer_ACR:
    def __init__(self, cfg_path):
        self.load_config(cfg_path)
        self.setup_environment()
        self.load_policy()
        self.is_recurrent = False

    def load_config(self, cfg_path):
        """Load visualization configuration"""
        with open(cfg_path, 'r') as f:
            self.cfg = yaml.safe_load(f)
        
        # Set default device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def setup_environment(self):
        """Create and configure the environment"""
        env_kwargs = {
            "render_mode": self.cfg["env"]["render_mode"],
        }
        
        # Add video recording if enabled
        if self.cfg["visualization"]["record_video"]:
            os.makedirs(self.cfg["visualization"]["video_dir"], exist_ok=True)
            env_kwargs.update({
                "render_mode": "rgb_array",
                "width": self.cfg["env"]["width"],
                "height": self.cfg["env"]["height"],
            })
        
        # Create base environment
        self.env = gym.make(
            self.cfg["env"]["id"],
            **env_kwargs
        )
        
        # Wrap for RSL-RL compatibility
        self.env = GymMujocoWrapper(
            self.env,
            device=self.device,
            is_finite_horizon=False
        )

        # Optional: Add video recorder wrapper
        if self.cfg["visualization"]["record_video"]:
            from gymnasium.wrappers import RecordVideo
            self.env = RecordVideo(
                self.env,
                self.cfg["visualization"]["video_dir"],
                episode_trigger=lambda x: True,
                name_prefix="policy_visualization"
            )

    # 3) load_policy：根据 yaml 实例化对应的策略，并在 recurrent 情况下初始化 rnn 状态
    def load_policy(self):
        pol_cfg = self.cfg["policy"]
        class_name = pol_cfg.get("class_name", "ActorCritic")

        common_kwargs = dict(
            num_actions=self.env.num_actions,
            num_actor_obs=self.env.num_obs,
            num_critic_obs=self.env.num_obs,
            actor_hidden_dims=pol_cfg["actor_hidden_dims"],
            critic_hidden_dims=pol_cfg["critic_hidden_dims"],
            activation=pol_cfg.get("activation", "elu"),
        )

        if class_name == "ActorCriticRecurrent":
            assert ActorCriticRecurrent is not None, "ActorCriticRecurrent 未找到，请确认 rsl_rl 版本。"
            self.policy = ActorCriticRecurrent(
                rnn_type=pol_cfg.get("rnn_type", "lstm"),
                rnn_hidden_size=pol_cfg.get("rnn_hidden_size", 64),
                rnn_num_layers=pol_cfg.get("rnn_num_layers", 1),
                **common_kwargs
            ).to(self.device)
            self.is_recurrent = True

            # 初始化 RNN 状态与 masks
            # 一般需要 [num_layers, num_envs, hidden_size]（某些实现是 [num_envs, hidden_size]）
            num_envs = getattr(self.env, "num_envs", 1)
            hidden = pol_cfg.get("rnn_hidden_size", 64)
            layers = pol_cfg.get("rnn_num_layers", 1)

            # 兼容 LSTM/GRU：如果是 LSTM，通常需要 (h, c)
            self.rnn_states = {
                "actor": torch.zeros(layers, num_envs, hidden, device=self.device),
                "critic": torch.zeros(layers, num_envs, hidden, device=self.device),
            }
            # masks: 1.0 表示未终止，0.0 表示新 episode（很多实现用这个重置 RNN）
            self.masks = torch.ones(num_envs, 1, device=self.device)
        else:
            self.policy = ActorCritic(**common_kwargs).to(self.device)
            self.is_recurrent = False

        # 加载 checkpoint（映射到当前 device）
        checkpoint = torch.load(pol_cfg["checkpoint"], map_location=self.device)
        # 如果结构完全匹配，用 strict=True（默认）；若你还看到少量无关 keys，可暂时设 strict=False
        self.policy.load_state_dict(checkpoint["model_state_dict"])
        self.policy.eval()
        print(f"Loaded policy from {pol_cfg['checkpoint']}")


    # 4) run：根据是否 recurrent，走不同推理分支并维护 masks / rnn_states
    def run(self):
        print(f"Starting visualization for {self.cfg['visualization']['num_episodes']} episodes...")

        for episode in range(self.cfg["visualization"]["num_episodes"]):
            obs, _ = self.env.reset()
            episode_reward = 0.0
            done = False

            # 每个 episode 重置 RNN
            if self.is_recurrent:
                pol_cfg = self.cfg["policy"]
                num_envs = getattr(self.env, "num_envs", 1)
                hidden = pol_cfg.get("rnn_hidden_size", 64)
                layers = pol_cfg.get("rnn_num_layers", 1)
                self.rnn_states["actor"].zero_()
                self.rnn_states["critic"].zero_()
                self.masks.fill_(1.0)

            while not done:
                with torch.no_grad():
                    if self.is_recurrent:
                        # 兼容可能的两种返回签名
                        try:
                            actions, self.rnn_states = self.policy.act_inference(
                                obs, self.rnn_states, self.masks
                            )
                        except TypeError:
                            actions = self.policy.act_inference(
                                obs, self.rnn_states, self.masks
                            )
                    else:
                        actions = self.policy.act_inference(obs)

                obs, reward, done, info = self.env.step(actions)
                episode_reward += reward.item()

                # 更新 masks（done→0.0；未 done→1.0）
                if self.is_recurrent:
                    # 有的 wrapper 返回 bool；有的 batched。统一成 tensor [num_envs, 1]
                    if isinstance(done, bool):
                        done_tensor = torch.tensor([[done]], device=self.device, dtype=torch.float32)
                    else:
                        done_tensor = done.to(self.device).float().view(-1, 1)
                    self.masks = 1.0 - done_tensor.clamp(0, 1)

                time.sleep(1.0 / (self.env.max_episode_length * self.cfg["visualization"]["speedup"]))

            print(f"Episode {episode + 1}: Reward = {episode_reward:.1f}")

        self.env.close()


def main():
    parser = argparse.ArgumentParser(description="RSL-RL Policy Visualizer")
    parser.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).resolve().parent / "configs" / "visualize.yaml"),
        help="Path to configuration file"
    )
    args = parser.parse_args()
    
    visualizer = PolicyVisualizer_ACR(args.config)
    visualizer.run()

if __name__ == "__main__":
    main()