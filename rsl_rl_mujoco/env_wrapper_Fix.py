import numpy as np
import gymnasium as gym
import torch
from rsl_rl.env import VecEnv


class GymMujocoWrapperFix(VecEnv):
    """
    A wrapper for Gymnasium MuJoCo environments that adapts them to the interface expected by RSL-RL.
    """

    def __init__(
        self,
        env,
        clip_actions: float = 10.0,
        is_finite_horizon: bool = True,
        device: str = "cpu",
    ):
        if isinstance(env, list):
            self.envs = env
        else:
            self.envs = [env]
        self.num_envs = len(self.envs)
        self.clip_actions = clip_actions
        self.device = torch.device(device)

        self._cfg = type("Cfg", (), {"is_finite_horizon": is_finite_horizon})
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space

        self.num_actions = gym.spaces.flatdim(self.action_space)
        self.num_obs = gym.spaces.flatdim(self.observation_space)
        self.num_privileged_obs = 0

        self.max_episode_length = getattr(self.envs[0].spec, "max_episode_steps", 10000)
        self.episode_length_buf = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )

        self._obs = None
        self._modify_action_space()
        self.reset()

    @property
    def cfg(self) -> object:
        return self._cfg

    def get_observations(self) -> tuple[torch.Tensor, dict]:
            # 用 from_numpy 避免不必要拷贝
            obs_tensor = torch.from_numpy(self._obs).to(device=self.device, dtype=torch.float32)
            return obs_tensor, {"observations": {"policy": obs_tensor}}

    def reset(self) -> tuple[torch.Tensor, dict]:
        obs_list = []
        for i, env in enumerate(self.envs):
            obs, _ = env.reset()
            obs_list.append(obs)
            # 清零长度
            self.episode_length_buf[i] = 0
        self._obs = np.stack(obs_list, axis=0).astype(np.float32)
        return self.get_observations()

    def step(
        self, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        if self.clip_actions is not None:
            actions = torch.clamp(actions, -self.clip_actions, self.clip_actions)

        # 转 numpy float32（Gym 会 upcast 到 float64）
        actions_np = actions.detach().cpu().numpy().astype(np.float32)

        obs_list, rewards, dones, timeouts = [], [], [], []
        for i, env in enumerate(self.envs):
            obs, rew, terminated, truncated, _ = env.step(actions_np[i])
            done = terminated or truncated

            # 记录长度
            self.episode_length_buf[i] += 1
            # 记录 time_out（只在非有限视野下使用）
            timeouts.append(truncated)

            if done:
                obs, _ = env.reset()
                self.episode_length_buf[i] = 0

            obs_list.append(obs)
            rewards.append(rew)
            dones.append(done)

        self._obs = np.stack(obs_list, axis=0).astype(np.float32)

        obs_tensor = torch.from_numpy(self._obs).to(device=self.device, dtype=torch.float32)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        dones_tensor = torch.tensor(dones, dtype=torch.long, device=self.device)

        extras = {"observations": {"policy": obs_tensor}}
        if not self.cfg.is_finite_horizon:
            # 关键修复：把“因时间上限产生的截断”传回去
            timeouts_tensor = torch.tensor(timeouts, dtype=torch.long, device=self.device)
            extras["time_outs"] = timeouts_tensor

        return obs_tensor, rewards_tensor, dones_tensor, extras

    def close(self):
        for env in self.envs:
            env.close()

    def seed(self, seed: int = -1) -> int:
        for env in self.envs:
            env.reset(seed=seed)
        return seed

    def _modify_action_space(self):
        if self.clip_actions is None:
            return
        self.action_space = gym.spaces.Box(
            low=-self.clip_actions,
            high=self.clip_actions,
            shape=(self.num_actions,),
            # 统一 float32，和 RSL-RL 默认保持一致
            dtype=np.float32,
        )