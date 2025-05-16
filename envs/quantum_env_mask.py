import gymnasium as gym
from gymnasium import spaces
import numpy as np
from .noise_models import SupconNoise
from .job_generator import JobGenerator

SCALE = 0.001          # 奖励缩放系数

class QuantumSchedulerMaskEnv(gym.Env):
    """
    - M 台机器 (默认 3)
    - 每台机器动作 Discrete(J+1) → 选 job_id+1 或 0=空闲
    - obs 额外输出 action_mask: shape (M, J+1)  bool
    """
    metadata = {"render_modes": []}

    def __init__(self,
                 max_jobs: int = 10,
                 num_machines: int = 3, max_episode_steps=2000,
                 alpha=1.0, beta=0.001, gamma=1.0):
        super().__init__()
        self.max_jobs = max_jobs
        self.M = num_machines
        self.alpha, self.beta, self.gamma = alpha, beta, gamma

        self.machines = [SupconNoise(np.random.uniform(0.005, 0.02))
                         for _ in range(self.M)]
        self.job_gen = JobGenerator(max_qubits=8)

        self.obs_jobs = spaces.Box(0.0, 1.0, (self.max_jobs, 5), np.float32)
        self.obs_macs = spaces.Box(0.0, 1.0, (self.M, 6), np.float32)
        self.obs_stats = spaces.Box(0.0, 1.0, (3,), np.float32)

        # 多离散拆分为 M*Discrete
        self.action_space = spaces.MultiDiscrete([self.max_jobs + 1] * self.M)

        self.max_episode_steps = max_episode_steps

        self.observation_space = spaces.Dict(
            {"jobs": self.obs_jobs,
             "macs": self.obs_macs,
             "stats": self.obs_stats,
             "action_mask": spaces.Box(0, 1, (self.M, self.max_jobs + 1), bool)}
        )

        self.queue = []
        self.elapsed = 0

    # ------------------------------------------------------------------ #
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.queue = [self.job_gen.sample_job() for _ in range(np.random.randint(1, 6))]
        self.elapsed = 0
        return self._get_obs(), {}

    # ------------------------------------------------------------------ #
    def step(self, action):
        rewards = 0.0
        chosen_jobs = set()

        for m_idx, act in enumerate(action):
            if act == 0:
                continue
            j_idx = act - 1
            # --- 动作合法性 ---
            if j_idx >= len(self.queue) or j_idx in chosen_jobs:
                rewards -= 1.0           # 强罚
                continue
            chosen_jobs.add(j_idx)
            job = self.queue[j_idx]

            # --- 执行作业 ---
            err = self.machines[m_idx].step()
            fidelity = np.exp(-err * job["depth"])
            success = fidelity >= job["fid_req"]
            rewards += (self.alpha * (1 if success else -1)
                        + self.gamma * fidelity
                        - self.beta * job["wait"])

        # 移除已执行作业（按索引倒序 pop 避免错位）
        for idx in sorted(chosen_jobs, reverse=True):
            self.queue.pop(idx)

        # 噪声漂移
        for mac in self.machines:
            mac.step()

        # 队列等待时间 +1
        for j in self.queue:
            j["wait"] += 1.0

        # 新作业
        for _ in range(np.random.poisson(2)):
            if len(self.queue) < self.max_jobs:
                self.queue.append(self.job_gen.sample_job())

        self.elapsed += 1
        # 奖励缩放
        reward_scaled = rewards * SCALE

        # 是否 TimeLimit 截断
        terminated = False
        truncated = False
        if self.elapsed >= self.max_episode_steps:
            truncated = True
        # -------- 新增/修改结束 --------

        return self._get_obs(), reward_scaled, terminated, truncated, {}

    # ------------------------------------------------------------------ #
    def _get_action_mask(self):
        """
        mask[i, k] = True  ⇒  机器 i 可以选择 动作 k
        动作 0 总是合法；动作 k>0 合法当且仅当 (k-1)<len(queue)
        """
        mask = np.zeros((self.M, self.max_jobs + 1), dtype=bool)
        mask[:, 0] = True  # 空闲合法
        valid_jobs = len(self.queue)
        if valid_jobs:
            mask[:, 1:valid_jobs + 1] = True
        return mask

    def _get_obs(self):
        jobs_arr = np.zeros((self.max_jobs, 5), np.float32)
        for i, j in enumerate(self.queue):
            jobs_arr[i] = [
                j["qubits"] / 8,
                j["depth"] / 200,
                j["shots"] / 8192,
                min(j["wait"] / 100, 1.0),
                j["fid_req"]
            ]

        mac_arr = np.zeros((self.M, 6), np.float32)
        for i, mac in enumerate(self.machines):
            mac_arr[i] = [
                1.0, mac.base, 0.5, 0.7, 0.0,
                (mac.t % mac.recalib) / mac.recalib
            ]

        stats = np.array([
            len(self.queue) / self.max_jobs,
            np.mean([j["wait"] for j in self.queue]) / 100 if self.queue else 0.0,
            self.elapsed / 1000
        ], np.float32)

        return {
            "jobs": jobs_arr,
            "macs": mac_arr,
            "stats": stats,
            "action_mask": self._get_action_mask(),
        }
