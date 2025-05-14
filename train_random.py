# train_random.py
import numpy as np
from envs.quantum_env import QuantumSchedulerEnv

env = QuantumSchedulerEnv()
obs, _ = env.reset()
for t in range(1000):
    act = env.action_space.sample()
    obs, rew, _, _, _ = env.step(act)
print("随机动作 1000 步运行完成 ✅")
