# train_ppo.py
from stable_baselines3 import PPO
from envs.quantum_env import QuantumSchedulerEnv

env = QuantumSchedulerEnv()
model = PPO("MultiInputPolicy", env,
            n_steps=256, batch_size=64,
            learning_rate=3e-4,
            verbose=1, device="cuda")

model.learn(total_timesteps=5000)  # 先跑 5k step smoke test
model.save("ppo_quantum_sched_mvp")
print("✅ PPO 训练完成, 模型已保存")
