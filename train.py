import yaml, gymnasium
from stable_baselines3 import PPO
from models.policy import SharedEncoder, MultiHeadPolicy

cfg = yaml.safe_load(open("configs/env_config.yaml"))
env = gymnasium.make("QuantumScheduler-v0", cfg=cfg)   # 注册后即可 gym.make

model = PPO(
    "MultiInputPolicy",
    env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=4,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    policy_kwargs=dict(
        features_extractor_class=SharedEncoder,
        features_extractor_kwargs=dict(),
    ),
    verbose=1,
    device="cuda",
)
model.learn(total_timesteps=1_000_000)
model.save("ppo_quantum_sched")
