from gymnasium.wrappers import TimeLimit
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from envs.quantum_env_mask import QuantumSchedulerMaskEnv

base_env = QuantumSchedulerMaskEnv()

def mask_fn(env):
    return env._get_action_mask() # 确保返回一维数组

wrapped = ActionMasker(base_env, mask_fn)
env = TimeLimit(wrapped, max_episode_steps=2000)

model = MaskablePPO(
    "MultiInputPolicy",
    env,
    learning_rate=3e-4,
    n_steps=512,
    batch_size=128,
    tensorboard_log="logs_mask/",
    device="cuda",
    verbose=1,
)

model.learn(total_timesteps=20_000)
model.save("ppo_mask_v1")
print("✅ Maskable PPO 训练 2w 步完成")
