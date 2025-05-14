from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks
from envs.quantum_env_mask import QuantumSchedulerMaskEnv


def mask_fn(env):
    # SB3-Contrib 需要一个 “action_masks” callable
    return env._get_action_mask()


env = QuantumSchedulerMaskEnv()
model = MaskablePPO(
    "MultiInputPolicy",
    env,
    learning_rate=3e-4,
    n_steps=512,
    batch_size=128,
    tensorboard_log="logs_mask/",
    device="cuda",
    verbose=1
)

# 关键！注册 mask 回调
# model.set_action_masks(mask_fn)

model.learn(total_timesteps=20000)
model.save("ppo_mask_v1")
print("✅ Maskable PPO 训练 2w 步完成")
