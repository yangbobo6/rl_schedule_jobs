from sb3_contrib import MaskablePPO
from stable_baselines3.common.utils import get_linear_fn
from vec_env_factory import build_vec_env
import torch as th
from multiprocessing import freeze_support

if __name__ == '__main__':
    freeze_support()  # Windows下多进程支持
    
    vec_env = build_vec_env(n_envs=8, seed=42)

    model = MaskablePPO(
        "MultiInputPolicy",
        vec_env,
        learning_rate=get_linear_fn(3e-4, end=1e-6, end_fraction=0.1),
        n_steps=1024,            # 每 env 1024 → 总 8k 样本
        batch_size=2048,
        ent_coef=0.01,
        vf_coef=0.5,
        gae_lambda=0.9,
        normalize_advantage=True,
        tensorboard_log="logs_parallel/",
        device="cuda" if th.cuda.is_available() else "cpu",
        verbose=1,
    )

    model.learn(total_timesteps=1_000_000)
    model.save("ppo_parallel_best")
