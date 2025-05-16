from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from gymnasium.wrappers import TimeLimit
from envs.quantum_env_mask import QuantumSchedulerMaskEnv

def make_env(rank, seed=0):
    def _init():
        base = QuantumSchedulerMaskEnv()
        base.reset(seed=seed + rank)
        def mask_fn(e): return e._get_action_mask()
        env = ActionMasker(base, mask_fn)
        env = TimeLimit(env, max_episode_steps=2000)
        return env
    return _init

def build_vec_env(n_envs: int = 8, seed: int = 0):
    return make_vec_env(
        env_id=make_env(0, seed),
        n_envs=n_envs,
        seed=seed,
        vec_env_cls=SubprocVecEnv,
    )
