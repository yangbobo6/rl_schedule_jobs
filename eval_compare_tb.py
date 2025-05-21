import numpy as np, torch as th
from tqdm import trange
from torch.utils.tensorboard import SummaryWriter
from sb3_contrib import MaskablePPO

from envs.quantum_env_mask import QuantumSchedulerMaskEnv
from sb3_contrib.common.wrappers import ActionMasker
from gymnasium.wrappers import TimeLimit


# ---------------- 环境工厂 ----------------
def make_env(seed=None):
    base = QuantumSchedulerMaskEnv()
    env = ActionMasker(base, lambda e: e._get_action_mask())
    env = TimeLimit(env, max_episode_steps=2000)
    if seed is not None:
        env.seed(seed)
    return env


# ---------------- 三个策略 ----------------
def fifo_policy(env):
    base_env = env.unwrapped  # 获取原始环境
    acts = np.zeros(base_env.M, int)
    for m in range(base_env.M):
        if m < len(base_env.queue):
            acts[m] = m + 1
    return acts


def lowest_noise_policy(env):
    base_env = env.unwrapped
    order_mac = np.argsort([mac.base for mac in base_env.machines])
    order_job = np.argsort([-j["depth"] for j in base_env.queue])
    acts = np.zeros(base_env.M, int)
    for k, mac_id in enumerate(order_mac):
        if k < len(order_job):
            acts[mac_id] = order_job[k] + 1
    return acts


# ------------- 单 Episode 运行 -------------
def run_episode(env, rl=None, pol_fn=None):
    obs, _ = env.reset()
    ep_r = 0
    while True:
        if rl:
            act, _ = rl.predict(obs, deterministic=True)
        else:
            act = pol_fn(env)
        obs, r, term, trunc, _ = env.step(act)
        ep_r += r
        if term or trunc: break
    return ep_r


# ---------- 带 TensorBoard 评测 ----------
def evaluate(name, rl=None, pol_fn=None, n=100, writer=None, step_offset=0):
    rews = []
    for ep in trange(n, desc=name):
        env = make_env()
        r = run_episode(env, rl, pol_fn)
        rews.append(r)
        if writer:
            writer.add_scalar(f"{name}/reward", r, step_offset + ep)
        env.close()
    mean_r = np.mean(rews)
    if writer:
        writer.add_scalar(f"{name}/mean_reward", mean_r, step_offset + n)
    return mean_r


if __name__ == "__main__":
    N_EVAL = 100
    writer = SummaryWriter("logs_eval/")

    # 1) RL
    rl_model = MaskablePPO.load("ppo_parallel_best.zip", device="cpu")
    rl_mean = evaluate("RL", rl=rl_model, n=N_EVAL, writer=writer)

    # 2) FIFO
    fifo_mean = evaluate("FIFO", pol_fn=fifo_policy, n=N_EVAL,
                         writer=writer, step_offset=N_EVAL)

    # 3) Lowest-Noise
    ln_mean = evaluate("LOWEST", pol_fn=lowest_noise_policy, n=N_EVAL,
                       writer=writer, step_offset=2 * N_EVAL)

    # 打印不同策略成功率
    success_rl = evaluate("RL", make_env, 50, writer=None, rl=rl_model)
    success_fifo = evaluate("FIFO", make_env, 50, writer=None, pol_fn=fifo_policy)
    success_ln = evaluate("LOWEST", make_env, 50, writer=None, pol_fn=lowest_noise_policy)

    writer.close()

    print("\n=== Mean reward ({} eps) ===".format(N_EVAL))
    print(f"RL        : {rl_mean:.3f}")
    print(f"FIFO      : {fifo_mean:.3f}")
    print(f"LOWEST    : {ln_mean:.3f}")
