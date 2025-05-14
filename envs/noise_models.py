# envs/noise_models.py
import numpy as np
from matplotlib import pyplot as plt


class SupconNoise:
    """
    简化的超导门错误率模型:
    - base_error: 初始误码率
    - drift_sigma: 每 step 高斯漂移
    - recalib_hours: 每多少 step 重校准(误码率重抽)
    """
    def __init__(self, base_error=0.01, drift_sigma=1e-4, recalib_hours=200):
        self.base = base_error
        self.sigma = drift_sigma
        self.recalib = recalib_hours
        self.t = 0

    def step(self) -> float:
        self.t += 1
        # 校准
        if self.t % self.recalib == 0:
            self.base = max(1e-4, np.random.uniform(0.005, 0.02))
        # 漂移
        noise = np.random.normal(0, self.sigma)
        cur = max(1e-4, self.base + noise)
        return cur


if __name__ == "__main__":
    # 创建噪声模型实例
    model = SupconNoise(base_error=0.01, drift_sigma=1e-4, recalib_hours=200)

    errors = []
    steps = 1000  # 模拟 1000 步

    for _ in range(steps):
        error = model.step()
        errors.append(error)

    # 可视化误码率随时间的变化
    plt.figure(figsize=(10, 4))
    plt.plot(errors, label="Gate Error Rate")
    plt.axhline(0.01, color='gray', linestyle='--', label="Initial Error Rate")
    plt.xlabel("Step (Time Unit)")
    plt.ylabel("Error Rate")
    plt.title("SupconNoise Error Drift Over Time")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()