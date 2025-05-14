import torch as th
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class SharedEncoder(BaseFeaturesExtractor):
    def __init__(self, obs_space: spaces.Dict):
        super().__init__(obs_space, features_dim=128)
        self.job_mlp = nn.Sequential(nn.Linear(5, 32), nn.ReLU(), nn.Linear(32, 32))
        self.mac_mlp = nn.Sequential(nn.Linear(6, 32), nn.ReLU(), nn.Linear(32, 32))
        self.final = nn.Sequential(nn.Linear(32 * 2, 128), nn.ReLU())

    def forward(self, obs):
        j = obs["jobs"]  # (B, J, 5)
        m = obs["macs"]  # (B, M, 6)
        j_enc = self.job_mlp(j).mean(dim=1)  # mean pooling
        m_enc = self.mac_mlp(m).mean(dim=1)
        x = th.cat([j_enc, m_enc], dim=-1)
        return self.final(x)  # (B, 128)


class MultiHeadPolicy(nn.Module):
    def __init__(self, n_machines, n_jobs):
        super().__init__()
        self.heads = nn.ModuleList([
            nn.Linear(128, n_jobs + 1) for _ in range(n_machines)
        ])

    def forward(self, feat):
        logits = [h(feat) for h in self.heads]  # list of (B, n_jobs+1)
        return th.stack(logits, dim=1)  # (B, M, n_jobs+1)
