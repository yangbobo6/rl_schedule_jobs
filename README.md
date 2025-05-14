quantum_rl_scheduler/
│
├── envs/
│   ├── quantum_env.py      # Gymnasium 环境模拟器
│   └── noise_models.py     # 噪声/校准漂移工具
│
├── models/
│   ├── policy.py           # Encoder + 多头策略 + Value
│   └── network_blocks.py   # MLP / Transformer 通用组件
│
├── train.py                # 训练入口 (PPO)
├── eval.py                 # 测试/可视化
├── configs/
│   ├── env_config.yaml     # 机器&作业参数
│   └── ppo_config.yaml     # 超参数
└── README.md
