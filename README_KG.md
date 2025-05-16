# 知识图谱抽取 - RL训练

本项目使用verl框架实现了知识图谱抽取的强化学习（PPO）训练流程。

## 项目结构

```
verl/
├── run_kg_extraction.py        # 主运行脚本
├── verl/
│   ├── trainer/
│   │   ├── main_ppo_kg.py      # KG抽取专用PPO训练脚本
│   │   └── config/             # 配置文件目录
│   └── utils/
│       ├── kg_rewards.py       # 知识图谱抽取奖励函数
│       └── kg_data_converter.py # 数据转换工具
└── data/
    └── kg_extraction/          # 数据存储目录
```

## 功能模块

1. **奖励函数（kg_rewards.py）**:
   - 实现了多种奖励指标：格式奖励、XML标签奖励、图正确性奖励、推理质量奖励
   - 支持综合评分，加权组合多个奖励指标

2. **数据转换（kg_data_converter.py）**:
   - 转换CSV格式数据为verl框架所需的JSONL格式
   - 支持训练集/验证集分割
   - 自动生成hydra配置文件

3. **训练脚本（main_ppo_kg.py）**:
   - 基于verl框架的PPO训练实现
   - 支持自定义奖励函数
   - 支持命令行参数覆盖配置文件

## 使用方法

### 1. 准备数据

```bash
python examples/data_preprocess/kg.py --local_dir ./data/kg_extraction
```

### 2. 验证LLM

```bash
python tests/kg_extract/test_local_evaluator.py
```

### 3. Debug LLM reward值计算

```
HYDRA_FULL_ERROR=1 python verl/trainer/main_ppo_kg.py debug_locally=true
```


### 4. Reward 的设计

