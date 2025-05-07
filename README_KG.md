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
python run_kg_extraction.py --input_csv /path/to/your/data.csv --data_dir ./data/kg_extraction --only_prepare_data
```

### 2. 运行训练

```bash
python run_kg_extraction.py --data_dir ./data/kg_extraction --model_path /path/to/your/model --output_dir ./outputs/kg_extraction --skip_data_preparation
```

### 3. 一步完成（准备数据并训练）

```bash
python run_kg_extraction.py --input_csv /path/to/your/data.csv --model_path /path/to/your/model --output_dir ./outputs/kg_extraction
```

## 高级配置

可以通过编辑生成的配置文件（`data/kg_extraction/kg_extraction_config.yaml`）来调整更多训练参数：

- PPO算法参数（学习率、batch大小、KL惩罚等）
- 奖励函数权重
- 训练资源配置

## 数据格式

输入CSV文件应包含以下列：
- `input_text`: 需要抽取知识图谱的文本
- `ground_truth`: 标准知识图谱（JSON格式）

生成的JSONL文件遵循verl框架的数据格式规范。

## 奖励机制

模型输出格式要求：
```
<reasoning>
推理过程...
</reasoning>
<answer>
{
  "nodes": [
    {"id": "node1", "label": "实体1", "type": "类型1"},
    ...
  ],
  "edges": [
    {"source": "node1", "target": "node2", "relation": "关系"},
    ...
  ]
}
</answer>
```

奖励计算基于：
1. 输出格式合规性（0.1权重）
2. XML标签使用正确性（0.1权重）
3. 图结构正确性与完整性（0.6权重）
4. 推理质量（0.2权重） 