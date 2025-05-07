# 知识图谱抽取PPO训练

本目录包含用于知识图谱抽取的PPO训练脚本，基于verl框架实现。

## 使用方法

### 启动训练

推荐使用辅助脚本 `run_kg_trainer.py` 启动训练，它会自动处理命令行参数并显示默认值：

```bash
python verl/verl/trainer/run_kg_trainer.py
```

该脚本支持以下命令行参数：

- `--config_path`: 自定义配置文件路径 (默认: `/mnt/afs/tanka/shihao/project/data/kg_extraction/kg_extraction_config.yaml`)
- `--data_path`: 训练数据路径 (默认: `/mnt/afs/tanka/shihao/data/kg_extraction/kg_extraction_train.jsonl`)
- `--model_path`: 基础模型路径 (默认: `/mnt/afs/tanka/shihao/model/Qwen2.5-0.5B-Instruct`)
- `--output_dir`: 输出目录 (默认: `/mnt/afs/tanka/shihao/outputs/kg_extraction_debug`)
- `--debug`: 调试模式，使用较小的训练配置
- `--quick_test`: 快速测试模式，仅初始化训练环境但不开始训练

### 示例

1. 使用默认参数运行：
```bash
python verl/verl/trainer/run_kg_trainer.py
```

2. 使用调试模式运行：
```bash
python verl/verl/trainer/run_kg_trainer.py --debug
```

3. 指定自定义配置文件和模型路径：
```bash
python verl/verl/trainer/run_kg_trainer.py --config_path=/path/to/config.yaml --model_path=/path/to/model
```

4. 快速测试环境初始化（不执行实际训练）：
```bash
python verl/verl/trainer/run_kg_trainer.py --quick_test
```

## 直接使用Hydra（不推荐）

如果需要直接使用Hydra接口，可以使用原始脚本，但需要遵循Hydra参数格式：

```bash
python verl/verl/trainer/main_ppo_kg.py --config-path=/path/to/config/dir --config-name=config_file data.train_path=/path/to/data
```

注意：这种方式不会显示默认参数值。

## 配置文件格式

配置文件使用YAML格式，包含以下主要部分：

- `data`: 数据相关配置
- `actor_rollout_ref`: Actor模型配置 
- `critic`: Critic模型配置
- `trainer`: 训练器配置
- `algorithm`: 算法配置
- `reward_model`: 奖励模型配置

详细配置项请参考配置文件示例。 