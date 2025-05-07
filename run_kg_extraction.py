#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import sys
from pathlib import Path

# 添加项目根目录到Python路径
script_dir = Path(__file__).resolve().parent
sys.path.append(str(script_dir))

# 导入所需模块
from verl.utils.kg_data_converter import (
    convert_to_jsonl,
    create_train_val_split,
    generate_hydra_config
)

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="知识图谱抽取训练流程")
    
    # 数据参数
    parser.add_argument("--input_csv", type=str, 
                        default="/mnt/afs/tanka/shihao/project/tanka-kg-extract/scripts/polished_rl_training_data.csv",
                        help="输入CSV文件路径")
    parser.add_argument("--data_dir", type=str, 
                        default="./data/kg_extraction",
                        help="数据输出目录")
    parser.add_argument("--val_ratio", type=float, 
                        default=0.1, 
                        help="验证集比例")
    
    # 模型参数
    parser.add_argument("--model_path", type=str, 
                        default="/mnt/afs/tanka/shihao/model/Qwen2.5-0.5B-Instruct",
                        help="基础模型路径")
    
    # 训练参数
    parser.add_argument("--output_dir", type=str, 
                        default="./outputs/kg_extraction",
                        help="训练输出目录")
    parser.add_argument("--n_gpus", type=int, 
                        default=2, 
                        help="使用的GPU数量")
    parser.add_argument("--max_steps", type=int,
                        default=1000,
                        help="训练的最大步数")
    parser.add_argument("--batch_size", type=int,
                        default=128,
                        help="训练的批量大小")
    parser.add_argument("--learning_rate", type=float,
                        default=5e-6,
                        help="学习率")
    
    # 提示模板配置
    parser.add_argument("--language", type=str,
                        default="en",
                        choices=["en", "zh"],
                        help="提示模板语言 (en: 英文, zh: 中文)")
    
    # 流程控制
    parser.add_argument("--only_prepare_data", action="store_true",
                        help="仅准备数据，不进行训练")
    parser.add_argument("--skip_data_preparation", action="store_true",
                        help="跳过数据准备，直接训练")
    parser.add_argument("--debug", action="store_true", 
                        help="调试模式，只处理少量数据")
    
    return parser.parse_args()

def get_prompt_template(language="en"):
    """获取不同语言的提示模板"""
    if language == "zh":
        return """请分析以下文本，提取出知识图谱：

{input_text}

请按以下步骤操作：
1. 在<think>标签内解释你的分析和推理过程
2. 在<answer>标签内提供符合以下格式的JSON知识图谱

格式示例：
{{"nodes": [{{"id": 1, "label": "实体名称", "type": "实体类型"}}, ...], 
"edges": [{{"source": 1, "target": 2, "relation": "关系类型", "description": "关系解释", "keywords": ["关键词1", "关键词2"], "strength": 8, "msg_ids": [0, 1, 2]}}, ...]}}

要求：
- 严格遵循上述JSON格式
- 每个关系/边需包含：
  - source: 源实体ID
  - target: 目标实体ID
  - relation: 关系类型
  - description: 详细说明实体间关系
  - keywords: 可搜索的关键词列表
  - strength: 关系强度(1-10)
  - msg_ids: 提到该关系的消息ID列表
- msg_ids不能为空
"""
    else:  # 英文默认模板
        return """Analyze the following text and extract a knowledge graph:

{input_text}

Instructions:
1. Explain your reasoning inside <think> tags
2. Output a structured knowledge graph in JSON format inside <answer> tags

Example format:
{{"nodes": [{{"id": 1, "label": "EntityName", "type": "EntityType"}}, ...], 
"edges": [{{"source": 1, "target": 2, "relation": "RELATION_TYPE", "description": "Explanation of the relationship", "keywords": ["key1", "key2"], "strength": 8, "msg_ids": [0, 1, 2]}}, ...]}}

Requirements:
- Strictly follow the JSON format above
- For each relationship/edge, include:
  - source: ID of the source entity
  - target: ID of the target entity
  - relation: Relationship type
  - description: Detailed explanation of the relationship
  - keywords: List of searchable terms
  - strength: Relationship strength (1-10)
  - msg_ids: List of message IDs where this relationship was mentioned
- msg_ids cannot be empty
"""

def main():
    """主函数"""
    args = parse_args()
    
    # 创建必要的目录
    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 设置文件路径
    jsonl_path = os.path.join(args.data_dir, "kg_extraction_all.jsonl")
    train_path = os.path.join(args.data_dir, "kg_extraction_train.jsonl") 
    val_path = os.path.join(args.data_dir, "kg_extraction_val.jsonl")
    config_path = os.path.join(args.data_dir, "kg_extraction_config.yaml")
    
    # 数据准备
    if not args.skip_data_preparation:
        print("步骤 1: 准备数据...")
        # 获取适当的提示模板
        prompt_template = get_prompt_template(args.language)
        
        # 转换CSV到JSONL
        convert_to_jsonl(
            input_csv_path=args.input_csv,
            output_jsonl_path=jsonl_path,
            prompt_template=prompt_template,
            data_source_tag="kg_extraction"
        )
        
        # 分割训练集和验证集
        create_train_val_split(
            input_jsonl_path=jsonl_path,
            train_output_path=train_path,
            val_output_path=val_path,
            val_ratio=args.val_ratio
        )
        
        # 生成配置文件
        generate_hydra_config(
            model_path=args.model_path,
            train_data_path=train_path,
            val_data_path=val_path,
            output_dir=args.output_dir,
            reward_function_path=os.path.join(script_dir, "verl/utils/kg_rewards.py"),
            reward_function_name="compute_kg_extraction_reward",
            n_gpus_per_node=args.n_gpus,
            output_config_path=config_path
        )
        
        print(f"数据准备完成，训练数据: {train_path}, 验证数据: {val_path}")
        print(f"配置文件生成在: {config_path}")
    
    # 训练
    if not args.only_prepare_data:
        print("步骤 2: 开始训练...")
        
        # 构建训练命令
        train_cmd = [
            "python",
            os.path.join(script_dir, "verl/trainer/main_ppo_kg.py"),
            f"--config_path={config_path}",
            f"--output_dir={args.output_dir}"
        ]
        
        # 添加其他训练参数
        if args.max_steps != 1000:
            train_cmd.append(f"algorithm.epochs={args.max_steps}")
        
        if args.batch_size != 128:
            train_cmd.append(f"algorithm.batch_size={args.batch_size}")
            
        if args.learning_rate != 5e-6:
            train_cmd.append(f"algorithm.lr={args.learning_rate}")
            
        if args.debug:
            train_cmd.append("trainer.total_episodes=5")
            train_cmd.append("trainer.eval_episodes=2")
        
        # 执行训练命令
        cmd_str = " ".join(train_cmd)
        print(f"执行命令: {cmd_str}")
        
        os.environ["PYTHONPATH"] = f"{script_dir}:{os.environ.get('PYTHONPATH', '')}"
        os.system(cmd_str)
    
    print("完成！")

if __name__ == "__main__":
    main() 