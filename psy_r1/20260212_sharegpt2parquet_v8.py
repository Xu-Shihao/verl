#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v8: ShareGPT to Parquet converter for mixed RL training.
Generates 3 task types per conversation: binary, multiclass, recommendation.

Usage:
    python 20260212_sharegpt2parquet_v8.py
    python 20260212_sharegpt2parquet_v8.py --task_types binary multiclass recommendation
    python 20260212_sharegpt2parquet_v8.py --output_dir /path/to/output
"""

import json
import pandas as pd
import os
import argparse
from typing import Dict, List, Any, Optional, Tuple
from collections import Counter

# ============================================================
# Constants
# ============================================================

TASK_TYPES = ["binary", "multiclass", "recommendation"]

DATA_SOURCE_MAP = {
    "binary":         "SMHC_RL_binary",
    "multiclass":     "SMHC_RL_multiclass",
    "recommendation": "SMHC_RL_recommendation",
}

# ============================================================
# Prompt Templates
# ============================================================

# Binary (2-class): 抑郁 vs 焦虑
BINARY_SYSTEM_PROMPT = """你是一位经验丰富的精神科医生。请阅读以下病人来精神科问诊的对话记录，并判断病人更可能患有抑郁症还是焦虑症。"""

BINARY_USER_TEMPLATE = """[问诊对话开始]
{text}
[问诊对话结束]

请用中文一步一步思考，并将思考过程放在<think>xxx</think>中输出，之后将"抑郁"或者"焦虑"的结果放在<box>xxx</box>中输出。"""

# Multiclass (4-class): 抑郁/焦虑/mix/others
MULTICLASS_SYSTEM_PROMPT = """你是一位经验丰富的精神科医生。请阅读以下病人来精神科问诊的对话记录，并判断病人的主要心理健康状况。抑郁症以**持续的情绪低落、兴趣减退和精力缺乏**为主，而焦虑症则以**过度担忧、紧张不安和对未来事件的恐惧**为主要特点。"""

MULTICLASS_USER_TEMPLATE = """[问诊对话开始]
{text}
[问诊对话结束]

请从以下四个选项中选择最合适的诊断：
- 抑郁：主要表现为抑郁症状, 满足ICD诊断要求。
- 焦虑：主要表现为焦虑症状, 满足ICD诊断要求。
- mix：同时表现出明显的抑郁和焦虑症状均满足ICD诊断要求，或者都没有满足单诊断为抑郁和焦虑的程度。
- others：其他心理健康问题（比如双向情感障碍，精神分裂症，等等）或正常状态。

请用中文一步一步思考，并将思考过程放在<think>xxx</think>中输出，之后将结果（"抑郁"、"焦虑"、"mix"或"others"）放在<box>xxx</box>中输出。"""

# Recommendation: ICD-10 code prediction (same as v7)
RECOMMENDATION_SYSTEM_PROMPT = """你是一位经验丰富的精神科医生。请阅读以下初次精神科门诊的问诊对话记录，并根据ICD-10国际疾病分类标准，仔细分析后输出患者诊断结束后的ICD-10诊断代码。

## 注意：
1. 问诊对话为初次问诊，在症状严重程度和细节不可判断的时候，请推荐未特指的icd code。
2. 诊断结果可能包含1至2个icd-10诊断结果，大多只包含一个但不超过2个。
3. 用分号分隔不同的代码。
4. 需要严格根据icd-10标准来进行诊断的分析, 避免猜测和无根据的诊断，避免诊断错误。"""

RECOMMENDATION_USER_TEMPLATE = """
[问诊对话开始]
{text}
[问诊对话结束]

## 输出格式：
请用中文一步一步思考，并将思考过程放在<think>xxx</think>中输出，之后将最后诊断的ICD-10代码必须放在<box>xxx</box>中输出，用分号分隔，格式如：<think>xxx</think><box>Fxx.x;Fxx.x;Fxx.x</box>。"""


# ============================================================
# Label Mapping Functions
# ============================================================

def get_binary_label(answer: List[str]) -> Optional[str]:
    """
    Map ICD answer codes to binary label.
    Returns "抑郁" for pure F32, "焦虑" for pure F41, None otherwise.
    """
    if not answer or not isinstance(answer, list):
        return None
    codes = sorted(answer)
    if codes == ["F32"]:
        return "抑郁"
    elif codes == ["F41"]:
        return "焦虑"
    return None


def get_multiclass_label(answer: List[str]) -> str:
    """
    Map ICD answer codes to 4-class label.
    Returns one of: "抑郁", "焦虑", "mix", "others"
    """
    if not answer or not isinstance(answer, list):
        return "others"

    has_f32 = "F32" in answer
    has_f41 = "F41" in answer

    if has_f32 and has_f41:
        return "mix"
    elif has_f32 and not has_f41:
        return "抑郁"
    elif has_f41 and not has_f32:
        return "焦虑"
    else:
        return "others"


# ============================================================
# Conversation Text Extraction
# ============================================================

def extract_conversation_text(conversations: List[Dict]) -> Tuple[str, str]:
    """
    Extract the raw conversation text and original gpt response from sharegpt format.
    Returns (text, gpt_content).
    """
    human_content = ""
    gpt_content = ""

    for conv in conversations:
        if conv['from'] == 'human':
            human_content = conv['value']
        elif conv['from'] == 'gpt':
            gpt_content = conv['value']

    # Extract dialogue between markers
    text = human_content
    start_marker = "[问诊对话开始]"
    end_marker = "[问诊对话结束]"
    start_idx = text.find(start_marker)
    end_idx = text.find(end_marker)
    if start_idx != -1 and end_idx != -1:
        text = text[start_idx + len(start_marker):end_idx].strip()

    return text, gpt_content


# ============================================================
# Prompt Builders
# ============================================================

def build_binary_prompt(text: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": BINARY_SYSTEM_PROMPT},
        {"role": "user", "content": BINARY_USER_TEMPLATE.format(text=text)},
    ]


def build_multiclass_prompt(text: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": MULTICLASS_SYSTEM_PROMPT},
        {"role": "user", "content": MULTICLASS_USER_TEMPLATE.format(text=text)},
    ]


def build_recommendation_prompt(text: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": RECOMMENDATION_SYSTEM_PROMPT},
        {"role": "user", "content": RECOMMENDATION_USER_TEMPLATE.format(text=text)},
    ]


PROMPT_BUILDERS = {
    "binary": build_binary_prompt,
    "multiclass": build_multiclass_prompt,
    "recommendation": build_recommendation_prompt,
}


# ============================================================
# Core: Multi-task Sample Generator
# ============================================================

def make_map_fn(split: str, task_types: List[str] = None):
    """
    Create a data transformation function that produces multiple samples per record.

    Args:
        split: "train" or "val"
        task_types: Which task types to generate. Default = all 3.

    Returns:
        A function that takes (data_item, idx) and returns List[Dict]
    """
    if task_types is None:
        task_types = TASK_TYPES

    def process_fn(data_item: Dict[str, Any], idx: int) -> List[Dict[str, Any]]:
        samples = []
        visit_number = data_item['visit_number']
        conversations = data_item['conversations']
        answer = data_item.get("answer", [])
        full_icd_code = data_item.get("ground_truth_codes", [])

        # Handle empty answer
        if not answer or (isinstance(answer, list) and len(answer) == 0):
            answer = ["Others"]

        # Extract conversation text once (shared across all task types)
        text, gpt_content = extract_conversation_text(conversations)

        for task_type in task_types:
            # Determine ground truth for this task type
            if task_type == "binary":
                label = get_binary_label(answer)
                if label is None:
                    continue  # Skip: not applicable for binary
                ground_truth = [label]  # list for parquet consistency
            elif task_type == "multiclass":
                ground_truth = [get_multiclass_label(answer)]  # list for parquet consistency
            elif task_type == "recommendation":
                ground_truth = answer  # list
            else:
                continue

            # Build the prompt
            prompt = PROMPT_BUILDERS[task_type](text)

            sample = {
                "data_source": DATA_SOURCE_MAP[task_type],
                "prompt": prompt,
                "ability": "medical_diagnosis",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": ground_truth,
                },
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "visit_number": visit_number,
                    "original_response": gpt_content,
                    "full_icd_code": full_icd_code,
                    "task_type": task_type,
                    "original_answer": answer,
                },
            }
            samples.append(sample)

        return samples

    return process_fn


# ============================================================
# Utility Functions
# ============================================================

def validate_files(files: List[str]) -> List[str]:
    """Validate file list, remove non-existent files."""
    valid_files = []
    for file_path in files:
        if os.path.exists(file_path):
            valid_files.append(file_path)
            print(f"  [OK] {file_path}")
        else:
            print(f"  [SKIP] 文件不存在: {file_path}")
    return valid_files


def remove_duplicates_by_visit_number(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Remove duplicates by visit_number, keep first occurrence."""
    seen_ids = set()
    deduplicated = []
    dup_count = 0
    for item in data:
        visit_number = item.get('visit_number', '')
        if visit_number not in seen_ids:
            seen_ids.add(visit_number)
            deduplicated.append(item)
        else:
            dup_count += 1
    if dup_count > 0:
        print(f"  去重: 移除 {dup_count} 条重复数据")
    return deduplicated


def print_dataset_statistics(converted_data: List[Dict], label: str):
    """Print distribution of task types and ground truth labels."""
    print(f"\n{'='*60}")
    print(f"{label} 数据集统计")
    print(f"{'='*60}")
    print(f"总样本数: {len(converted_data)}")

    # Task type distribution
    task_counter = Counter()
    gt_by_task = {}
    for item in converted_data:
        task_type = item['extra_info']['task_type']
        task_counter[task_type] += 1
        gt = item['reward_model']['ground_truth']
        if task_type not in gt_by_task:
            gt_by_task[task_type] = Counter()
        gt_key = str(gt) if isinstance(gt, list) else gt
        gt_by_task[task_type][gt_key] += 1

    print(f"\n任务类型分布:")
    for task_type in TASK_TYPES:
        count = task_counter.get(task_type, 0)
        print(f"  {task_type}: {count} 条")

    for task_type in TASK_TYPES:
        if task_type in gt_by_task:
            print(f"\n{task_type} ground_truth 分布 (top 10):")
            for gt_val, count in gt_by_task[task_type].most_common(10):
                print(f"  {gt_val}: {count}")


# ============================================================
# Main Conversion
# ============================================================

def convert_sharegpt_to_parquet(
    train_files: List[str],
    val_files: List[str],
    output_dir: str,
    task_types: List[str] = None,
    remove_duplicates: bool = False,
):
    """
    Main conversion pipeline.
    """
    if task_types is None:
        task_types = TASK_TYPES

    print("="*60)
    print("v8: ShareGPT -> Parquet (混合RL训练数据)")
    print(f"任务类型: {task_types}")
    print("="*60)

    # Validate files
    print("\n验证训练文件:")
    train_files = validate_files(train_files)
    print("验证验证文件:")
    val_files = validate_files(val_files)

    if not train_files:
        raise ValueError("没有找到有效的训练文件")
    if not val_files:
        raise ValueError("没有找到有效的验证文件")

    # Read training data
    train_data = []
    for f in train_files:
        print(f"读取训练数据: {f}")
        with open(f, 'r', encoding='utf-8') as fp:
            file_data = json.load(fp)
            train_data.extend(file_data)
            print(f"  读取到 {len(file_data)} 条")

    # Read validation data
    val_data = []
    for f in val_files:
        print(f"读取验证数据: {f}")
        with open(f, 'r', encoding='utf-8') as fp:
            file_data = json.load(fp)
            val_data.extend(file_data)
            print(f"  读取到 {len(file_data)} 条")

    print(f"\n原始数据: 训练 {len(train_data)} 条, 验证 {len(val_data)} 条")

    # Deduplication (before multi-task expansion)
    if remove_duplicates:
        print("\n执行去重...")
        train_data = remove_duplicates_by_visit_number(train_data)
        val_data = remove_duplicates_by_visit_number(val_data)
        print(f"去重后: 训练 {len(train_data)} 条, 验证 {len(val_data)} 条")

    # Create map functions
    train_map_fn = make_map_fn('train', task_types)
    val_map_fn = make_map_fn('val', task_types)

    # Convert training data
    print("\n转换训练数据...")
    converted_train = []
    for idx, item in enumerate(train_data):
        samples = train_map_fn(item, idx)
        converted_train.extend(samples)

    # Convert validation data
    print("转换验证数据...")
    converted_val = []
    for idx, item in enumerate(val_data):
        samples = val_map_fn(item, idx)
        converted_val.extend(samples)

    # Print statistics
    print_dataset_statistics(converted_train, "训练")
    print_dataset_statistics(converted_val, "验证")

    # Save to parquet
    os.makedirs(output_dir, exist_ok=True)

    train_df = pd.DataFrame(converted_train)
    val_df = pd.DataFrame(converted_val)

    train_path = os.path.join(output_dir, 'train.parquet')
    val_path = os.path.join(output_dir, 'val.parquet')

    train_df.to_parquet(train_path, index=False)
    val_df.to_parquet(val_path, index=False)

    print(f"\n{'='*60}")
    print(f"输出完成:")
    print(f"  训练: {train_path} ({len(converted_train)} 条)")
    print(f"  验证: {val_path} ({len(converted_val)} 条)")
    print(f"{'='*60}")

    # Show sample data
    print("\n样例数据 (训练集前3条, 每种任务各1条):")
    shown_tasks = set()
    for item in converted_train:
        task_type = item['extra_info']['task_type']
        if task_type not in shown_tasks:
            shown_tasks.add(task_type)
            print(f"\n--- {task_type} ---")
            print(json.dumps(
                {k: v for k, v in item.items() if k != 'prompt'},
                ensure_ascii=False, indent=2
            ))
            # Show prompt summary
            print(f"  prompt[0] (system): {item['prompt'][0]['content'][:80]}...")
            print(f"  prompt[1] (user): ...{item['prompt'][1]['content'][-80:]}...")
        if len(shown_tasks) == len(task_types):
            break

    return train_df, val_df


# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='v8: ShareGPT to Parquet for mixed RL training (binary + multiclass + recommendation)'
    )
    parser.add_argument(
        '--train_files', nargs='+',
        default=[
            "/tcci_mnt/shihao/project/LLaMA-Factory/shihao/data/mirodiag-16k-kimi-k2-0905-rl-train-sharegpt.json",
        ],
        help='训练数据文件路径列表'
    )
    parser.add_argument(
        '--val_files', nargs='+',
        default=[
            "/tcci_mnt/shihao/project/LLaMA-Factory/shihao/data/mirodiag-16k-kimi-k2-0905-validation-sharegpt.json",
        ],
        help='验证数据文件路径列表'
    )
    parser.add_argument(
        '--output_dir', type=str,
        default="/tcci_mnt/shihao/project/verl/psy_r1/SMHC_data_v8",
        help='输出目录'
    )
    parser.add_argument(
        '--task_types', nargs='+',
        default=TASK_TYPES,
        choices=TASK_TYPES,
        help='要生成的任务类型'
    )
    parser.add_argument(
        '--remove_duplicates', action=argparse.BooleanOptionalAction, default=True,
        help='根据visit_number去重（默认开启，RL数据同一visit_number有多条rollout，GRPO只需唯一prompt；用 --no-remove-duplicates 关闭）'
    )
    return parser.parse_args()


def main():
    args = parse_args()
    convert_sharegpt_to_parquet(
        train_files=args.train_files,
        val_files=args.val_files,
        output_dir=args.output_dir,
        task_types=args.task_types,
        remove_duplicates=args.remove_duplicates,
    )


if __name__ == "__main__":
    main()
