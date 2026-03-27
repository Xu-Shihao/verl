#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v8 混合RL奖励函数
支持 binary (2分类)、multiclass (4分类)、recommendation (ICD-10推荐) 三种任务的统一奖励计算。

根据 data_source 字段自动分发到对应的奖励计算逻辑：
- SMHC_RL_binary        → 提取<box>中的"抑郁"/"焦虑"，与ground_truth比较
- SMHC_RL_multiclass    → 提取<box>中的"抑郁"/"焦虑"/"mix"/"others"，与ground_truth比较
- SMHC_RL_recommendation → 使用现有的 calculate_icd_reward
"""

import re
import torch
import random
import numpy as np
from collections import Counter

# 导入现有的ICD奖励计算
from psy_r1.verl.utils.dapo_reward_score_psy import calculate_icd_reward


# ============================================================
# Binary / Multiclass 奖励计算
# ============================================================

def extract_box_content(response_text):
    """从模型响应中提取最后一个<box>...</box>中的内容"""
    if not response_text:
        return None
    pattern = r'<box>(.*?)</box>'
    matches = list(re.finditer(pattern, response_text, re.DOTALL))
    if matches:
        return matches[-1].group(1).strip()
    return None


def calculate_binary_reward(solution_str, ground_truth, extra_info=None, debug=False):
    """
    计算二分类任务的奖励: 抑郁 vs 焦虑

    ground_truth: ["抑郁"] 或 ["焦虑"]（列表格式，来自parquet）
    """
    # 解析ground_truth
    if isinstance(ground_truth, (list, np.ndarray)):
        gt_label = str(ground_truth[0]) if len(ground_truth) > 0 else ""
    else:
        gt_label = str(ground_truth)

    # 提取<box>内容
    predicted = extract_box_content(solution_str)
    has_box = predicted is not None

    if predicted is None:
        predicted = ""

    # 标准化预测结果
    predicted_clean = predicted.strip()

    # 格式分数: 是否有<box>标签且内容是有效的分类结果
    valid_labels = {"抑郁", "焦虑"}
    format_score = 1.0 if (has_box and predicted_clean in valid_labels) else 0.0

    # 准确性: 预测是否与ground_truth匹配
    exact_match = 1.0 if predicted_clean == gt_label else 0.0

    # 总分: format_score * 0.2 + format_score * exact_match * 0.8 (与ICD奖励保持一致)
    total_score = format_score * 0.2 + format_score * exact_match * 0.8

    result = {
        "score": total_score,
        "exact_match": exact_match,
        "format_score": format_score,
        "predicted": predicted_clean,
        "ground_truth_label": gt_label,
        "acc": exact_match > 0,
        "data_source": "binary",
        "task_type": "binary",
    }

    if debug:
        print(f"[Binary] predicted='{predicted_clean}', gt='{gt_label}', score={total_score}")

    return result


def calculate_multiclass_reward(solution_str, ground_truth, extra_info=None, debug=False):
    """
    计算四分类任务的奖励: 抑郁 / 焦虑 / mix / others

    ground_truth: ["抑郁"], ["焦虑"], ["mix"], 或 ["others"]
    """
    # 解析ground_truth
    if isinstance(ground_truth, (list, np.ndarray)):
        gt_label = str(ground_truth[0]) if len(ground_truth) > 0 else ""
    else:
        gt_label = str(ground_truth)

    # 提取<box>内容
    predicted = extract_box_content(solution_str)
    has_box = predicted is not None

    if predicted is None:
        predicted = ""

    # 标准化预测结果
    predicted_clean = predicted.strip().lower()
    gt_clean = gt_label.strip().lower()

    # 处理中文/英文等价 (抑郁 = depression, 焦虑 = anxiety)
    label_mapping = {
        "抑郁": "抑郁", "depression": "抑郁", "抑郁症": "抑郁",
        "焦虑": "焦虑", "anxiety": "焦虑", "焦虑症": "焦虑",
        "mix": "mix", "混合": "mix", "mixed": "mix",
        "others": "others", "其他": "others", "other": "others",
    }

    predicted_normalized = label_mapping.get(predicted_clean, predicted_clean)
    gt_normalized = label_mapping.get(gt_clean, gt_clean)

    # 格式分数: 是否有<box>标签且内容是有效的分类结果
    valid_labels = {"抑郁", "焦虑", "mix", "others"}
    format_score = 1.0 if (has_box and predicted_normalized in valid_labels) else 0.0

    # 准确性
    exact_match = 1.0 if predicted_normalized == gt_normalized else 0.0

    # 总分
    total_score = format_score * 0.2 + format_score * exact_match * 0.8

    result = {
        "score": total_score,
        "exact_match": exact_match,
        "format_score": format_score,
        "predicted": predicted_normalized,
        "ground_truth_label": gt_normalized,
        "acc": exact_match > 0,
        "data_source": "multiclass",
        "task_type": "multiclass",
    }

    if debug:
        print(f"[Multiclass] predicted='{predicted_normalized}', gt='{gt_normalized}', score={total_score}")

    return result


# ============================================================
# 统一分发函数 (DAPO compute_score 接口)
# ============================================================

def compute_score_v8(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    """
    v8 混合RL的统一奖励计算函数。

    按照 data_source 分发到不同的奖励计算逻辑。
    兼容 DAPO reward manager 的 compute_score 接口:
        compute_score(data_source, solution_str, ground_truth, extra_info)

    Args:
        data_source: 数据源标识 ("SMHC_RL_binary", "SMHC_RL_multiclass", "SMHC_RL_recommendation", 等)
        solution_str: 模型生成的响应文本
        ground_truth: ground truth (列表或字符串)
        extra_info: 额外信息字典

    Returns:
        dict: 包含 "score", "acc", "exact_match", "format_score" 等字段
    """
    if "binary" in str(data_source).lower():
        return calculate_binary_reward(solution_str, ground_truth, extra_info)
    elif "multiclass" in str(data_source).lower():
        return calculate_multiclass_reward(solution_str, ground_truth, extra_info)
    elif "recommendation" in str(data_source).lower():
        return calculate_icd_reward(solution_str, ground_truth, extra_info)
    else:
        # 兜底: 尝试从 extra_info 中获取 task_type
        task_type = None
        if isinstance(extra_info, dict):
            task_type = extra_info.get("task_type", None)

        if task_type == "binary":
            return calculate_binary_reward(solution_str, ground_truth, extra_info)
        elif task_type == "multiclass":
            return calculate_multiclass_reward(solution_str, ground_truth, extra_info)
        elif task_type == "recommendation":
            return calculate_icd_reward(solution_str, ground_truth, extra_info)
        else:
            # 最终兜底: 使用ICD奖励
            print(f"[WARNING] Unknown data_source='{data_source}', falling back to ICD reward")
            return calculate_icd_reward(solution_str, ground_truth, extra_info)


# ============================================================
# GRPO reward wrapper (批量接口, 用于 main_ppo_psy.py 风格的训练器)
# ============================================================

def create_v8_reward_fn(is_validation=None, tokenizer=None, task_reward_weights=None):
    """
    创建 v8 混合RL的奖励函数 (GRPO批量接口)

    Args:
        is_validation: "val"=验证模式, "train"=训练模式, None=无日志
        tokenizer: 用于解码token的tokenizer
        task_reward_weights: 任务级reward权重字典，如 {"binary": 1.0, "multiclass": 0.5, "recommendation": 1.5}

    Returns:
        reward function compatible with VERL RayPPOTrainer
    """
    if task_reward_weights is None:
        task_reward_weights = {"binary": 1.0, "multiclass": 1.0, "recommendation": 1.0}

    printed_first_example = [False]

    # Rollout统计
    rollout_stats = {
        'total_samples': 0,
        'batch_count': 0,
        'by_task': {
            'binary': {'total': 0, 'correct': 0, 'format_ok': 0, 'total_score': 0.0},
            'multiclass': {'total': 0, 'correct': 0, 'format_ok': 0, 'total_score': 0.0},
            'recommendation': {'total': 0, 'correct': 0, 'format_ok': 0, 'total_score': 0.0},
        }
    }

    def v8_reward_wrapper(data, return_dict=False):
        batch_size = len(data.batch["responses"])
        device = data.batch["responses"].device
        responses = data.batch["responses"]

        prompts = None
        if "prompts" in data.batch:
            prompts = data.batch["prompts"]
        elif "input_ids" in data.batch:
            prompts = data.batch["input_ids"]

        # 计算有效响应长度
        attention_mask = data.batch.get("attention_mask", None)
        if attention_mask is not None:
            prompt_length = prompts.shape[1] if prompts is not None else 0
            valid_response_lengths = attention_mask[:, prompt_length:].sum(dim=-1)
        else:
            valid_response_lengths = torch.full((batch_size,), responses.shape[1], device=responses.device)

        reward_tensor = torch.zeros_like(responses, dtype=torch.float32)

        # 收集指标
        psy_metrics = {
            "diagnosis_accuracy": [],
            "format_accuracy": [],
            "total_score": [],
        }

        random_sample_idx = random.randint(0, batch_size - 1) if batch_size > 0 else 0

        for i in range(batch_size):
            # 解码响应
            if tokenizer:
                valid_len = valid_response_lengths[i].item()
                valid_response_ids = responses[i][:valid_len]
                response_text = tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            else:
                response_text = str(responses[i])

            # 获取ground truth和data_source
            try:
                data_item = data[i]
                reward_model_info = data_item.non_tensor_batch.get("reward_model", {})
                ground_truth = reward_model_info.get("ground_truth", []) if isinstance(reward_model_info, dict) else []
                data_source = data_item.non_tensor_batch.get("data_source", "unknown")
                extra_info = data_item.non_tensor_batch.get("extra_info", {})
            except Exception:
                ground_truth = []
                data_source = "unknown"
                extra_info = {}

            # 计算奖励
            try:
                result = compute_score_v8(
                    data_source=data_source,
                    solution_str=response_text,
                    ground_truth=ground_truth,
                    extra_info=extra_info,
                )
                score = result.get("score", 0.0) if isinstance(result, dict) else float(result)
                is_correct = result.get("acc", False) if isinstance(result, dict) else False
                format_ok = result.get("format_score", 0.0) > 0 if isinstance(result, dict) else False

                # 应用任务级reward权重
                task_type_for_weight = extra_info.get("task_type", "recommendation") if isinstance(extra_info, dict) else "recommendation"
                weight = task_reward_weights.get(task_type_for_weight, 1.0)
                score = score * weight
            except Exception as e:
                if is_validation:
                    print(f"[v8 Reward Error] {e}")
                score = 0.0
                result = {"score": 0.0, "error": str(e)}
                is_correct = False
                format_ok = False

            # 放置奖励到最后一个有效token位置
            valid_len = valid_response_lengths[i].item()
            if valid_len > 0:
                reward_tensor[i, valid_len - 1] = score

            # 收集指标
            psy_metrics["diagnosis_accuracy"].append(1.0 if is_correct else 0.0)
            psy_metrics["format_accuracy"].append(1.0 if format_ok else 0.0)
            psy_metrics["total_score"].append(score)

            # 更新统计
            task_type = extra_info.get("task_type", "recommendation") if isinstance(extra_info, dict) else "recommendation"
            if task_type in rollout_stats['by_task']:
                rollout_stats['by_task'][task_type]['total'] += 1
                if is_correct:
                    rollout_stats['by_task'][task_type]['correct'] += 1
                if format_ok:
                    rollout_stats['by_task'][task_type]['format_ok'] += 1
                rollout_stats['by_task'][task_type]['total_score'] += score

            # 打印第一个样本（仅一次）
            if is_validation and not printed_first_example[0] and i == 0:
                printed_first_example[0] = True
                mode_str = "验证" if is_validation == "val" else "训练"
                print(f"\n{'='*60}")
                print(f"[v8 {mode_str}] 第一个样本详情:")
                print(f"  data_source: {data_source}")
                print(f"  task_type: {task_type}")
                print(f"  ground_truth: {ground_truth}")
                print(f"  response (last 200): ...{response_text[-200:]}")
                print(f"  result: {result}")
                print(f"{'='*60}\n")

            # 随机样本输出
            if is_validation and i == random_sample_idx:
                mode_str = "验证" if is_validation == "val" else "训练"
                print(f"[v8 {mode_str} Sample] task={task_type}, score={score:.3f}, "
                      f"acc={is_correct}, format={format_ok}, gt={ground_truth}")

        # 更新全局统计
        rollout_stats['total_samples'] += batch_size
        rollout_stats['batch_count'] += 1

        # 每10个batch输出一次统计
        if is_validation and rollout_stats['batch_count'] % 10 == 0:
            mode_str = "验证" if is_validation == "val" else "训练"
            print(f"\n[v8 {mode_str} Stats] batch #{rollout_stats['batch_count']}, "
                  f"total_samples={rollout_stats['total_samples']}")
            for task, stats in rollout_stats['by_task'].items():
                if stats['total'] > 0:
                    acc = stats['correct'] / stats['total']
                    fmt = stats['format_ok'] / stats['total']
                    avg_score = stats['total_score'] / stats['total']
                    print(f"  {task}: n={stats['total']}, acc={acc:.3f}, format={fmt:.3f}, avg_score={avg_score:.3f}")

        if return_dict:
            # 将 psy_metrics 中的每个列表单独作为 key 添加（与 dapo_reward_score_psy.py 保持一致）
            extra_info = {}
            for metric_name, metric_values in psy_metrics.items():
                if metric_values:  # 只有当有数据时才添加
                    extra_info[f"psy_{metric_name}"] = metric_values
            # 添加 acc 以便与其他代码兼容
            if psy_metrics["diagnosis_accuracy"]:
                extra_info["acc"] = psy_metrics["diagnosis_accuracy"]
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": extra_info,
            }
        return reward_tensor

    return v8_reward_wrapper
