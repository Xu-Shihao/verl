#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
心理诊断任务的奖励计算函数 - DAPO版本
基于GRPO的psy_reward_function，添加DAPO专用的rollout详细输出功能
"""

import sys
import os
import torch
import numpy as np
import random

# 添加路径以便导入
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# 导入GRPO中的奖励函数
from psy_rewards import psy_reward_function


def create_dapo_psy_reward_fn(is_validation=None, tokenizer=None, use_symptom_reward=False, symptom_alpha=0.1):
    """
    创建DAPO专用的心理诊断奖励函数，包含详细的rollout输出功能
    
    Args:
        is_validation: 模式控制 - "val" 为验证模式, "train" 为训练模式, None 为无日志模式
        tokenizer: 用于解码token的tokenizer
        use_symptom_reward: 是否启用症状识别奖励
        symptom_alpha: 症状F1分数在奖励函数中的权重
    Returns:
        与VERL奖励管理器接口兼容的奖励函数
    """
    
    # 跟踪是否已打印第一个验证示例
    printed_validation_example = [False]
    # 跟踪是否已打印第一个训练示例  
    printed_training_example = [False]
    
    # Rollout统计信息跟踪
    rollout_stats = {
        'total_samples': 0,
        'correct_samples': 0,
        'format_correct_samples': 0,
        'total_score': 0.0,
        'total_symptom_coverage': 0.0,  # 累计症状F1分数
        'symptom_samples': 0,  # 有症状数据的样本数
        'rollout_count': 0,
        'batch_count': 0
    }
    
    def dapo_psy_reward_wrapper(data, return_dict=False):
        """
        DAPO专用的心理诊断奖励包装函数
        
        Args:
            data: DataProto对象，包含批次数据
            return_dict: 是否返回详细结果字典
            
        Returns:
            reward_tensor 或包含奖励信息的字典
        """
        
        # 从data中提取必要信息
        batch_size = len(data.batch["responses"])
        device = data.batch["responses"].device
        
        # 获取responses和ground truth
        responses = data.batch["responses"]
        
        # 尝试从data中获取prompts
        prompts = None
        if "prompts" in data.batch:
            prompts = data.batch["prompts"]
        elif "input_ids" in data.batch:
            prompts = data.batch["input_ids"]
        
        # 初始化奖励张量
        reward_scores = []
        extra_info = {"reward": []}
        
        # 收集验证阶段的详细指标信息
        psy_metrics = {
            "diagnosis_accuracy": [],
            "symptom_accuracy": [],
            "format_accuracy": [],
            "total_score": []
        }
        
        # 批次统计信息
        batch_correct = 0
        batch_format_correct = 0
        batch_total_score = 0.0
        batch_symptom_coverage = 0.0  # 当前批次症状F1分数累计
        batch_symptom_samples = 0     # 当前批次有症状数据的样本数
        
        # 随机选择一个样本用于详细输出（每个批次都输出）
        random_sample_idx = random.randint(0, batch_size - 1) if batch_size > 0 else 0
        sample_details = []  # 存储所有样本的详细信息，用于随机输出
        
        # 计算有效响应长度（使用attention mask）
        attention_mask = data.batch.get("attention_mask", None)
        if attention_mask is not None:
            prompt_length = prompts.shape[1] if prompts is not None else 0
            valid_response_lengths = attention_mask[:, prompt_length:].sum(dim=-1)
        else:
            # 兜底：假设所有响应都有效
            valid_response_lengths = torch.full((batch_size,), responses.shape[1], device=responses.device)
        
        for i in range(batch_size):
            
            # 解码响应为文本（仅有效部分）
            if tokenizer:
                valid_len = valid_response_lengths[i].item()
                valid_response_ids = responses[i][:valid_len]
                response_text = tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            else:
                response_text = str(responses[i])
            
            # 解码prompt为文本（如果可用）
            prompt_text = ""
            if prompts is not None and tokenizer:
                try:
                    # 对于prompts，我们也需要考虑有效长度
                    if attention_mask is not None:
                        prompt_length = prompts.shape[1]
                        valid_prompt_length = attention_mask[i, :prompt_length].sum().item()
                        valid_prompt_ids = prompts[i][-valid_prompt_length:] if valid_prompt_length > 0 else prompts[i]
                        prompt_text = tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
                    else:
                        prompt_text = tokenizer.decode(prompts[i], skip_special_tokens=True)
                except:
                    prompt_text = "无法解码prompt"
            
            # 获取该样本的个别数据项
            try:
                data_item = data[i]  # 获取第i个样本的数据
                reward_model_info = data_item.non_tensor_batch.get("reward_model", {})
                ground_truth = "未知诊断"
                
                # 调试信息用于ground truth提取（仅在启用日志时显示）
                if is_validation is not None and i == 0:
                    if (is_validation == "val" and not printed_validation_example[0]) or \
                       (is_validation == "train" and not printed_training_example[0]):
                        mode_str = "验证" if is_validation == "val" else "训练"
                        print(f"[DEBUG DAPO {mode_str}] data_item.non_tensor_batch keys: {list(data_item.non_tensor_batch.keys())}")
                        print(f"[DEBUG DAPO {mode_str}] reward_model_info type: {type(reward_model_info)}")
                        print(f"[DEBUG DAPO {mode_str}] reward_model_info: {reward_model_info}")
                
                if isinstance(reward_model_info, dict) and "ground_truth" in reward_model_info:
                    ground_truth = reward_model_info["ground_truth"]
                    if is_validation is not None and i == 0:
                        if (is_validation == "val" and not printed_validation_example[0]) or \
                           (is_validation == "train" and not printed_training_example[0]):
                            mode_str = "验证" if is_validation == "val" else "训练"
                            print(f"[DEBUG DAPO {mode_str}] Ground truth from data_item reward_model: {ground_truth}")
                else:
                    # 兜底到原始的批次级别访问
                    batch_reward_model = data.non_tensor_batch.get("reward_model", {})
                    if isinstance(batch_reward_model, list) and len(batch_reward_model) > i:
                        ground_truth = batch_reward_model[i].get("ground_truth", "未知诊断")
                    elif isinstance(batch_reward_model, dict):
                        ground_truth = batch_reward_model.get("ground_truth", "未知诊断")
                    
                    if is_validation is not None and i == 0:
                        if (is_validation == "val" and not printed_validation_example[0]) or \
                           (is_validation == "train" and not printed_training_example[0]):
                            mode_str = "验证" if is_validation == "val" else "训练"
                            print(f"[DEBUG DAPO {mode_str}] Ground truth from batch fallback: {ground_truth}")
                        
            except Exception as e:
                # 如果data[i]访问失败，兜底到原始方法
                if is_validation is not None and i == 0:
                    if (is_validation == "val" and not printed_validation_example[0]) or \
                       (is_validation == "train" and not printed_training_example[0]):
                        mode_str = "验证" if is_validation == "val" else "训练"
                        print(f"[DEBUG DAPO {mode_str}] Failed to access data[{i}]: {e}")
                    
                reward_model_info = data.non_tensor_batch.get("reward_model", {})
                ground_truth = "未知诊断"
                
                if isinstance(reward_model_info, list) and len(reward_model_info) > i:
                    ground_truth = reward_model_info[i].get("ground_truth", "未知诊断")
                elif isinstance(reward_model_info, dict):
                    ground_truth = reward_model_info.get("ground_truth", "未知诊断")
                
                if is_validation is not None and i == 0:
                    if (is_validation == "val" and not printed_validation_example[0]) or \
                       (is_validation == "train" and not printed_training_example[0]):
                        mode_str = "验证" if is_validation == "val" else "训练"
                        print(f"[DEBUG DAPO {mode_str}] Ground truth from exception fallback: {ground_truth}")
            
            # 计算心理诊断奖励分数
            try:
                # 从reward_model_info或extra_info中提取patient_id
                patient_id = None
                if isinstance(reward_model_info, dict):
                    patient_id = reward_model_info.get("patient_id")
                
                # 如果在reward_model_info中没找到，尝试extra_info
                if patient_id is None:
                    try:
                        extra_info_data = data_item.non_tensor_batch.get("extra_info", {})
                        if isinstance(extra_info_data, dict):
                            patient_id = extra_info_data.get("patient_id")
                            
                        # patient_id提取的调试信息（仅在启用日志时显示）
                        if is_validation is not None and i == 0:
                            if (is_validation == "val" and not printed_validation_example[0]) or \
                               (is_validation == "train" and not printed_training_example[0]):
                                mode_str = "验证" if is_validation == "val" else "训练"
                                print(f"[DEBUG DAPO {mode_str}] extra_info_data: {extra_info_data}")
                                print(f"[DEBUG DAPO {mode_str}] Extracted patient_id: {patient_id}")
                    except Exception as e:
                        if is_validation is not None and i == 0:
                            if (is_validation == "val" and not printed_validation_example[0]) or \
                               (is_validation == "train" and not printed_training_example[0]):
                                mode_str = "验证" if is_validation == "val" else "训练"
                                print(f"[DEBUG DAPO {mode_str}] Error extracting patient_id: {e}")
                
                # 使用psy_reward_function（如果可用）
                if psy_reward_function is not None:
                    # 为psy_reward_function准备extra_info
                    extra_info_for_reward = {"patient_id": patient_id} if patient_id else None
                    result = psy_reward_function(
                        data_source="psy_diagnosis",
                        solution_str=response_text,
                        ground_truth=ground_truth,
                        extra_info=extra_info_for_reward,
                        use_symptom_reward=use_symptom_reward,
                        symptom_alpha=symptom_alpha,
                        tracker=None  # 不传递tracker，避免个别样本记录，只在批次级别记录
                    )
                else:
                    # 兜底到原始函数
                    result = register_psy_reward_function(
                        data_source="psy_diagnosis",
                        solution_str=response_text,
                        ground_truth=ground_truth,
                        extra_info={"patient_id": patient_id} if patient_id else None,
                        use_symptom_reward=use_symptom_reward,
                        symptom_alpha=symptom_alpha
                    )
                    
                if isinstance(result, dict):
                    score = result.get("score", 0.0)
                    is_correct = result.get('acc', False)
                    format_ok = result.get('format_score', 0.0) > 0
                else:
                    score = float(result)
                    is_correct = False
                    format_ok = False
                    
            except Exception as e:
                if is_validation is not None:
                    print(f"Error computing DAPO PSY reward: {e}")
                score = 0.0
                result = {"score": 0.0, "error": str(e)}
                is_correct = False
                format_ok = False
            
            # 收集验证阶段的详细指标（用于返回给_validate方法）
            if isinstance(result, dict):
                psy_metrics["diagnosis_accuracy"].append(result.get("diagnosis_score", 0.0))
                psy_metrics["symptom_accuracy"].append(result.get("symptom_f1", 0.0))
                psy_metrics["format_accuracy"].append(result.get("format_score", 0.0))
                psy_metrics["total_score"].append(score)
            
            # 更新批次统计
            batch_total_score += score
            if is_correct:
                batch_correct += 1
            if format_ok:
                batch_format_correct += 1
            
            # 更新症状统计（始终计算，不管是否启用症状奖励）
            if isinstance(result, dict):
                symptom_f1 = result.get('symptom_f1', 0.0)
                # 只有当有有效的症状数据时才统计（避免None或错误情况）
                if 'error' not in result and patient_id is not None:
                    batch_symptom_coverage += symptom_f1  # 使用F1分数累计
                    batch_symptom_samples += 1
            
            # 打印验证示例（第一个样本）
            if is_validation == "val" and not printed_validation_example[0] and i == 0:
                print("\n" + "="*80)
                print("【DAPO验证阶段奖励计算示例】")
                print("="*80)
                
                if isinstance(result, dict):
                    extracted = result.get('extracted_diagnosis', 'N/A')
                    diagnosis_score = result.get('diagnosis_score', 0.0)
                    symptom_f1 = result.get('symptom_f1', 0.0)
                    format_score = result.get('format_score', 0.0)
                    
                    print(f"提取的诊断: {extracted}")
                    print(f"Ground Truth诊断: {ground_truth}")
                    print(f"Diagnosis Accuracy (F1): {diagnosis_score:.3f}")
                    print(f"Symptom Accuracy (F1): {symptom_f1:.3f}")
                    print(f"Format Accuracy: {format_score:.3f}")
                    print(f"Total Score: {score:.3f}")
                    
                    # 显示症状confusion metrics（一行显示）
                    if 'true_positive' in result:
                        tp = result.get('true_positive', 0)
                        fp = result.get('false_positive', 0)
                        tn = result.get('true_negative', 0)
                        fn = result.get('false_negative', 0)
                        print(f"Symptom Confusion: TP={tp}, FP={fp}, TN={tn}, FN={fn}")
                    
                    if 'error' in result:
                        print(f"计算错误: {result['error']}")
                
                print("="*80 + "\n")
                printed_validation_example[0] = True
            
            # 打印训练示例（第一个样本）
            elif is_validation == "train" and not printed_training_example[0] and i == 0:
                print("\n" + "="*80)
                print("【DAPO训练阶段奖励计算示例】")
                print("="*80)
                
                if isinstance(result, dict):
                    extracted = result.get('extracted_diagnosis', 'N/A')
                    diagnosis_score = result.get('diagnosis_score', 0.0)
                    symptom_f1 = result.get('symptom_f1', 0.0)
                    format_score = result.get('format_score', 0.0)
                    
                    print(f"提取的诊断: {extracted}")
                    print(f"Ground Truth诊断: {ground_truth}")
                    print(f"Diagnosis Accuracy (F1): {diagnosis_score:.3f}")
                    print(f"Symptom Accuracy (F1): {symptom_f1:.3f}")
                    print(f"Format Accuracy: {format_score:.3f}")
                    print(f"Total Score: {score:.3f}")
                    
                    # 显示症状confusion metrics（一行显示）
                    if 'true_positive' in result:
                        tp = result.get('true_positive', 0)
                        fp = result.get('false_positive', 0)
                        tn = result.get('true_negative', 0)
                        fn = result.get('false_negative', 0)
                        print(f"Symptom Confusion: TP={tp}, FP={fp}, TN={tn}, FN={fn}")
                    
                    if 'error' in result:
                        print(f"计算错误: {result['error']}")
                
                print("="*80 + "\n")
                printed_training_example[0] = True
            
            reward_scores.append(score)
            extra_info["reward"].append(score)
            
            # 收集样本详细信息用于随机输出
            sample_info = {
                'index': i,
                'prompt_text': prompt_text,
                'response_text': response_text,
                'ground_truth': ground_truth,
                'result': result,
                'score': score,
                'is_correct': is_correct,
                'format_ok': format_ok,
                'patient_id': patient_id
            }
            sample_details.append(sample_info)
        
        # 更新全局rollout统计（仅在训练模式且启用日志时）
        if is_validation == "train":
            rollout_stats['total_samples'] += batch_size
            rollout_stats['correct_samples'] += batch_correct
            rollout_stats['format_correct_samples'] += batch_format_correct
            rollout_stats['total_score'] += batch_total_score
            rollout_stats['total_symptom_coverage'] += batch_symptom_coverage
            rollout_stats['symptom_samples'] += batch_symptom_samples
            rollout_stats['batch_count'] += 1
            rollout_stats['rollout_count'] += 1
            
            # 每隔几个批次打印rollout统计
            if rollout_stats['batch_count'] % 10 == 0 or rollout_stats['batch_count'] <= 5:
                diagnosis_accuracy = rollout_stats['correct_samples'] / rollout_stats['total_samples'] * 100
                format_accuracy = rollout_stats['format_correct_samples'] / rollout_stats['total_samples'] * 100
                avg_total_score = rollout_stats['total_score'] / rollout_stats['total_samples']
                
                # 计算症状F1分数
                symptom_accuracy = 0.0
                if rollout_stats['symptom_samples'] > 0:
                    symptom_accuracy = rollout_stats['total_symptom_coverage'] / rollout_stats['symptom_samples'] * 100
                
                print("\n" + "-"*60)
                print(f"【DAPO Rollout 统计 - Batch {rollout_stats['batch_count']}】")
                print(f"样本数: {rollout_stats['total_samples']}")
                print(f"Diagnosis Accuracy (F1): {diagnosis_accuracy:.2f}%")
                print(f"Symptom Accuracy (F1): {symptom_accuracy:.2f}%")
                print(f"Format Accuracy: {format_accuracy:.2f}%")
                print(f"Total Score: {avg_total_score:.4f}")
                print("-"*60 + "\n")
        
        # 每个批次都输出一个随机选择的样本详细信息（训练和验证都输出）
        if is_validation is not None and sample_details:
            random_sample = sample_details[random_sample_idx]
            mode_str = "验证" if is_validation == "val" else "训练"
            
            print("\n" + "~"*80)
            print(f"【随机样本详情 - {mode_str}阶段 - 样本 #{random_sample['index']+1}/{len(sample_details)}】")
            print("~"*80)
            
            # 显示prompt（截断过长的内容）
            prompt_display = random_sample['prompt_text']
            # if len(prompt_display) > 200:
            #     prompt_display = prompt_display[:200] + "..."
            print(f"Prompt: \n{prompt_display}")
            
            # 显示response（截断过长的内容）
            response_display = random_sample['response_text']
            # if len(response_display) > 300:
            #     response_display = response_display[:300] + "..."
            print(f"Response: \n{response_display}")
            
            print(f"Ground Truth: {random_sample['ground_truth']}")
            
            # 显示详细指标
            result = random_sample['result']
            if isinstance(result, dict):
                extracted = result.get('extracted_diagnosis', 'N/A')
                diagnosis_score = result.get('diagnosis_score', 0.0)
                symptom_f1 = result.get('symptom_f1', 0.0)
                format_score = result.get('format_score', 0.0)
                
                print(f"Extracted Diagnosis: {extracted}")
                print(f"Diagnosis Accuracy (F1): {diagnosis_score:.3f}")
                print(f"Symptom Accuracy (F1): {symptom_f1:.3f}")
                print(f"Format Accuracy: {format_score:.3f}")
                print(f"Total Score: {random_sample['score']:.3f}")
                print(f"Is Correct: {random_sample['is_correct']}")
                print(f"Format OK: {random_sample['format_ok']}")
                
                if random_sample['patient_id']:
                    print(f"Patient ID: {random_sample['patient_id']}")
                
                # 显示症状confusion metrics（如果有）
                if 'true_positive' in result:
                    tp = result.get('true_positive', 0)
                    fp = result.get('false_positive', 0)
                    tn = result.get('true_negative', 0)
                    fn = result.get('false_negative', 0)
                    print(f"Symptom Confusion: TP={tp}, FP={fp}, TN={tn}, FN={fn}")
                
                if 'error' in result:
                    print(f"计算错误: {result['error']}")
            else:
                print(f"Score: {random_sample['score']:.3f}")
            
            print("~"*80 + "\n")
        
        # 转换为VERL期望的张量格式
        reward_tensor = torch.tensor(reward_scores, device=device, dtype=torch.float32)
        
        # 扩展到token级别的奖励（通常只是将奖励放在最后一个token）
        response_length = responses.shape[1]
        token_level_rewards = torch.zeros((batch_size, response_length), device=device, dtype=torch.float32)
        
        # 将奖励放在最后一个有效token位置
        if "attention_mask" in data.batch:
            attention_mask = data.batch["attention_mask"]
            prompt_length = attention_mask.shape[1] - response_length
            response_mask = attention_mask[:, prompt_length:]
            
            for i in range(batch_size):
                # 找到最后一个有效token位置
                valid_positions = torch.where(response_mask[i] > 0)[0]
                if len(valid_positions) > 0:
                    last_pos = valid_positions[-1]
                    token_level_rewards[i, last_pos] = reward_tensor[i]
        else:
            # 默认：将奖励放在最后位置
            token_level_rewards[:, -1] = reward_tensor
        
        if return_dict:
            # 准备返回的字典，包含PSY相关的详细指标
            result_dict = {
                "reward_tensor": token_level_rewards,
                "reward_extra_info": extra_info,
            }
            
            # 添加PSY相关的详细指标到reward_extra_info中（但不覆盖reward列表）
            for metric_name, metric_values in psy_metrics.items():
                if metric_values:  # 只有当有数据时才添加
                    # 使用不同的键名避免与reward冲突
                    result_dict["reward_extra_info"][f"psy_{metric_name}"] = metric_values
            
            if psy_metrics["diagnosis_accuracy"]:
                result_dict["reward_extra_info"]["acc"] = psy_metrics["diagnosis_accuracy"]
            
            return result_dict
        else:
            return token_level_rewards
    
    return dapo_psy_reward_wrapper


def register_psy_reward_function(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    """
    心理诊断任务的奖励函数 - DAPO版本
    直接调用GRPO的psy_reward_function，确保完全一致
    """
    print(f"[DAPO] Processing reward for data_source: {data_source}")
    
    # 从kwargs中提取症状奖励相关参数
    use_symptom_reward = kwargs.get('use_symptom_reward', False)
    symptom_alpha = kwargs.get('symptom_alpha', 0.1)
    
    try:
        # 直接使用GRPO中的psy_reward_function，确保完全一致
        result = psy_reward_function(
            data_source=data_source,
            solution_str=solution_str,
            ground_truth=ground_truth,
            extra_info=extra_info,
            use_symptom_reward=use_symptom_reward,
            symptom_alpha=symptom_alpha,
            tracker=None  # DAPO不需要tracker
        )
        
        # 确保结果格式正确
        if isinstance(result, dict):
            # 添加DAPO可能需要的额外信息
            result["data_source"] = data_source
            
            # 确保包含所有必要字段
            if "score" not in result and "total_score" in result:
                result["score"] = result["total_score"]
            
            print(f"[DAPO] Reward computed - Total Score: {result.get('score', 0.0):.3f}, "
                  f"Diagnosis Accuracy: {result.get('diagnosis_score', 0.0):.3f}, "
                  f"Symptom Accuracy: {result.get('symptom_f1', 0.0):.3f}, "
                  f"Format Accuracy: {result.get('format_score', 0.0):.3f}")
                  
            return result
        else:
            # 兜底处理
            return {
                "score": float(result),
                "diagnosis_score": float(result),
                "symptom_coverage": 0.0,
                "acc": bool(result > 0),
                "data_source": data_source
            }
            
    except Exception as e:
        print(f"[DAPO ERROR] Error in register_psy_reward_function: {e}")
        import traceback
        traceback.print_exc()
        return {
            "score": 0.0,
            "diagnosis_score": 0.0,
            "symptom_coverage": 0.0,
            "acc": False,
            "error": str(e),
            "data_source": data_source
        }


if __name__ == "__main__":
    # 测试DAPO奖励函数与GRPO的一致性
    print("Testing DAPO reward function consistency with GRPO...")
    
    test_cases = [
        {
            "data_source": "SMHC_SFT_auxiliary_diagnosis",
            "solution_str": "根据患者的症状表现，包括持续的情绪低落、兴趣减退、精力缺乏等，诊断为抑郁症。<box>抑郁症</box>",
            "ground_truth": "抑郁症",
            "extra_info": {"patient_id": "1_conv0"},
            "use_symptom_reward": False
        },
        {
            "data_source": "SMHC_SFT_auxiliary_diagnosis", 
            "solution_str": "根据患者的症状表现，包括持续的情绪低落、兴趣减退、精力缺乏等，诊断为抑郁症。<box>抑郁症</box>",
            "ground_truth": "抑郁症",
            "extra_info": {"patient_id": "1_conv0"},
            "use_symptom_reward": True,
            "symptom_alpha": 0.1
        }
    ]
    
    for i, test_case in enumerate(test_cases):
        print(f"\nTest Case {i+1}:")
        print(f"  Data Source: {test_case['data_source']}")
        print(f"  Use Symptom Reward: {test_case.get('use_symptom_reward', False)}")
        
        result = register_psy_reward_function(
            data_source=test_case["data_source"],
            solution_str=test_case["solution_str"],
            ground_truth=test_case["ground_truth"],
            extra_info=test_case.get("extra_info"),
            use_symptom_reward=test_case.get("use_symptom_reward", False),
            symptom_alpha=test_case.get("symptom_alpha", 0.1)
        )
        
        print(f"  Total Score: {result.get('score', 0.0):.3f}")
        print(f"  Diagnosis Score: {result.get('diagnosis_score', 0.0):.3f}")
        print(f"  Symptom Coverage: {result.get('symptom_coverage', 0.0):.3f}")
        print(f"  Accuracy: {result.get('acc', False)}")
    
    print("\n✓ DAPO reward function testing completed!")
