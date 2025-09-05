# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
PSY specific PPO trainer for psychological diagnosis scenarios.
"""

import os
import socket

import hydra
import ray
from omegaconf import OmegaConf

from verl.experimental.dataset.sampler import AbstractSampler
from verl.trainer.constants_ppo import PPO_RAY_RUNTIME_ENV
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.trainer.ppo.reward import load_reward_manager
from verl.utils.device import is_cuda_available
from verl.utils.import_utils import load_extern_type
import torch


# 简化版本：直接使用原有的 RayPPOTrainer，不做复杂的集成
# 我们将在 reward 函数中检查是否有可用的 tracker

def calculate_batch_average_metrics(batch_results):
    """
    计算批次中所有样本的平均指标
    
    Args:
        batch_results: 批次中所有样本的结果列表
        
    Returns:
        包含平均指标的字典
    """
    if not batch_results:
        return {}
    
    # 初始化累计值
    total_score = 0.0
    total_diagnosis_score = 0.0
    total_symptom_coverage = 0.0
    total_format_score = 0.0
    total_acc = 0.0
    total_symptom_count = 0
    total_extracted_symptom_count = 0
    valid_samples = 0
    symptom_samples = 0
    
    # 累计所有样本的指标
    for result in batch_results:
        if isinstance(result, dict) and 'error' not in result:
            valid_samples += 1
            
            total_score += result.get('score', 0.0)
            total_diagnosis_score += result.get('diagnosis_score', 0.0)
            total_format_score += result.get('format_score', 0.0)
            total_acc += float(result.get('acc', False))
            
            # 症状相关指标（只有在有症状数据时才统计）
            if 'symptom_coverage' in result and result.get('total_symptoms', 0) > 0:
                total_symptom_coverage += result.get('symptom_coverage', 0.0)
                total_symptom_count += result.get('total_symptoms', 0)
                total_extracted_symptom_count += len(result.get('extracted_symptoms', []))
                symptom_samples += 1
    
    if valid_samples == 0:
        return {}
    
    # 计算平均值
    avg_metrics = {
        'score': total_score / valid_samples,
        'diagnosis_score': total_diagnosis_score / valid_samples,
        'format_score': total_format_score / valid_samples,
        'acc': total_acc / valid_samples,
        'total_symptoms': total_symptom_count / max(symptom_samples, 1),
        'extracted_symptoms_count': total_extracted_symptom_count / max(symptom_samples, 1),
    }
    
    # 症状覆盖率（只有在有症状样本时才计算）
    if symptom_samples > 0:
        avg_metrics['symptom_coverage'] = total_symptom_coverage / symptom_samples
    else:
        avg_metrics['symptom_coverage'] = 0.0
    
    return avg_metrics


def log_batch_metrics_to_tracking(batch_avg_metrics, tracker=None):
    """
    将批次平均指标记录到tracking系统
    
    Args:
        batch_avg_metrics: 批次平均指标字典
        tracker: Tracking实例
    """
    if not batch_avg_metrics or tracker is None:
        return
    
    # 转换为tracking格式的指标
    tracking_metrics = {
        "rewards/total_score": batch_avg_metrics.get('score', 0.0),
        "rewards/diagnosis_accuracy": batch_avg_metrics.get('diagnosis_score', 0.0),
        "rewards/symptom_accuracy": batch_avg_metrics.get('symptom_coverage', 0.0),
        "rewards/format_accuracy": batch_avg_metrics.get('format_score', 0.0),
        "rewards/exact_match": batch_avg_metrics.get('acc', 0.0),
        "symptoms/total_count": batch_avg_metrics.get('total_symptoms', 0.0),
        "symptoms/extracted_count": batch_avg_metrics.get('extracted_symptoms_count', 0.0),
    }
    
    try:
        # 记录到tracking系统
        tracker.log(data=tracking_metrics)
        print(f"[INFO] Logged batch average metrics: {tracking_metrics}")
    except Exception as e:
        print(f"[WARNING] Failed to log batch metrics: {e}")

def create_psy_reward_fn(is_validation=None, tokenizer=None, use_symptom_reward=False, tracker=None, symptom_alpha=0.1):
    """
    Create PSY-specific reward function for psychological diagnosis
    
    Args:
        is_validation: Mode for logging control - "val" for validation, "train" for training, None for no logs
        tokenizer: The tokenizer to use for decoding tokens
        use_symptom_reward: Whether to enable symptom identification reward
        tracker: Tracking instance for logging metrics to verl official tracking system
        symptom_alpha: The alpha value for symptom coverage in the reward function
    Returns:
        A reward function compatible with VERL reward manager interface
    """
    # Import our PSY reward functions
    try:
        from psy_r1.verl.utils.psy_rewards import compute_diagnosis_score, psy_reward_function
    except ImportError:
        # Fallback if the import fails
        from verl.utils.reward_score import default_compute_score as compute_diagnosis_score
        psy_reward_function = None
    
    # Track whether we've printed the first validation example
    printed_validation_example = [False]
    # Track whether we've printed the first training example
    printed_training_example = [False]
    
    # Rollout statistics tracking
    rollout_stats = {
        'total_samples': 0,
        'correct_samples': 0,
        'format_correct_samples': 0,
        'total_score': 0.0,
        'total_symptom_coverage': 0.0,  # 累计症状覆盖率
        'symptom_samples': 0,  # 有症状数据的样本数
        'rollout_count': 0,
        'batch_count': 0
    }
    
    def psy_reward_wrapper(data, return_dict=False, symptom_alpha=0.1):
        """
        Wrapper function that adapts PSY reward function to VERL interface
        
        Args:
            data: DataProto object containing batch data
            return_dict: Whether to return detailed results as dict
            symptom_alpha: The alpha value for symptom coverage in the reward function
            
        Returns:
            reward_tensor or dict with reward information
        """
        
        # Extract necessary information from data
        batch_size = len(data.batch["responses"])
        device = data.batch["responses"].device
        
        # Get responses and ground truth from data
        responses = data.batch["responses"]
        
        # Try to get prompts from data
        prompts = None
        if "prompts" in data.batch:
            prompts = data.batch["prompts"]
        elif "input_ids" in data.batch:
            prompts = data.batch["input_ids"]
        
        # Initialize reward tensors
        reward_scores = []
        extra_info = {"reward": []}
        
        # 收集验证阶段的详细指标信息
        psy_metrics = {
            "diagnosis_accuracy": [],
            "symptom_accuracy": [],
            "format_accuracy": [],
            "exact_match": [],
            "total_symptoms": [],
            "extracted_symptoms_count": []
        }
        
        # Batch statistics for this rollout
        batch_correct = 0
        batch_format_correct = 0
        batch_total_score = 0.0
        batch_symptom_coverage = 0.0  # 当前批次症状覆盖率累计
        batch_symptom_samples = 0     # 当前批次有症状数据的样本数
        
        # 收集批次结果用于计算平均值
        batch_results = []
        
        # Calculate valid response lengths using attention mask
        attention_mask = data.batch.get("attention_mask", None)
        if attention_mask is not None:
            prompt_length = prompts.shape[1] if prompts is not None else 0
            valid_response_lengths = attention_mask[:, prompt_length:].sum(dim=-1)
        else:
            # Fallback: assume all responses are valid
            valid_response_lengths = torch.full((batch_size,), responses.shape[1], device=responses.device)
        
        for i in range(batch_size):
            
            # Decode response to text (only valid part)
            if tokenizer:
                valid_len = valid_response_lengths[i].item()
                valid_response_ids = responses[i][:valid_len]
                response_text = tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            else:
                response_text = str(responses[i])
            
            # Decode prompt to text if available
            prompt_text = ""
            if prompts is not None and tokenizer:
                try:
                    # For prompts, we also need to consider valid length
                    if attention_mask is not None:
                        prompt_length = prompts.shape[1]
                        valid_prompt_length = attention_mask[i, :prompt_length].sum().item()
                        valid_prompt_ids = prompts[i][-valid_prompt_length:] if valid_prompt_length > 0 else prompts[i]
                        prompt_text = tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
                    else:
                        prompt_text = tokenizer.decode(prompts[i], skip_special_tokens=True)
                except:
                    prompt_text = "无法解码prompt"
            
            # Get individual data item for this sample
            try:
                data_item = data[i]  # Get the i-th sample's data
                reward_model_info = data_item.non_tensor_batch.get("reward_model", {})
                ground_truth = "未知诊断"
                
                # Debug information for ground truth extraction (only show when logging is enabled)
                if is_validation is not None and i == 0:
                    if (is_validation == "val" and not printed_validation_example[0]) or \
                       (is_validation == "train" and not printed_training_example[0]):
                        mode_str = "验证" if is_validation == "val" else "训练"
                        print(f"[DEBUG {mode_str}] data_item.non_tensor_batch keys: {list(data_item.non_tensor_batch.keys())}")
                        print(f"[DEBUG {mode_str}] reward_model_info type: {type(reward_model_info)}")
                        print(f"[DEBUG {mode_str}] reward_model_info: {reward_model_info}")
                
                if isinstance(reward_model_info, dict) and "ground_truth" in reward_model_info:
                    ground_truth = reward_model_info["ground_truth"]
                    if is_validation is not None and i == 0:
                        if (is_validation == "val" and not printed_validation_example[0]) or \
                           (is_validation == "train" and not printed_training_example[0]):
                            mode_str = "验证" if is_validation == "val" else "训练"
                            print(f"[DEBUG {mode_str}] Ground truth from data_item reward_model: {ground_truth}")
                else:
                    # Fallback to the original batch-level access
                    batch_reward_model = data.non_tensor_batch.get("reward_model", {})
                    if isinstance(batch_reward_model, list) and len(batch_reward_model) > i:
                        ground_truth = batch_reward_model[i].get("ground_truth", "未知诊断")
                    elif isinstance(batch_reward_model, dict):
                        ground_truth = batch_reward_model.get("ground_truth", "未知诊断")
                    
                    if is_validation is not None and i == 0:
                        if (is_validation == "val" and not printed_validation_example[0]) or \
                           (is_validation == "train" and not printed_training_example[0]):
                            mode_str = "验证" if is_validation == "val" else "训练"
                            print(f"[DEBUG {mode_str}] Ground truth from batch fallback: {ground_truth}")
                        
            except Exception as e:
                # If data[i] access fails, fallback to original method
                if is_validation is not None and i == 0:
                    if (is_validation == "val" and not printed_validation_example[0]) or \
                       (is_validation == "train" and not printed_training_example[0]):
                        mode_str = "验证" if is_validation == "val" else "训练"
                        print(f"[DEBUG {mode_str}] Failed to access data[{i}]: {e}")
                    
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
                        print(f"[DEBUG {mode_str}] Ground truth from exception fallback: {ground_truth}")
            
            # Compute PSY reward score
            try:
                # Extract patient_id from reward_model_info or extra_info
                patient_id = None
                if isinstance(reward_model_info, dict):
                    patient_id = reward_model_info.get("patient_id")
                
                # If not found in reward_model_info, try extra_info
                if patient_id is None:
                    try:
                        extra_info_data = data_item.non_tensor_batch.get("extra_info", {})
                        if isinstance(extra_info_data, dict):
                            patient_id = extra_info_data.get("patient_id")
                            
                        # Debug info for patient_id extraction (only show when logging is enabled)
                        if is_validation is not None and i == 0:
                            if (is_validation == "val" and not printed_validation_example[0]) or \
                               (is_validation == "train" and not printed_training_example[0]):
                                mode_str = "验证" if is_validation == "val" else "训练"
                                print(f"[DEBUG {mode_str}] extra_info_data: {extra_info_data}")
                                print(f"[DEBUG {mode_str}] Extracted patient_id: {patient_id}")
                    except Exception as e:
                        if is_validation is not None and i == 0:
                            if (is_validation == "val" and not printed_validation_example[0]) or \
                               (is_validation == "train" and not printed_training_example[0]):
                                mode_str = "验证" if is_validation == "val" else "训练"
                                print(f"[DEBUG {mode_str}] Error extracting patient_id: {e}")
                
                # Use psy_reward_function if available (with or without tracker)
                if psy_reward_function is not None:
                    # Prepare extra_info for psy_reward_function
                    extra_info_for_reward = {"patient_id": patient_id} if patient_id else None
                    result = psy_reward_function(
                        data_source="psy_diagnosis",
                        solution_str=response_text,
                        ground_truth=ground_truth,
                        extra_info=extra_info_for_reward,
                        use_symptom_reward=use_symptom_reward,
                        symptom_alpha=symptom_alpha,
                        tracker=tracker  # tracker can be None, psy_reward_function handles it
                    )
                else:
                    # Fallback to original function
                    result = compute_diagnosis_score(
                        response_text, 
                        ground_truth, 
                        patient_id=patient_id,
                        return_details=True, 
                        use_symptom_reward=use_symptom_reward
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
                    print(f"Error computing PSY reward: {e}")
                score = 0.0
                result = {"score": 0.0, "error": str(e)}
                is_correct = False
                format_ok = False
            
            # 收集样本结果用于批量平均
            if isinstance(result, dict):
                batch_results.append(result)
                
                # 收集验证阶段的详细指标（用于返回给_validate方法）
                psy_metrics["diagnosis_accuracy"].append(result.get("diagnosis_score", 0.0))
                psy_metrics["symptom_accuracy"].append(result.get("symptom_coverage", 0.0))
                psy_metrics["format_accuracy"].append(result.get("format_score", 0.0))
                psy_metrics["exact_match"].append(float(result.get("acc", False)))
                psy_metrics["total_symptoms"].append(result.get("total_symptoms", 0))
                psy_metrics["extracted_symptoms_count"].append(len(result.get("extracted_symptoms", [])))
            
            # Update batch statistics
            batch_total_score += score
            if is_correct:
                batch_correct += 1
            if format_ok:
                batch_format_correct += 1
            
            # Update symptom statistics (only when symptom reward is enabled)
            if use_symptom_reward and isinstance(result, dict):
                symptom_coverage = result.get('symptom_coverage', 0.0)
                # 只有当有有效的症状数据时才统计（避免None或错误情况）
                if 'error' not in result and patient_id is not None:
                    batch_symptom_coverage += symptom_coverage
                    batch_symptom_samples += 1
            
            # Print validation example for the first sample
            if is_validation == "val" and not printed_validation_example[0] and i == 0:
                print("\n" + "="*80)
                print("【验证阶段奖励计算示例】")
                print("="*80)
                
                # Display prompt (no truncation)
                prompt_display = prompt_text if prompt_text else "无prompt信息"
                print(f"问诊提示:\n{prompt_display}")
                
                print("-" * 40)
                
                # Display response (no truncation)
                response_display = response_text if response_text else "无回复内容"
                print(f"模型回复:\n{response_display}")
                
                print("-" * 40)
                
                # Display evaluation details
                print(f"标准答案: {ground_truth}")
                print(f"奖励分数: {score:.2f}")
                
                if isinstance(result, dict):
                    extracted = result.get('extracted_diagnosis', 'N/A')
                    format_ok = result.get('format_score', 0.0) > 0
                    answer_correct = result.get('acc', False)
                    
                    print(f"提取的诊断: {extracted}")
                    print(f"格式正确: {'是' if format_ok else '否'}")
                    print(f"诊断正确: {'是' if answer_correct else '否'}")
                    
                    # 显示症状相关信息（如果启用了症状奖励）
                    if use_symptom_reward:
                        symptom_coverage = result.get('symptom_coverage', 0.0)
                        extracted_symptoms = result.get('extracted_symptoms', [])
                        total_symptoms = result.get('total_symptoms', 0)
                        diagnosis_score = result.get('diagnosis_score', 0.0)
                        print(f"患者ID: {patient_id if patient_id else 'N/A'}")
                        print(f"诊断分数: {diagnosis_score:.3f}")
                        print(f"症状覆盖率: {symptom_coverage:.3f}")
                        print(f"提取症状数: {len(extracted_symptoms)}/{total_symptoms}")
                        if extracted_symptoms:
                            print(f"提取的症状: {extracted_symptoms}")
                    
                    if 'error' in result:
                        print(f"计算错误: {result['error']}")
                
                print("="*80 + "\n")
                # printed_validation_example[0] = True
            
            # Print training example for the first sample
            elif is_validation == "train" and not printed_training_example[0] and i == 0:
                print("\n" + "="*80)
                print("【训练阶段奖励计算示例】")
                print("="*80)
                
                # Display prompt (no truncation)
                prompt_display = prompt_text if prompt_text else "无prompt信息"
                print(f"问诊提示:\n{prompt_display}")
                
                print("-" * 40)
                
                # Display response (no truncation)
                response_display = response_text if response_text else "无回复内容"
                print(f"模型回复:\n{response_display}")
                
                print("-" * 40)
                
                # Display evaluation details
                print(f"标准答案: {ground_truth}")
                print(f"奖励分数: {score:.2f}")
                
                if isinstance(result, dict):
                    extracted = result.get('extracted_diagnosis', 'N/A')
                    format_ok = result.get('format_score', 0.0) > 0
                    answer_correct = result.get('acc', False)
                    
                    print(f"提取的诊断: {extracted}")
                    print(f"格式正确: {'是' if format_ok else '否'}")
                    print(f"诊断正确: {'是' if answer_correct else '否'}")
                    
                    # 显示症状相关信息（如果启用了症状奖励）
                    if use_symptom_reward:
                        symptom_coverage = result.get('symptom_coverage', 0.0)
                        extracted_symptoms = result.get('extracted_symptoms', [])
                        total_symptoms = result.get('total_symptoms', 0)
                        diagnosis_score = result.get('diagnosis_score', 0.0)
                        print(f"患者ID: {patient_id if patient_id else 'N/A'}")
                        print(f"诊断分数: {diagnosis_score:.3f}")
                        print(f"症状覆盖率: {symptom_coverage:.3f}")
                        print(f"提取症状数: {len(extracted_symptoms)}/{total_symptoms}")
                        if extracted_symptoms:
                            print(f"提取的症状: {extracted_symptoms}")
                    
                    if 'error' in result:
                        print(f"计算错误: {result['error']}")
                
                print("="*80 + "\n")
                # printed_training_example[0] = True
            
            reward_scores.append(score)
            extra_info["reward"].append(score)
        
        # 在批次结束时计算平均指标并记录到tracking系统
        if batch_results and tracker is not None:
            # 计算批次平均指标
            batch_avg_metrics = calculate_batch_average_metrics(batch_results)
            
            # 记录到tracking系统
            log_batch_metrics_to_tracking(batch_avg_metrics, tracker=tracker)
        
        # Update global rollout statistics (only for training mode with logging enabled)
        if is_validation == "train":
            rollout_stats['total_samples'] += batch_size
            rollout_stats['correct_samples'] += batch_correct
            rollout_stats['format_correct_samples'] += batch_format_correct
            rollout_stats['total_score'] += batch_total_score
            rollout_stats['total_symptom_coverage'] += batch_symptom_coverage
            rollout_stats['symptom_samples'] += batch_symptom_samples
            rollout_stats['batch_count'] += 1
            rollout_stats['rollout_count'] += 1
            
            # Print rollout statistics every few batches
            if rollout_stats['batch_count'] % 10 == 0 or rollout_stats['batch_count'] <= 5:
                accuracy = rollout_stats['correct_samples'] / rollout_stats['total_samples'] * 100
                format_accuracy = rollout_stats['format_correct_samples'] / rollout_stats['total_samples'] * 100
                avg_score = rollout_stats['total_score'] / rollout_stats['total_samples']
                
                # 计算症状识别率（仅在启用症状奖励时显示）
                symptom_accuracy = 0.0
                if use_symptom_reward and rollout_stats['symptom_samples'] > 0:
                    symptom_accuracy = rollout_stats['total_symptom_coverage'] / rollout_stats['symptom_samples'] * 100
                
                print("\n" + "-"*60)
                print(f"【GRPO Rollout 统计 - Batch {rollout_stats['batch_count']}】")
                print(f"总样本数: {rollout_stats['total_samples']}")
                print(f"诊断正确数: {rollout_stats['correct_samples']}")
                print(f"格式正确数: {rollout_stats['format_correct_samples']}")
                print(f"诊断正确率: {accuracy:.2f}%")
                print(f"格式正确率: {format_accuracy:.2f}%")
                print(f"平均奖励分数: {avg_score:.4f}")
                
                # 显示症状识别统计（仅在启用症状奖励时）
                if use_symptom_reward:
                    print(f"症状样本数: {rollout_stats['symptom_samples']}")
                    print(f"平均症状识别率: {symptom_accuracy:.2f}%")
                    print(f"当前批次症状样本: {batch_symptom_samples}")
                    if batch_symptom_samples > 0:
                        batch_symptom_accuracy = batch_symptom_coverage / batch_symptom_samples * 100
                        print(f"当前批次症状识别率: {batch_symptom_accuracy:.2f}%")
                
                print(f"当前批次正确率: {batch_correct}/{batch_size} = {batch_correct/batch_size*100:.2f}%")
                print("-"*60 + "\n")
        
        # Convert to tensor format expected by VERL
        reward_tensor = torch.tensor(reward_scores, device=device, dtype=torch.float32)
        
        # Expand to token-level rewards (typically just put the reward at the last token)
        response_length = responses.shape[1]
        token_level_rewards = torch.zeros((batch_size, response_length), device=device, dtype=torch.float32)
        
        # Put the reward at the last valid token position
        if "attention_mask" in data.batch:
            attention_mask = data.batch["attention_mask"]
            prompt_length = attention_mask.shape[1] - response_length
            response_mask = attention_mask[:, prompt_length:]
            
            for i in range(batch_size):
                # Find the last valid token position
                valid_positions = torch.where(response_mask[i] > 0)[0]
                if len(valid_positions) > 0:
                    last_pos = valid_positions[-1]
                    token_level_rewards[i, last_pos] = reward_tensor[i]
        else:
            # Default: put reward at the last position
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
            
            return result_dict
        else:
            return token_level_rewards
    
    return psy_reward_wrapper


@hydra.main(config_path="config", config_name="ppo_trainer_psy", version_base=None)
def main(config):
    """Main entry point for PSY-specific PPO training with Hydra configuration management.

    Args:
        config: Hydra configuration dictionary containing training parameters
                for psychological diagnosis scenarios.
    """
    run_ppo(config)


# Define a function to run the PPO-like training process
def run_ppo(config) -> None:
    """Initialize Ray cluster and run distributed PPO training process.

    Args:
        config: Training configuration object containing all necessary parameters
                for distributed PPO training including Ray initialization settings,
                model paths, and training hyperparameters.
    """
    # Check if Ray is not initialized
    if not ray.is_initialized():
        # Initialize Ray with a local cluster configuration
        # Set environment variables in the runtime environment to control tokenizer parallelism,
        # NCCL debug level, VLLM logging level, and allow runtime LoRA updating
        # `num_cpus` specifies the number of CPU cores Ray can use, obtained from the configuration
        ray.init(
            runtime_env=PPO_RAY_RUNTIME_ENV,
            num_cpus=config.ray_init.num_cpus,
        )

    # Create a remote instance of the TaskRunner class, and
    # Execute the `run` method of the TaskRunner instance remotely and wait for it to complete
    if (
        is_cuda_available
        and config.trainer.get("profile_steps") is not None
        and len(config.trainer.get("profile_steps", [])) > 0
    ):
        nsight_options = OmegaConf.to_container(config.trainer.controller_nsight_options)
        runner = TaskRunner.options(runtime_env={"nsight": nsight_options}).remote()
    else:
        runner = TaskRunner.remote()
    ray.get(runner.run.remote(config))

    # [Optional] get the path of the timeline trace file from the configuration, default to None
    # This file is used for performance analysis
    timeline_json_file = config.ray_init.get("timeline_json_file", None)
    if timeline_json_file:
        ray.timeline(filename=timeline_json_file)


@ray.remote(num_cpus=1)  # please make sure main_task is not scheduled on head
class TaskRunner:
    """Ray remote class for executing distributed PPO training tasks.

    This class encapsulates the main training logic and runs as a Ray remote actor
    to enable distributed execution across multiple nodes and GPUs.
    """
    def run(self, config):
        """Execute the main PPO training workflow.

        This method sets up the distributed training environment, initializes
        workers, datasets, and reward functions, then starts the training process.

        Args:
            config: Training configuration object containing all parameters needed
                   for setting up and running the PPO training process.
        """
        # Print the initial configuration. `resolve=True` will evaluate symbolic values.
        from pprint import pprint

        from omegaconf import OmegaConf

        from verl.utils.fs import copy_to_local

        print(f"TaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        # Download the checkpoint from HDFS to the local machine.
        # `use_shm` determines whether to use shared memory, which could lead to faster model loading if turned on
        local_path = copy_to_local(
            config.actor_rollout_ref.model.path, use_shm=config.actor_rollout_ref.model.get("use_shm", False)
        )

        # Instantiate the tokenizer and processor.
        from verl.utils import hf_processor, hf_tokenizer

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        # Used for multimodal LLM, could be None
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)
        
        # Define worker classes based on the actor strategy.
        if config.actor_rollout_ref.actor.strategy in {"fsdp", "fsdp2"}:
            assert config.critic.strategy in {"fsdp", "fsdp2"}
            from verl.single_controller.ray import RayWorkerGroup
            from verl.workers.fsdp_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker

            use_legacy_worker_impl = config.trainer.get("use_legacy_worker_impl", "auto")
            if use_legacy_worker_impl in ["auto", "enable"]:
                # import warnings
                # warnings.warn(f"Legacy worker impl is going to be deprecated, will be removed in the future. \
                #   Please set trainer.use_legacy_worker_impl = false to switch to the new worker implementation.")
                from verl.workers.fsdp_workers import CriticWorker
            elif use_legacy_worker_impl == "disable":
                from verl.workers.roles import CriticWorker

                print("Using new worker implementation")
            else:
                raise ValueError(f"Invalid use_legacy_worker_impl: {use_legacy_worker_impl}")

            actor_rollout_cls = (
                AsyncActorRolloutRefWorker
                if config.actor_rollout_ref.rollout.mode == "async"
                else ActorRolloutRefWorker
            )
            ray_worker_group_cls = RayWorkerGroup

        elif config.actor_rollout_ref.actor.strategy == "megatron":
            assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
            from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
            from verl.workers.megatron_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker, CriticWorker

            actor_rollout_cls = (
                AsyncActorRolloutRefWorker
                if config.actor_rollout_ref.rollout.mode == "async"
                else ActorRolloutRefWorker
            )
            ray_worker_group_cls = NVMegatronRayWorkerGroup

        else:
            raise NotImplementedError

        from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

        # Map roles to their corresponding remote worker classes.
        role_worker_mapping = {
            Role.ActorRollout: ray.remote(actor_rollout_cls),
            Role.Critic: ray.remote(CriticWorker),
        }

        # Define the resource pool specification.
        # Map roles to the resource pool.
        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        mapping = {
            Role.ActorRollout: global_pool_id,
            Role.Critic: global_pool_id,
        }

        # We should adopt a multi-source reward function here:
        # - for rule-based rm, we directly call a reward score
        # - for model-based rm, we call a model
        # - for code related prompt, we send to a sandbox if there are test cases
        # finally, we combine all the rewards together
        # The reward type depends on the tag of the data
        if config.reward_model.enable:
            if config.reward_model.strategy in {"fsdp", "fsdp2"}:
                from verl.workers.fsdp_workers import RewardModelWorker
            elif config.reward_model.strategy == "megatron":
                from verl.workers.megatron_workers import RewardModelWorker
            else:
                raise NotImplementedError
            role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
            mapping[Role.RewardModel] = global_pool_id

        # Add a reference policy worker if KL loss or KL reward is used.
        if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
            role_worker_mapping[Role.RefPolicy] = ray.remote(ActorRolloutRefWorker)
            mapping[Role.RefPolicy] = global_pool_id

        # Load the reward manager for training and validation.
        # Use PSY-specific reward function for psychological diagnosis
        print("Loading PSY-specific reward function for psychological diagnosis...")
        
        # You can still use the original reward manager as a fallback
        # or combine with PSY rewards based on your needs
        if config.reward_model.get("use_psy_reward", True):
            # Check logging configurations
            show_train_examples = config.reward_model.get("show_training_examples", True)
            show_val_examples = config.reward_model.get("show_validation_examples", True)
            symptom_alpha = config.reward_model.get("symptom_alpha", 0)
            
            # Check symptom reward configuration
            use_symptom_reward = config.reward_model.get("use_symptom_reward", False)
            
            # Determine logging modes based on configuration
            train_log_mode = "train" if show_train_examples else None
            val_log_mode = "val" if show_val_examples else None
            
            # Create separate reward functions for training and validation
            train_psy_reward_fn = create_psy_reward_fn(is_validation=train_log_mode, tokenizer=tokenizer, use_symptom_reward=use_symptom_reward, symptom_alpha=symptom_alpha)
            val_psy_reward_fn = create_psy_reward_fn(is_validation=val_log_mode, tokenizer=tokenizer, use_symptom_reward=use_symptom_reward, symptom_alpha=symptom_alpha)
            
            reward_fn = train_psy_reward_fn
            val_reward_fn = val_psy_reward_fn
            
            # Print configuration status
            train_status = "开启日志" if show_train_examples else "关闭日志"
            val_status = "开启日志" if show_val_examples else "关闭日志"
            symptom_status = "启用症状奖励" if use_symptom_reward else "仅诊断奖励"
            print(f"Using PSY reward function - 训练模式: {train_status}, 验证模式: {val_status}, 奖励模式: {symptom_status}")
        else:
            reward_fn = load_reward_manager(
                config, tokenizer, num_examine=0, **config.reward_model.get("reward_kwargs", {})
            )
            val_reward_fn = load_reward_manager(
                config, tokenizer, num_examine=1, **config.reward_model.get("reward_kwargs", {})
            )
            print("Using default reward manager")
        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

        from verl.utils.dataset.rl_dataset import collate_fn

        # Create training and validation datasets.
        train_dataset = create_rl_dataset(config.data.train_files, config.data, tokenizer, processor, is_train=True)
        val_dataset = create_rl_dataset(config.data.val_files, config.data, tokenizer, processor, is_train=False)
        train_sampler = create_rl_sampler(config.data, train_dataset)

        # Initialize the PPO trainer.
        trainer = RayPPOTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
        )
        # Initialize the workers of the trainer.
        trainer.init_workers()
        # Start the training process.
        trainer.fit()


def create_rl_dataset(data_paths, data_config, tokenizer, processor, is_train=True):
    """Create a dataset.

    Arguments:
        data_paths: List of paths to data files.
        data_config: The data config.
        tokenizer (Tokenizer): The tokenizer.
        processor (Processor): The processor.
        is_train: Whether this is for training data.

    Returns:
        dataset (Dataset): The dataset.
    """
    from torch.utils.data import Dataset

    from verl.utils.dataset.rl_dataset import RLHFDataset

    # Check if a custom dataset class is specified in the data configuration
    # and if the path to the custom class is provided
    if "custom_cls" in data_config and data_config.custom_cls.get("path", None) is not None:
        # Dynamically load the custom dataset class
        dataset_cls = load_extern_type(data_config.custom_cls.path, data_config.custom_cls.name)
        # Verify that the custom dataset class inherits from torch.utils.data.Dataset
        if not issubclass(dataset_cls, Dataset):
            raise TypeError(
                f"The custom dataset class '{data_config.custom_cls.name}' from "
                f"'{data_config.custom_cls.path}' must inherit from torch.utils.data.Dataset"
            )
    elif "datagen" in data_config and data_config.datagen.get("path", None) is not None and is_train:
        # If a data generation strategy is specified, use the DynamicGenDataset class
        from verl.utils.dataset.dynamicgen_dataset import DynamicGenDataset

        dataset_cls = DynamicGenDataset
        print("Using DynamicGenDataset for data generation.")

    else:
        # Use the default RLHFDataset class if no custom class is specified
        dataset_cls = RLHFDataset
    print(f"Using dataset class: {dataset_cls.__name__}")

    # Instantiate the dataset using the determined dataset class
    dataset = dataset_cls(
        data_files=data_paths,
        tokenizer=tokenizer,
        processor=processor,
        config=data_config,
    )

    return dataset


def create_rl_sampler(data_config, dataset):
    """Create a sampler for the dataset.

    Arguments:
        data_config: The data config.
        dataset (Dataset): The dataset.

    Returns:
        sampler (Sampler): The sampler.
    """
    import torch
    from torch.utils.data import RandomSampler, SequentialSampler
    from omegaconf import OmegaConf

    # Check if custom sampler is configured using safe access
    sampler_config = OmegaConf.select(data_config, "sampler")
    if sampler_config is not None and sampler_config.get("class_path", None) is not None:
        curriculum_class = load_extern_type(
            sampler_config.class_path,
            sampler_config.class_name,
        )
        sampler = curriculum_class(
            data_source=dataset,
            data_config=data_config,
        )
        assert isinstance(sampler, AbstractSampler)
        assert data_config.get("dataloader_num_workers", 8) == 0, (
            "If using curriculum, num_workers must be 0 to prevent data caching. "
            "If the dataloader caches data before the batch is done the "
            "curriculum sampler won't have the opportunity to reorder it. "
        )

    # Use a sampler to facilitate checkpoint resumption.
    # If shuffling is enabled in the data configuration, create a random sampler.
    elif data_config.shuffle:
        train_dataloader_generator = torch.Generator()
        train_dataloader_generator.manual_seed(data_config.get("seed", 1))
        sampler = RandomSampler(data_source=dataset, generator=train_dataloader_generator)
    else:
        # If shuffling is disabled, use a sequential sampler to iterate through the dataset in order.
        sampler = SequentialSampler(data_source=dataset)

    return sampler


if __name__ == "__main__":
    main()
