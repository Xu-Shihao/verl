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
import re
import pandas as pd

# 添加路径以便导入
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# 导入GRPO中的奖励函数
from psy_rewards import psy_reward_function


class ICD10Utils:
    """ICD-10代码处理工具类"""
    
    # 定义11种允许的疾病大类
    ALLOWED_DISEASES = {'F32', 'F41', 'F39', 'F51', 'F98', 'F42', 'F31', 'F43', 'F45', 'F20', 'Z71'}
    
    @staticmethod
    def extract_major_class(code):
        """
        从ICD-10代码中提取大类
        
        Args:
            code: ICD-10代码 (如 'F32.900', 'F41.100')
            
        Returns:
            str: 大类代码 (如 'F32', 'F41') 或 None
        """
        if pd.isna(code) or code is None:
            return None
        
        code_str = str(code).strip()
        # 匹配F开头的代码或Z71
        major_match = re.match(r'(F\d+|Z71)', code_str)
        if major_match:
            major_code = major_match.group(1)
            # 只返回允许的疾病大类
            return major_code if major_code in ICD10Utils.ALLOWED_DISEASES else None
        return None
    
    @staticmethod
    def extract_major_classes_from_list(diagnosis_codes):
        """
        从诊断代码列表中提取所有大类
        
        Args:
            diagnosis_codes: 诊断代码列表或单个代码
            
        Returns:
            list: 大类代码列表
        """
        # 首先检查是否为None或空值
        if diagnosis_codes is None:
            return []
        
        # 如果是列表，直接处理
        if isinstance(diagnosis_codes, list):
            if len(diagnosis_codes) == 0:
                return []
            diagnosis_list = diagnosis_codes
        else:
            # 对于非列表类型，检查是否为NaN
            try:
                if pd.isna(diagnosis_codes):
                    return []
            except (TypeError, ValueError):
                # 如果pd.isna()失败，说明不是标准的NaN值
                pass
            
            # 处理字符串格式的诊断代码
            if isinstance(diagnosis_codes, str):
                try:
                    # 尝试解析为列表
                    import ast
                    diagnosis_list = ast.literal_eval(diagnosis_codes)
                    if isinstance(diagnosis_list, list):
                        diagnosis_list = diagnosis_list
                    else:
                        diagnosis_list = [diagnosis_codes]
                except:
                    diagnosis_list = [diagnosis_codes]
            else:
                diagnosis_list = [diagnosis_codes]
        
        # 提取所有大类
        major_classes = []
        for code in diagnosis_list:
            major_class = ICD10Utils.extract_major_class(code)
            if major_class and major_class not in major_classes:
                major_classes.append(major_class)
        
        return major_classes


def extract_recommendation_codes(response_text, debug=False):
    """从响应文本中提取推荐的多个ICD-10大类代码（支持分号分隔格式）"""
    if not response_text:
        return []
    
    if debug:
        print(f"ICD Recommendation响应文本: {response_text}")
    
    # 提取<box>标签内的内容
    box_pattern = r'<box>(.*?)</box>'
    box_matches = list(re.finditer(box_pattern, response_text, re.DOTALL))
    
    if not box_matches:
        if debug:
            print("未找到<box>标签")
        return []
    
    # 使用最后一个匹配的<box>标签
    predicted_text = box_matches[-1].group(1).strip()
    
    if debug:
        print(f"提取的推荐代码文本: '{predicted_text}'")
    
    # 处理分号分隔的格式
    recommended_codes = []
    
    # 首先尝试分号分隔的格式
    if ';' in predicted_text:
        # 按分号分割
        parts = predicted_text.split(';')
        for part in parts:
            part = part.strip()
            # 从每个部分提取ICD-10代码
            icd10_matches = re.findall(r'F\d+|Z71', part)
            recommended_codes.extend(icd10_matches)
    else:
        # 如果没有分号，尝试其他分隔符或按行分割
        # 尝试逗号分隔
        if ',' in predicted_text:
            parts = predicted_text.split(',')
            for part in parts:
                part = part.strip()
                icd10_matches = re.findall(r'F\d+|Z71', part)
                recommended_codes.extend(icd10_matches)
        else:
            # 按行分割（兼容旧格式）
            lines = predicted_text.split('\n')
            for line in lines:
                line = line.strip()
                if line:
                    icd10_matches = re.findall(r'F\d+|Z71', line)
                    recommended_codes.extend(icd10_matches)
    
    # 如果上述方法都没有找到代码，直接在整个文本中搜索
    if not recommended_codes:
        icd10_matches = re.findall(r'F\d+|Z71', predicted_text)
        recommended_codes.extend(icd10_matches)
    
    # 去重并保持顺序
    final_codes = []
    seen = set()
    for code in recommended_codes:
        if code not in seen:
            final_codes.append(code)
            seen.add(code)
    
    if debug:
        print(f"提取的推荐代码列表: {final_codes}")
        if ';' in predicted_text:
            print("使用分号分隔格式解析")
        elif ',' in predicted_text:
            print("使用逗号分隔格式解析")
        else:
            print("使用行分隔格式解析")
    
    return final_codes


def _map_to_12class(major_codes):
    """
    将ICD大类代码列表映射到12分类体系（11类 + Others）
    
    不在ALLOWED_DISEASES中的代码统一映射为"Others"
    
    例如:
      ['F32', 'F41'] → ['F32', 'F41']   (均在11类中)
      ['F33']        → ['Others']        (F33不在11类中)
      ['F33', 'F41'] → ['F41', 'Others'] (F33→Others, F41保留)
      ['F28', 'F33'] → ['Others']        (两个都不在11类中，去重后只有Others)
    """
    mapped = []
    seen = set()
    for code in major_codes:
        if code in ICD10Utils.ALLOWED_DISEASES:
            if code not in seen:
                seen.add(code)
                mapped.append(code)
        else:
            if "Others" not in seen:
                seen.add("Others")
                mapped.append("Others")
    return mapped


def calculate_icd_reward(solution_str, ground_truth, extra_info=None, debug=False):
    """
    计算ICD代码推荐任务的奖励分数
    
    12分类体系：11个已知ICD大类 + Others（不在11类中的统一归为Others）
    - ground truth中不在11类的代码 → 映射为Others
    - 模型预测中不在11类的代码 → 映射为Others
    - 两边都按12分类来比较exact match
    
    Args:
        solution_str: 模型生成的响应文本
        ground_truth: 真实的ICD代码（字符串或列表）
        extra_info: 额外信息（暂未使用）
        debug: 是否输出调试信息
        
    Returns:
        dict: 包含各种指标的字典
    """
    try:
        # 提取预测的ICD代码
        predicted_codes = extract_recommendation_codes(solution_str, debug=debug)
        
        # 处理ground truth
        if isinstance(ground_truth, str):
            # 尝试解析字符串格式的ground truth
            try:
                import ast
                gt_codes = ast.literal_eval(ground_truth)
                if not isinstance(gt_codes, list):
                    gt_codes = [ground_truth]
            except:
                gt_codes = [ground_truth]
        elif isinstance(ground_truth, list):
            gt_codes = ground_truth
        else:
            gt_codes = [str(ground_truth)] if ground_truth else []
        
        # 提取大类代码（不过滤，保留所有有效的Fxx/Z71代码）
        predicted_major_codes_raw = ICD10Utils.extract_major_classes_from_list(predicted_codes)
        gt_major_codes_raw = _extract_all_major_codes(gt_codes)
        
        # 映射到12分类体系（不在11类中的 → Others）
        predicted_major_codes = _map_to_12class(predicted_major_codes_raw)
        gt_major_codes = _map_to_12class(gt_major_codes_raw)
        
        if debug:
            print(f"预测的大类代码(原始): {predicted_major_codes_raw}")
            print(f"真实的大类代码(原始): {gt_major_codes_raw}")
            print(f"预测的大类代码(12分类): {predicted_major_codes}")
            print(f"真实的大类代码(12分类): {gt_major_codes}")
        
        # 计算格式正确性（是否有<box>标签且提取到代码）
        has_box_format = '<box>' in solution_str and '</box>' in solution_str
        format_score = 1.0 if (has_box_format and len(predicted_codes) > 0) else 0.0
        
        # 计算exact match（集合完全匹配，基于12分类）
        predicted_set = set(predicted_major_codes)
        gt_set = set(gt_major_codes)
        exact_match = 1.0 if predicted_set == gt_set else 0.0
        
        # 计算F1分数（用于更细粒度的评估）
        if len(gt_set) == 0 and len(predicted_set) == 0:
            # 都为空，完全正确
            precision = 1.0
            recall = 1.0
            f1_score = 1.0
        elif len(gt_set) == 0:
            # 真实为空，但预测不为空
            precision = 0.0
            recall = 1.0  # 召回率定义为1（没有遗漏）
            f1_score = 0.0
        elif len(predicted_set) == 0:
            # 预测为空，但真实不为空
            precision = 1.0  # 精确率定义为1（没有错误预测）
            recall = 0.0
            f1_score = 0.0
        else:
            # 都不为空，正常计算
            intersection = predicted_set.intersection(gt_set)
            precision = len(intersection) / len(predicted_set)
            recall = len(intersection) / len(gt_set)
            f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # 计算总分（新的计算方式：format_acc * 0.2 + format_acc * exact_match_acc * 0.8）
        total_score = format_score * 0.2 + format_score * exact_match * 0.8
        
        result = {
            "score": total_score,
            "exact_match": exact_match,
            "format_score": format_score,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "predicted_codes": predicted_codes,
            "predicted_major_codes": predicted_major_codes,
            "gt_codes": gt_codes,
            "gt_major_codes": gt_major_codes,
            "acc": exact_match > 0,  # 兼容性字段
            "data_source": "icd_recommendation"
        }
        
        if debug:
            print(f"ICD奖励计算结果: {result}")
        
        return result
        
    except Exception as e:
        if debug:
            print(f"ICD奖励计算错误: {e}")
        return {
            "score": 0.0,
            "exact_match": 0.0,
            "format_score": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
            "predicted_codes": [],
            "predicted_major_codes": [],
            "gt_codes": [],
            "gt_major_codes": [],
            "acc": False,
            "error": str(e),
            "data_source": "icd_recommendation"
        }


def _extract_all_major_codes(codes):
    """
    从代码列表中提取所有ICD大类代码（不过滤ALLOWED_DISEASES）
    
    与ICD10Utils.extract_major_classes_from_list不同，此函数保留所有有效的Fxx/Z71代码，
    不通过ALLOWED_DISEASES过滤。
    """
    if not codes:
        return []
    
    if not isinstance(codes, list):
        codes = [codes]
    
    major_codes = []
    seen = set()
    for code in codes:
        if code is None:
            continue
        code_str = str(code).strip()
        match = re.match(r'([FZ]\d{2})', code_str.upper())
        if match:
            major = match.group(1)
            if major not in seen:
                seen.add(major)
                major_codes.append(major)
    
    return major_codes


def create_dapo_psy_reward_fn(is_validation=None, tokenizer=None, use_symptom_reward=False, symptom_alpha=0.1, use_icd_reward=False):
    """
    创建DAPO专用的心理诊断奖励函数，包含详细的rollout输出功能
    
    Args:
        is_validation: 模式控制 - "val" 为验证模式, "train" 为训练模式, None 为无日志模式
        tokenizer: 用于解码token的tokenizer
        use_symptom_reward: 是否启用症状识别奖励
        symptom_alpha: 症状F1分数在奖励函数中的权重
        use_icd_reward: 是否启用ICD代码推荐奖励
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
        'total_icd_exact_match': 0.0,  # 累计ICD exact match分数
        'total_icd_format_score': 0.0,  # 累计ICD格式分数
        'icd_samples': 0,  # 有ICD数据的样本数
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
            "total_score": [],
            "icd_exact_match": [],
            "icd_format_score": [],
            "icd_f1_score": []
        }
        
        # 批次统计信息
        batch_correct = 0
        batch_format_correct = 0
        batch_total_score = 0.0
        batch_symptom_coverage = 0.0  # 当前批次症状F1分数累计
        batch_symptom_samples = 0     # 当前批次有症状数据的样本数
        batch_icd_exact_match = 0.0   # 当前批次ICD exact match累计
        batch_icd_format_score = 0.0  # 当前批次ICD格式分数累计
        batch_icd_samples = 0         # 当前批次有ICD数据的样本数
        
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
                
                # 根据开关组合决定奖励计算方式
                if use_icd_reward:
                    # 计算ICD代码推荐奖励
                    icd_result = calculate_icd_reward(
                        solution_str=response_text,
                        ground_truth=ground_truth,
                        extra_info={"patient_id": patient_id} if patient_id else None,
                        debug=(is_validation is not None and i == 0)
                    )
                    
                    if use_symptom_reward:
                        # 同时启用症状奖励，需要获取症状F1分数
                        if psy_reward_function is not None:
                            # 为psy_reward_function准备extra_info
                            extra_info_for_reward = {"patient_id": patient_id} if patient_id else None
                            psy_result = psy_reward_function(
                                data_source="psy_diagnosis",
                                solution_str=response_text,
                                ground_truth=ground_truth,
                                extra_info=extra_info_for_reward,
                                use_symptom_reward=True,
                                symptom_alpha=symptom_alpha,
                                tracker=None  # 不传递tracker，避免个别样本记录，只在批次级别记录
                            )
                        else:
                            # 兜底到原始函数
                            psy_result = register_psy_reward_function(
                                data_source="psy_diagnosis",
                                solution_str=response_text,
                                ground_truth=ground_truth,
                                extra_info={"patient_id": patient_id} if patient_id else None,
                                use_symptom_reward=True,
                                symptom_alpha=symptom_alpha
                            )
                        
                        # 组合ICD奖励和症状奖励
                        if isinstance(psy_result, dict):
                            symptom_f1 = psy_result.get('symptom_f1', 0.0)
                            # 将症状F1分数以symptom_alpha为系数加入ICD奖励中
                            combined_score = icd_result.get("score", 0.0) + symptom_alpha * symptom_f1
                            
                            # 创建组合结果
                            result = icd_result.copy()
                            result["score"] = combined_score
                            result["symptom_f1"] = symptom_f1
                            result["icd_base_score"] = icd_result.get("score", 0.0)
                            result["symptom_contribution"] = symptom_alpha * symptom_f1
                        else:
                            result = icd_result
                    else:
                        # 只使用ICD奖励
                        result = icd_result
                else:
                    # 使用原有的心理诊断奖励
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
                if use_icd_reward:
                    # ICD奖励模式的指标
                    psy_metrics["icd_exact_match"].append(result.get("exact_match", 0.0))
                    psy_metrics["icd_format_score"].append(result.get("format_score", 0.0))
                    psy_metrics["icd_f1_score"].append(result.get("f1_score", 0.0))
                    psy_metrics["diagnosis_accuracy"].append(result.get("exact_match", 0.0))  # 兼容性
                    psy_metrics["format_accuracy"].append(result.get("format_score", 0.0))
                else:
                    # 原有心理诊断奖励模式的指标
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
                if use_icd_reward:
                    # 更新ICD统计
                    if 'error' not in result:
                        batch_icd_exact_match += result.get('exact_match', 0.0)
                        batch_icd_format_score += result.get('format_score', 0.0)
                        batch_icd_samples += 1
                else:
                    # 更新症状统计
                    symptom_f1 = result.get('symptom_f1', 0.0)
                    # 只有当有有效的症状数据时才统计（避免None或错误情况）
                    if 'error' not in result and patient_id is not None:
                        batch_symptom_coverage += symptom_f1  # 使用F1分数累计
                        batch_symptom_samples += 1
            
            # 打印验证示例（第一个样本）
            if is_validation == "val" and not printed_validation_example[0] and i == 0:
                print("\n" + "="*80)
                if use_icd_reward and use_symptom_reward:
                    print("【DAPO验证阶段ICD+症状组合奖励计算示例】")
                elif use_icd_reward:
                    print("【DAPO验证阶段ICD奖励计算示例】")
                else:
                    print("【DAPO验证阶段奖励计算示例】")
                print("="*80)
                
                if isinstance(result, dict):
                    if use_icd_reward:
                        # ICD奖励模式的显示
                        predicted_codes = result.get('predicted_codes', [])
                        predicted_major_codes = result.get('predicted_major_codes', [])
                        gt_codes = result.get('gt_codes', [])
                        gt_major_codes = result.get('gt_major_codes', [])
                        exact_match = result.get('exact_match', 0.0)
                        format_score = result.get('format_score', 0.0)
                        f1_score = result.get('f1_score', 0.0)
                        
                        print(f"预测的ICD代码: {predicted_codes}")
                        print(f"预测的大类代码: {predicted_major_codes}")
                        print(f"Ground Truth代码: {gt_codes}")
                        print(f"Ground Truth大类: {gt_major_codes}")
                        print(f"Exact Match: {exact_match:.3f}")
                        print(f"Format Score: {format_score:.3f}")
                        print(f"F1 Score: {f1_score:.3f}")
                        print(f"Total Score: {score:.3f}")
                    else:
                        # 原有心理诊断奖励模式的显示
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
                if use_icd_reward:
                    print("【DAPO训练阶段ICD奖励计算示例】")
                else:
                    print("【DAPO训练阶段奖励计算示例】")
                print("="*80)
                
                if isinstance(result, dict):
                    if use_icd_reward:
                        # ICD奖励模式的显示
                        predicted_codes = result.get('predicted_codes', [])
                        predicted_major_codes = result.get('predicted_major_codes', [])
                        gt_codes = result.get('gt_codes', [])
                        gt_major_codes = result.get('gt_major_codes', [])
                        exact_match = result.get('exact_match', 0.0)
                        format_score = result.get('format_score', 0.0)
                        f1_score = result.get('f1_score', 0.0)
                        
                        print(f"预测的ICD代码: {predicted_codes}")
                        print(f"预测的大类代码: {predicted_major_codes}")
                        print(f"Ground Truth代码: {gt_codes}")
                        print(f"Ground Truth大类: {gt_major_codes}")
                        print(f"Exact Match: {exact_match:.3f}")
                        print(f"Format Score: {format_score:.3f}")
                        print(f"F1 Score: {f1_score:.3f}")
                        print(f"Total Score: {score:.3f}")
                    else:
                        # 原有心理诊断奖励模式的显示
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
            rollout_stats['total_icd_exact_match'] += batch_icd_exact_match
            rollout_stats['total_icd_format_score'] += batch_icd_format_score
            rollout_stats['icd_samples'] += batch_icd_samples
            rollout_stats['batch_count'] += 1
            rollout_stats['rollout_count'] += 1
            
            # 每隔几个批次打印rollout统计
            if rollout_stats['batch_count'] % 10 == 0 or rollout_stats['batch_count'] <= 5:
                diagnosis_accuracy = rollout_stats['correct_samples'] / rollout_stats['total_samples'] * 100
                format_accuracy = rollout_stats['format_correct_samples'] / rollout_stats['total_samples'] * 100
                avg_total_score = rollout_stats['total_score'] / rollout_stats['total_samples']
                
                print("\n" + "-"*60)
                if use_icd_reward:
                    print(f"【DAPO ICD Rollout 统计 - Batch {rollout_stats['batch_count']}】")
                    print(f"样本数: {rollout_stats['total_samples']}")
                    
                    # 计算ICD统计
                    icd_exact_match_accuracy = 0.0
                    icd_format_accuracy = 0.0
                    if rollout_stats['icd_samples'] > 0:
                        icd_exact_match_accuracy = rollout_stats['total_icd_exact_match'] / rollout_stats['icd_samples'] * 100
                        icd_format_accuracy = rollout_stats['total_icd_format_score'] / rollout_stats['icd_samples'] * 100
                    
                    print(f"ICD Exact Match: {icd_exact_match_accuracy:.2f}%")
                    print(f"ICD Format Accuracy: {icd_format_accuracy:.2f}%")
                    print(f"Total Score: {avg_total_score:.4f}")
                else:
                    print(f"【DAPO Rollout 统计 - Batch {rollout_stats['batch_count']}】")
                    print(f"样本数: {rollout_stats['total_samples']}")
                    print(f"Diagnosis Accuracy (F1): {diagnosis_accuracy:.2f}%")
                    
                    # 计算症状F1分数
                    symptom_accuracy = 0.0
                    if rollout_stats['symptom_samples'] > 0:
                        symptom_accuracy = rollout_stats['total_symptom_coverage'] / rollout_stats['symptom_samples'] * 100
                    
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
                if use_icd_reward:
                    # ICD奖励模式的详细显示
                    predicted_codes = result.get('predicted_codes', [])
                    predicted_major_codes = result.get('predicted_major_codes', [])
                    gt_codes = result.get('gt_codes', [])
                    gt_major_codes = result.get('gt_major_codes', [])
                    exact_match = result.get('exact_match', 0.0)
                    format_score = result.get('format_score', 0.0)
                    f1_score = result.get('f1_score', 0.0)
                    precision = result.get('precision', 0.0)
                    recall = result.get('recall', 0.0)
                    
                    print(f"Predicted ICD Codes: {predicted_codes}")
                    print(f"Predicted Major Codes: {predicted_major_codes}")
                    print(f"Ground Truth Codes: {gt_codes}")
                    print(f"Ground Truth Major: {gt_major_codes}")
                    print(f"Exact Match: {exact_match:.3f}")
                    print(f"Format Score: {format_score:.3f}")
                    print(f"F1 Score: {f1_score:.3f}")
                    print(f"Precision: {precision:.3f}")
                    print(f"Recall: {recall:.3f}")
                    print(f"Total Score: {random_sample['score']:.3f}")
                    print(f"Is Correct: {random_sample['is_correct']}")
                    print(f"Format OK: {random_sample['format_ok']}")
                else:
                    # 原有心理诊断奖励模式的详细显示
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
                    
                    # 显示症状confusion metrics（如果有）
                    if 'true_positive' in result:
                        tp = result.get('true_positive', 0)
                        fp = result.get('false_positive', 0)
                        tn = result.get('true_negative', 0)
                        fn = result.get('false_negative', 0)
                        print(f"Symptom Confusion: TP={tp}, FP={fp}, TN={tn}, FN={fn}")
                
                if random_sample['patient_id']:
                    print(f"Patient ID: {random_sample['patient_id']}")
                
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
    
    print("\n" + "="*60)
    print("Testing ICD reward function...")
    
    # ICD奖励测试用例
    icd_test_cases = [
        {
            "solution_str": "<think>患者表现出持续的情绪低落、兴趣减退、精力缺乏等症状，符合抑郁发作的诊断标准。</think><box>F32.1</box>",
            "ground_truth": ["F32.1"],
            "description": "完全匹配的ICD代码"
        },
        {
            "solution_str": "<think>患者同时表现出抑郁和焦虑症状。</think><box>F32.1;F41.1</box>",
            "ground_truth": ["F32.1", "F41.1"],
            "description": "多个ICD代码完全匹配"
        },
        {
            "solution_str": "<think>患者表现出抑郁症状。</think><box>F32.2</box>",
            "ground_truth": ["F32.1"],
            "description": "同大类但不同小类的ICD代码"
        },
        {
            "solution_str": "<think>患者表现出焦虑症状。</think><box>F41.0</box>",
            "ground_truth": ["F32.1"],
            "description": "不同大类的ICD代码"
        },
        {
            "solution_str": "患者表现出抑郁症状，诊断为F32.1",
            "ground_truth": ["F32.1"],
            "description": "缺少<box>标签的格式错误"
        },
        {
            "solution_str": "<think>患者表现出抑郁和焦虑症状。</think><box>F32.1</box>",
            "ground_truth": ["F32.1", "F41.1"],
            "description": "预测不完整（遗漏部分诊断）"
        }
    ]
    
    for i, test_case in enumerate(icd_test_cases):
        print(f"\nICD Test Case {i+1}: {test_case['description']}")
        print(f"  Solution: {test_case['solution_str']}")
        print(f"  Ground Truth: {test_case['ground_truth']}")
        
        result = calculate_icd_reward(
            solution_str=test_case["solution_str"],
            ground_truth=test_case["ground_truth"],
            debug=True
        )
        
        print(f"  Total Score: {result.get('score', 0.0):.3f}")
        print(f"  Exact Match: {result.get('exact_match', 0.0):.3f}")
        print(f"  Format Score: {result.get('format_score', 0.0):.3f}")
        print(f"  F1 Score: {result.get('f1_score', 0.0):.3f}")
        print(f"  Precision: {result.get('precision', 0.0):.3f}")
        print(f"  Recall: {result.get('recall', 0.0):.3f}")
        print(f"  Predicted Major Codes: {result.get('predicted_major_codes', [])}")
        print(f"  GT Major Codes: {result.get('gt_major_codes', [])}")
    
    print("\n✓ DAPO reward function testing completed!")
    print("✓ ICD reward function testing completed!")
