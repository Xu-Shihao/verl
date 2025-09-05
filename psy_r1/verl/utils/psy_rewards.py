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

import re
import traceback
import json
import pandas as pd
import os
from typing import Union, Dict, Any, List

# 尝试导入wandb，如果不可用则使用空实现
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None

# 导入verl官方tracking机制
try:
    from verl.utils.tracking import Tracking
    VERL_TRACKING_AVAILABLE = True
except ImportError:
    VERL_TRACKING_AVAILABLE = False
    Tracking = None

# 全局变量，用于缓存症状数据
_symptom_data_cache = None
_mapping_cache = None  # 保留以保持兼容性，但不再使用
_symptom_columns_cache = None


def load_symptom_data():
    """
    加载症状数据，直接从新的症状识别结果文件中读取
    
    注意：已更新为使用新的数据文件 symptom_identification_from_pkl_results.xlsx
    这个文件直接包含 patient_id 列，无需再通过映射文件进行转换
    
    文件格式预期：
    - 第一列或名为 'patient_id' 的列：患者ID
    - 其余列：各种症状，值为0-1之间的概率
    
    Returns:
        tuple: (症状数据DataFrame, None (不再需要映射), 症状列名列表)
    """
    global _symptom_data_cache, _mapping_cache, _symptom_columns_cache
    
    if _symptom_data_cache is not None:
        return _symptom_data_cache, _mapping_cache, _symptom_columns_cache
    
    try:
        # 获取当前文件的目录
        current_dir = os.path.dirname(os.path.abspath(__file__))
        base_dir = os.path.dirname(os.path.dirname(current_dir))  # 向上两级到psy_r1目录
        
        # 读取新的症状数据文件
        symptom_file = os.path.join(base_dir, 'metadata', 'symptom_identification_from_pkl_results.xlsx')
        symptom_df = pd.read_excel(symptom_file)
        
        # 获取症状列名（排除patient_id列和其他非症状列）
        # 假设第一列是patient_id，其余列都是症状
        non_symptom_columns = ['patient_id', 'Patient_ID', 'PatientID', 'ID', 'id', "VisitNumber"]
        symptom_columns = [col for col in symptom_df.columns if col not in non_symptom_columns]
        
        # 确保patient_id列存在
        patient_id_col = None
        for col in ['patient_id', 'Patient_ID', 'PatientID', 'ID', 'id', "VisitNumber"]:
            if col in symptom_df.columns:
                patient_id_col = col
                break
        
        if patient_id_col is None:
            # 如果没有找到明确的patient_id列，假设第一列是patient_id
            patient_id_col = symptom_df.columns[0]
            print(f"[WARNING] 未找到明确的patient_id列，使用第一列: {patient_id_col}")
        
        # 如果列名不是'patient_id'，重命名为'patient_id'以保持一致性
        if patient_id_col != 'patient_id':
            symptom_df = symptom_df.rename(columns={patient_id_col: 'patient_id'})
            
        # 更新症状列名（移除patient_id）
        symptom_columns = [col for col in symptom_df.columns if col != 'patient_id']
        
        # 缓存数据（不再需要映射关系）
        _symptom_data_cache = symptom_df
        _mapping_cache = None  # 不再需要映射关系
        _symptom_columns_cache = symptom_columns
        
        print(f"[INFO] 成功加载症状数据: {symptom_df.shape[0]} 条记录, {len(symptom_columns)} 个症状")
        print(f"[INFO] Patient ID列: patient_id, 症状列数: {len(symptom_columns)}")
        
        return symptom_df, _mapping_cache, symptom_columns
        
    except Exception as e:
        print(f"[ERROR] 加载症状数据失败: {str(e)}")
        traceback.print_exc()
        return None, None, None


def extract_symptoms_from_text(text: str, symptom_list: List[str]) -> List[str]:
    """
    从文本中提取提到的症状
    
    Args:
        text: 要分析的文本
        symptom_list: 症状列表
        
    Returns:
        提取到的症状列表
    """
    if not text or not symptom_list:
        return []
    
    # 转换为小写进行匹配
    text_lower = text.lower()
    extracted_symptoms = []
    
    for symptom in symptom_list:
        if not symptom:
            continue
            
        symptom_lower = symptom.lower()
        
        # 检查症状是否在文本中
        if symptom_lower in text_lower:
            # 查找症状在文本中的位置
            positions = []
            start = 0
            while True:
                pos = text_lower.find(symptom_lower, start)
                if pos == -1:
                    break
                positions.append(pos)
                start = pos + 1
            
            # 对每个位置检查是否有否定词
            symptom_found = False
            for pos in positions:
                # 检查症状前后30个字符的上下文
                context_start = max(0, pos-30)
                context_end = min(len(text_lower), pos+len(symptom_lower)+30)
                context = text_lower[context_start:context_end]
                
                # 找到症状在context中的相对位置
                symptom_pos_in_context = pos - context_start
                
                # 检查否定模式
                is_negative = False
                
                # 1. 检查明确的否定模式
                before_symptom = context[:symptom_pos_in_context]
                after_symptom = context[symptom_pos_in_context + len(symptom_lower):]
                
                # 检查明确的否定关键词和模式
                full_context_around = text_lower[max(0, pos-15):min(len(text_lower), pos+len(symptom_lower)+15)]
                
                # 明确的否定模式
                negative_patterns = [
                    r'但没有' + re.escape(symptom_lower),
                    r'无' + re.escape(symptom_lower) + r'(表现|症状|倾向)',
                    r'无' + re.escape(symptom_lower) + r'$',  # "无xxx"在句尾
                    r'没有明显(的)?' + re.escape(symptom_lower),
                    r'未见' + re.escape(symptom_lower),
                    r'无明显' + re.escape(symptom_lower),
                    r'排除' + re.escape(symptom_lower),
                    r'否认' + re.escape(symptom_lower),
                ]
                
                for pattern in negative_patterns:
                    if re.search(pattern, full_context_around):
                        is_negative = True
                        break
                
                # 特殊情况：检查"但没有xxx表现"模式
                if '但没有' in full_context_around and '表现' in full_context_around:
                    # 找到"但没有"的位置
                    neg_start = full_context_around.find('但没有')
                    expr_start = full_context_around.find('表现')
                    symptom_start = full_context_around.find(symptom_lower)
                    
                    # 如果症状在"但没有"和"表现"之间，则为否定
                    if neg_start < symptom_start < expr_start:
                        is_negative = True
                
                if not is_negative:
                    symptom_found = True
                    break
            
            if symptom_found and symptom not in extracted_symptoms:
                extracted_symptoms.append(symptom)
    
    return extracted_symptoms


def calculate_symptom_coverage(patient_id_raw: Union[str, int], model_response: str) -> Dict[str, Any]:
    """
    计算症状覆盖率，直接从新的症状数据文件中获取数据
    
    注意：已更新为直接使用 patient_id 查询，无需映射转换
    
    Args:
        patient_id_raw: 患者ID（可能是格式为 "{id}_conv{i}" 的字符串或直接的数字ID）
        model_response: 模型回答
        
    Returns:
        包含症状覆盖率信息的字典
    """
    try:
        # 加载症状数据
        symptom_df, _, symptom_columns = load_symptom_data()
        
        if symptom_df is None or symptom_columns is None:
            return {
                "symptom_coverage": 0.0,
                "extracted_symptoms": [],
                "ground_truth_symptoms": [],
                "total_symptoms": 0,
                "error": "Failed to load symptom data"
            }
        
        # 解析patient_id，处理 "{id}_conv{i}" 格式
        try:
            if patient_id_raw is None:
                raise ValueError("patient_id cannot be None")
                
            if isinstance(patient_id_raw, str) and "_conv" in patient_id_raw:
                # 提取 {id}_conv{i} 中的 id 部分
                id_part = patient_id_raw.split("_conv")[0]
                if not id_part:  # 空ID部分
                    raise ValueError("Empty ID part in patient_id")
                patient_id = int(id_part)
                # print(f"[DEBUG] 解析患者ID: {patient_id_raw} -> {patient_id}")
            else:
                # 直接转换为整数
                patient_id = int(patient_id_raw)
        except (ValueError, AttributeError, TypeError) as e:
            return {
                "symptom_coverage": 0.0,
                "extracted_symptoms": [],
                "ground_truth_symptoms": [],
                "total_symptoms": 0,
                "error": f"Invalid patient ID format: {patient_id_raw}, error: {str(e)}"
            }
        
        # 直接从症状数据中查找对应的记录，不再需要映射
        patient_row = symptom_df[symptom_df['patient_id'] == patient_id]
        if patient_row.empty:
            return {
                "symptom_coverage": 0.0,
                "extracted_symptoms": [],
                "ground_truth_symptoms": [],
                "total_symptoms": 0,
                "error": f"Patient ID {patient_id} (from {patient_id_raw}) not found in symptom data"
            }
        
        # 获取患者的症状真值（概率大于0.5的症状）
        patient_symptoms = patient_row.iloc[0]
        ground_truth_symptoms = []
        
        for symptom in symptom_columns:
            if patient_symptoms[symptom] > 0.5:
                ground_truth_symptoms.append(symptom)
        
        if len(ground_truth_symptoms) == 0:
            return {
                "symptom_coverage": 1.0,  # 如果没有真实症状，覆盖率为100%
                "extracted_symptoms": [],
                "ground_truth_symptoms": [],
                "total_symptoms": 0,
            }
        
        # 从模型回答中提取症状
        extracted_symptoms = extract_symptoms_from_text(model_response, ground_truth_symptoms)
        
        # 计算覆盖率
        coverage = len(extracted_symptoms) / len(ground_truth_symptoms) if ground_truth_symptoms else 0.0
        
        return {
            "symptom_coverage": coverage,
            "extracted_symptoms": extracted_symptoms,
            "ground_truth_symptoms": ground_truth_symptoms,
            "total_symptoms": len(ground_truth_symptoms),
        }
        
    except Exception as e:
        print(f"[ERROR] 计算症状覆盖率失败: {str(e)}")
        traceback.print_exc()
        return {
            "symptom_coverage": 0.0,
            "extracted_symptoms": [],
            "ground_truth_symptoms": [],
            "total_symptoms": 0,
            "error": str(e)
        }


def log_metrics_to_tracking(result_dict: Dict[str, Any], step: int = None, tracker: 'Tracking' = None):
    """
    将评估指标记录到verl官方tracking系统
    
    Args:
        result_dict: 包含评估指标的字典
        step: 步骤数（可选）
        tracker: Tracking实例，如果为None则尝试使用wandb直接记录（向后兼容）
    """
    try:
        # 提取需要记录的指标
        metrics = {}
        
        # 总分数
        if "score" in result_dict:
            metrics["rewards/total_score"] = result_dict["score"]
        if "total_score" in result_dict:
            metrics["rewards/total_score"] = result_dict["total_score"]
            
        # 诊断正确率
        if "diagnosis_score" in result_dict:
            metrics["rewards/diagnosis_accuracy"] = result_dict["diagnosis_score"]
        if "diagnosis_accuracy" in result_dict:
            metrics["rewards/diagnosis_accuracy"] = result_dict["diagnosis_accuracy"]
            
        # 症状识别准确率
        if "symptom_coverage" in result_dict:
            metrics["rewards/symptom_accuracy"] = result_dict["symptom_coverage"]
        if "symptom_accuracy" in result_dict:
            metrics["rewards/symptom_accuracy"] = result_dict["symptom_accuracy"]
            
        # 格式正确率
        if "format_score" in result_dict:
            metrics["rewards/format_accuracy"] = result_dict["format_score"]
            
        # 准确率（诊断是否完全正确）
        if "acc" in result_dict:
            metrics["rewards/exact_match"] = float(result_dict["acc"])
            
        # 症状统计
        if "total_symptoms" in result_dict:
            metrics["symptoms/total_count"] = result_dict["total_symptoms"]
        if "extracted_symptoms" in result_dict and isinstance(result_dict["extracted_symptoms"], list):
            metrics["symptoms/extracted_count"] = len(result_dict["extracted_symptoms"])
    
        # 记录到tracking系统
        if metrics:
            if tracker is not None:
                # 使用verl官方tracking机制
                tracker.log(data=metrics, step=step)
            elif WANDB_AVAILABLE and wandb.run is not None:
                # 向后兼容：直接使用wandb记录
                if step is not None:
                    wandb.log(metrics, step=step)
                else:
                    wandb.log(metrics)
            else:
                print(f"[WARNING] No tracking system available. Metrics: {metrics}")
                
    except Exception as e:
        print(f"[WARNING] Failed to log metrics to tracking system: {str(e)}")


def extract_diagnosis_from_box(text: str) -> str:
    """
    从文本中提取 <box></box> 标签内的诊断结果
    
    Args:
        text: 包含诊断结果的文本
        
    Returns:
        提取的诊断结果，如果没有找到则返回空字符串
    """
    # 查找所有 <box> 的起始位置
    box_starts = []
    box_ends = []
    
    # 查找所有 <box> 标签位置（不区分大小写）
    for match in re.finditer(r'<box>', text, re.IGNORECASE):
        box_starts.append(match.end())
    
    # 查找所有 </box> 标签位置（不区分大小写）
    for match in re.finditer(r'</box>', text, re.IGNORECASE):
        box_ends.append(match.start())
    
    # 如果没有配对的标签，返回空字符串
    if not box_starts or not box_ends:
        return ""
    
    # 从后往前匹配，找到最后一个有效的 <box></box> 对
    for end_pos in reversed(box_ends):
        # 找到在这个 </box> 之前的最近的 <box>
        for start_pos in reversed(box_starts):
            if start_pos < end_pos:
                # 提取内容并返回
                content = text[start_pos:end_pos].strip()
                return content
    
    return ""


def normalize_diagnosis(diagnosis: str) -> str:
    """
    标准化诊断结果，去除多余的空白字符和标点符号
    
    Args:
        diagnosis: 原始诊断结果
        
    Returns:
        标准化后的诊断结果
    """
    if not diagnosis:
        return ""
    
    # 转换为小写并去除前后空白
    normalized = diagnosis.lower().strip()
    
    # 去除多余的空白字符
    normalized = re.sub(r'\s+', ' ', normalized)
    
    # 去除常见的标点符号
    normalized = re.sub(r'[.,;:!?，。；：！？]', '', normalized)
    
    return normalized


def check_diagnosis_correctness(predicted_diagnosis: str, ground_truth: Union[str, List[str]]) -> bool:
    """
    检查预测的诊断结果是否正确
    
    Args:
        predicted_diagnosis: 预测的诊断结果
        ground_truth: 正确的诊断结果，可以是字符串或字符串列表
        
    Returns:
        True 如果诊断正确，False 否则
    """
    if not predicted_diagnosis:
        return False
    
    # 标准化预测结果
    pred_norm = normalize_diagnosis(predicted_diagnosis)
    
    # 如果ground_truth是字符串，转换为列表
    if isinstance(ground_truth, str):
        gt_list = [ground_truth]
    else:
        gt_list = ground_truth
    
    # 检查是否与任一正确答案匹配
    for gt in gt_list:
        gt_norm = normalize_diagnosis(str(gt))
        
        # 精确匹配
        if pred_norm == gt_norm:
            return True
        
        # # 包含关系匹配（预测结果包含正确答案的关键词）
        # if gt_norm and gt_norm in pred_norm:
        #     return True
        
        # # 关键词匹配（正确答案的关键词在预测结果中）
        # gt_keywords = gt_norm.split()
        # if len(gt_keywords) > 0:
        #     # 至少匹配一半的关键词
        #     matched_keywords = sum(1 for keyword in gt_keywords if keyword in pred_norm)
        #     if matched_keywords >= len(gt_keywords) * 0.5:
        #         return True
    
    return False


def compute_diagnosis_score(
    model_response: str, 
    ground_truth: Union[str, List[str]], 
    patient_id: Union[str, int] = None,
    return_details: bool = False,
    use_symptom_reward: bool = False,
    symptom_alpha: float = 0.1
) -> Union[float, Dict[str, Any]]:
    """
    计算问诊诊断结果的正确率分数，可选包含症状覆盖率
    
    Args:
        model_response: 模型的完整回答
        ground_truth: 正确的诊断结果
        patient_id: 患者ID，支持格式 "{id}_conv{i}" 或直接数字ID
        return_details: 是否返回详细信息
        use_symptom_reward: 是否启用症状识别奖励功能
        
    Returns:
        如果return_details为False，返回总分数
        如果return_details为True，返回包含详细信息的字典
    """
    try:
        # 从模型回答中提取诊断结果
        extracted_diagnosis = extract_diagnosis_from_box(model_response)
        
        # 检查格式是否正确（是否能提取到诊断结果）
        format_correct = bool(extracted_diagnosis)
        
        # 检查诊断是否正确
        is_correct = False
        if format_correct:
            is_correct = check_diagnosis_correctness(extracted_diagnosis, ground_truth)
        
        # 计算诊断正确率分数
        diagnosis_score = 1.0 if is_correct else 0.0
        
        # 计算症状覆盖率（仅在启用时）
        symptom_info = {"symptom_coverage": 0.0}
        if use_symptom_reward and patient_id is not None:
            symptom_info = calculate_symptom_coverage(patient_id, model_response) # patient_id 为 {id}_conv{i}
        
        symptom_coverage = symptom_info.get("symptom_coverage", 0.0)
        
        # 计算总分数：根据是否启用症状奖励来决定
        if use_symptom_reward:
            # 新的评分公式：诊断正确率 + symptom_alpha * 症状识别准确率
            total_score = diagnosis_score + symptom_alpha * symptom_coverage
        else:
            # 传统评分：仅基于诊断正确率
            total_score = diagnosis_score
        
        if return_details:
            result = {
                "score": total_score,
                "diagnosis_score": diagnosis_score,
                "symptom_coverage": symptom_coverage,
                "format_score": 1.0 if format_correct else 0.0,
                "acc": is_correct,
                "extracted_diagnosis": extracted_diagnosis,
                "ground_truth": ground_truth,
            }
            # 添加症状相关信息
            result.update(symptom_info)
            return result
        else:
            return total_score
            
    except Exception as e:
        print(f"[ERROR] Error in compute_diagnosis_score: {str(e)}")
        traceback.print_exc()
        
        if return_details:
            return {
                "score": 0.0,
                "diagnosis_score": 0.0,
                "symptom_coverage": 0.0,
                "format_score": 0.0,
                "acc": False,
                "extracted_diagnosis": "",
                "ground_truth": ground_truth,
                "error": str(e),
            }
        else:
            return 0.0


def psy_reward_function(data_source: str, solution_str: str, ground_truth: str, extra_info=None, use_symptom_reward: bool = False, symptom_alpha: float = 0.1, tracker: 'Tracking' = None):
    """
    问诊场景的reward函数，兼容现有的reward函数接口
    
    Args:
        data_source: 数据源标识
        solution_str: 模型生成的解答
        ground_truth: 正确答案
        extra_info: 额外信息，应包含patient_id用于症状分析
        use_symptom_reward: 是否启用症状识别奖励功能（默认False保持向后兼容）
        tracker: Tracking实例，用于记录指标到verl官方tracking系统
        
    Returns:
        包含分数和额外信息的字典
    """
    try:
        # 从extra_info中提取patient_id
        patient_id = None
        if extra_info and isinstance(extra_info, dict):
            patient_id = extra_info.get("patient_id")
        
        # 计算诊断正确率和症状覆盖率（根据参数决定是否启用症状奖励）
        result = compute_diagnosis_score(solution_str, ground_truth, patient_id=patient_id, return_details=True, use_symptom_reward=use_symptom_reward, symptom_alpha=symptom_alpha)
        
        # 确保result是字典类型
        if isinstance(result, dict):
            # 如果有额外信息，添加到结果中
            if extra_info:
                result["extra_info"] = extra_info
            
            # 为了兼容wandb日志记录，添加一些额外的字段
            result["total_score"] = result.get("score", 0.0)
            result["diagnosis_accuracy"] = result.get("diagnosis_score", 0.0)
            result["symptom_accuracy"] = result.get("symptom_coverage", 0.0)
            
            # 注意：不再在这里立即记录指标到tracking系统
            # 指标记录现在在批次级别处理，以便记录平均值而不是每个样本的值
            
            return result
        else:
            # 如果不是字典，创建一个标准格式的字典
            return {
                "score": float(result),
                "total_score": float(result),
                "diagnosis_score": float(result),
                "diagnosis_accuracy": float(result),
                "symptom_coverage": 0.0,
                "symptom_accuracy": 0.0,
                "format_score": 1.0 if result > 0 else 0.0,
                "acc": bool(result > 0),
                "extracted_diagnosis": "",
                "ground_truth": ground_truth,
            }
        
    except Exception as e:
        print(f"[ERROR] Error in psy_reward_function: {str(e)}")
        traceback.print_exc()
        return {
            "score": 0.0,
            "total_score": 0.0,
            "diagnosis_score": 0.0,
            "diagnosis_accuracy": 0.0,
            "symptom_coverage": 0.0,
            "symptom_accuracy": 0.0,
            "format_score": 0.0,
            "acc": False,
            "extracted_diagnosis": "",
            "ground_truth": ground_truth,
            "error": str(e),
        }


# 为了兼容性，提供一个简化的接口
def compute_score(solution_str: str, ground_truth: str, patient_id: Union[str, int] = None, use_symptom_reward: bool = False) -> float:
    """
    简化的接口，直接返回分数
    
    Args:
        solution_str: 模型生成的解答
        ground_truth: 正确答案
        patient_id: 患者ID，支持格式 "{id}_conv{i}" 或直接数字ID
        use_symptom_reward: 是否启用症状识别奖励功能
        
    Returns:
        总分数（传统模式仅诊断正确率，新模式包含症状覆盖率）
    """
    result = compute_diagnosis_score(solution_str, ground_truth, patient_id=patient_id, return_details=False, use_symptom_reward=use_symptom_reward)
    if isinstance(result, dict):
        return float(result.get("score", 0.0))
    return float(result)


# 测试函数
def test_psy_rewards_with_tracking():
    """测试问诊reward函数与verl tracking系统的集成"""
    # 创建一个模拟的tracking实例
    try:
        from verl.utils.tracking import Tracking
        
        # 创建tracking实例用于测试
        tracker = Tracking(
            project_name="psy_test",
            experiment_name="tracking_integration_test",
            default_backend=["console"],  # 使用console后端进行测试
            config={"test": True}
        )
        
        print("Testing psy_rewards with verl tracking integration...")
        
        # 测试用例
        test_cases = [
            {
                "model_response": "根据患者的症状，包括焦虑、心悸、出汗等，我认为这是焦虑症。<box>焦虑症</box>",
                "ground_truth": "焦虑症",
                "patient_id": "1_conv0",
                "expected_diagnosis": True,
            },
            {
                "model_response": "患者表现出明显的抑郁倾向，悲伤、情感低落。<box>抑郁症</box>",
                "ground_truth": ["抑郁症", "重度抑郁"],
                "patient_id": "2_conv1",
                "expected_diagnosis": True,
            },
        ]
        
        for i, test_case in enumerate(test_cases):
            print(f"\nTest {i+1}: Testing with tracking integration")
            
            # 测试新的集成方式
            result = psy_reward_function(
                data_source="psy_diagnosis",
                solution_str=test_case["model_response"],
                ground_truth=test_case["ground_truth"],
                extra_info={"patient_id": test_case["patient_id"]},
                use_symptom_reward=True,
                tracker=tracker
            )
            
            print(f"  Total Score: {result.get('score', 0.0):.3f}")
            print(f"  Diagnosis Score: {result.get('diagnosis_score', 0.0):.3f}")
            print(f"  Symptom Coverage: {result.get('symptom_coverage', 0.0):.3f}")
            print(f"  Diagnosis Correct: {'✓' if result.get('acc', False) else '✗'}")
            
            # 验证指标是否被正确记录
            if 'extracted_symptoms' in result:
                print(f"  Extracted Symptoms: {result['extracted_symptoms']}")
            if 'ground_truth_symptoms' in result:
                print(f"  Ground Truth Symptoms: {result['ground_truth_symptoms']}")
        
        print("\n✓ Tracking integration test completed successfully!")
        return True
        
    except ImportError as e:
        print(f"[WARNING] Cannot test tracking integration: {e}")
        print("Falling back to traditional test...")
        return test_psy_rewards()
    except Exception as e:
        print(f"[ERROR] Tracking integration test failed: {e}")
        return False


def test_psy_rewards():
    """测试问诊reward函数，包括症状识别功能和向后兼容性"""
    # 测试用例
    test_cases = [
        {
            "model_response": "根据患者的症状，包括焦虑、心悸、出汗等，我认为这是焦虑症。<box>焦虑症</box>",
            "ground_truth": "焦虑症",
            "patient_id": "1_conv0",  # 新格式的患者ID
            "expected_diagnosis": True,
        },
        {
            "model_response": "患者表现出明显的抑郁倾向，悲伤、情感低落。<box>抑郁症</box>",
            "ground_truth": ["抑郁症", "重度抑郁"],
            "patient_id": "2_conv1",  # 新格式的患者ID
            "expected_diagnosis": True,
        },
        {
            "model_response": "这可能是双相情感障碍，患者有焦虑、激动表现。<box>双相障碍</box>",
            "ground_truth": "抑郁症",
            "patient_id": "3_conv2",  # 新格式的患者ID
            "expected_diagnosis": False,
        },
        {
            "model_response": "根据分析，患者没有明显的焦虑症状，无抑郁表现，诊断结果如下：焦虑症",
            "ground_truth": "焦虑症",
            "patient_id": 4,  # 传统数字格式也支持
            "expected_diagnosis": False,  # 没有box标签，格式不正确
        },
    ]
    
    print("Testing psy_rewards - 传统模式（不启用症状识别）...")
    for i, test_case in enumerate(test_cases):
        # 测试传统模式
        result_traditional = compute_diagnosis_score(
            test_case["model_response"], 
            test_case["ground_truth"], 
            patient_id=test_case.get("patient_id"),
            return_details=True,
            use_symptom_reward=False
        )
        
        # 测试新模式
        result_with_symptoms = compute_diagnosis_score(
            test_case["model_response"], 
            test_case["ground_truth"], 
            patient_id=test_case.get("patient_id"),
            return_details=True,
            use_symptom_reward=True
        )
        
        print(f"\nTest {i+1}:")
        print(f"  传统模式分数: {result_traditional.get('score', 0.0):.3f}")
        print(f"  新模式分数: {result_with_symptoms.get('score', 0.0):.3f}")
        print(f"  诊断正确: {'✓' if result_traditional.get('acc', False) else '✗'}")
        print(f"  症状覆盖率: {result_with_symptoms.get('symptom_coverage', 0.0):.3f}")
        
        result = result_with_symptoms  # 为了兼容后续代码
        
        # 确保result是字典类型
        if isinstance(result, dict):
            print(f"Test {i+1}:")
            print(f"  Total Score: {result.get('score', 0.0):.2f}")
            print(f"  Diagnosis Score: {result.get('diagnosis_score', 0.0):.2f}")
            print(f"  Symptom Coverage: {result.get('symptom_coverage', 0.0):.2f}")
            print(f"  Extracted Diagnosis: '{result.get('extracted_diagnosis', '')}'")
            print(f"  Ground Truth: {test_case['ground_truth']}")
            print(f"  Diagnosis Correct: {result.get('acc', False)}")
            if 'extracted_symptoms' in result:
                print(f"  Extracted Symptoms: {result['extracted_symptoms']}")
            if 'ground_truth_symptoms' in result:
                print(f"  Ground Truth Symptoms: {result['ground_truth_symptoms']}")
        else:
            print(f"Test {i+1}: Score={result}")
        print()
        
    # 测试症状提取功能
    print("Testing symptom extraction...")
    test_text = "患者有焦虑、心悸症状，但没有抑郁表现，无自杀倾向"
    test_symptoms = ["焦虑", "心悸", "抑郁", "自杀倾向", "失眠"]
    extracted = extract_symptoms_from_text(test_text, test_symptoms)
    print(f"Text: {test_text}")
    print(f"Available symptoms: {test_symptoms}")
    print(f"Extracted symptoms: {extracted}")
    print()


if __name__ == "__main__":
    # 首先测试tracking集成
    print("=" * 80)
    print("Testing PSY Rewards with VERL Tracking Integration")
    print("=" * 80)
    test_psy_rewards_with_tracking()
    
    print("\n" + "=" * 80)
    print("Testing PSY Rewards - Traditional Mode")
    print("=" * 80)
    test_psy_rewards()
