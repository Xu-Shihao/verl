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
from typing import Union, Dict, Any, List


def extract_diagnosis_from_box(text: str) -> str:
    """
    从文本中提取 <box></box> 标签内的诊断结果
    
    Args:
        text: 包含诊断结果的文本
        
    Returns:
        提取的诊断结果，如果没有找到则返回空字符串
    """
    # 使用正则表达式匹配 <box></box> 标签
    pattern = r'<box>(.*?)</box>'
    matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
    
    if matches:
        # 返回最后一个匹配的内容，去除前后空白
        return matches[-1].strip()
    
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
        
        # 包含关系匹配（预测结果包含正确答案的关键词）
        if gt_norm and gt_norm in pred_norm:
            return True
        
        # 关键词匹配（正确答案的关键词在预测结果中）
        gt_keywords = gt_norm.split()
        if len(gt_keywords) > 0:
            # 至少匹配一半的关键词
            matched_keywords = sum(1 for keyword in gt_keywords if keyword in pred_norm)
            if matched_keywords >= len(gt_keywords) * 0.5:
                return True
    
    return False


def compute_diagnosis_score(
    model_response: str, 
    ground_truth: Union[str, List[str]], 
    return_details: bool = False
) -> Union[float, Dict[str, Any]]:
    """
    计算问诊诊断结果的正确率分数
    
    Args:
        model_response: 模型的完整回答
        ground_truth: 正确的诊断结果
        return_details: 是否返回详细信息
        
    Returns:
        如果return_details为False，返回分数(0.0或1.0)
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
        
        # 计算分数
        score = 1.0 if is_correct else 0.0
        
        if return_details:
            return {
                "score": score,
                "format_score": 1.0 if format_correct else 0.0,
                "acc": is_correct,
                "extracted_diagnosis": extracted_diagnosis,
                "ground_truth": ground_truth,
            }
        else:
            return score
            
    except Exception as e:
        print(f"[ERROR] Error in compute_diagnosis_score: {str(e)}")
        traceback.print_exc()
        
        if return_details:
            return {
                "score": 0.0,
                "format_score": 0.0,
                "acc": False,
                "extracted_diagnosis": "",
                "ground_truth": ground_truth,
                "error": str(e),
            }
        else:
            return 0.0


def psy_reward_function(data_source: str, solution_str: str, ground_truth: str, extra_info=None):
    """
    问诊场景的reward函数，兼容现有的reward函数接口
    
    Args:
        data_source: 数据源标识
        solution_str: 模型生成的解答
        ground_truth: 正确答案
        extra_info: 额外信息（可选）
        
    Returns:
        包含分数和额外信息的字典
    """
    try:
        # 计算诊断正确率
        result = compute_diagnosis_score(solution_str, ground_truth, return_details=True)
        
        # 确保result是字典类型
        if isinstance(result, dict):
            # 如果有额外信息，可以在这里处理
            if extra_info:
                result["extra_info"] = extra_info
            return result
        else:
            # 如果不是字典，创建一个标准格式的字典
            return {
                "score": float(result),
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
            "format_score": 0.0,
            "acc": False,
            "extracted_diagnosis": "",
            "ground_truth": ground_truth,
            "error": str(e),
        }


# 为了兼容性，提供一个简化的接口
def compute_score(solution_str: str, ground_truth: str) -> float:
    """
    简化的接口，直接返回分数
    
    Args:
        solution_str: 模型生成的解答
        ground_truth: 正确答案
        
    Returns:
        正确率分数 (0.0 或 1.0)
    """
    result = compute_diagnosis_score(solution_str, ground_truth, return_details=False)
    if isinstance(result, dict):
        return float(result.get("score", 0.0))
    return float(result)


# 测试函数
def test_psy_rewards():
    """测试问诊reward函数"""
    # 测试用例
    test_cases = [
        {
            "model_response": "根据患者的症状，我认为这是焦虑症。<box>焦虑症</box>",
            "ground_truth": "焦虑症",
            "expected_score": 1.0,
        },
        {
            "model_response": "患者表现出明显的抑郁倾向。<box>抑郁症</box>",
            "ground_truth": ["抑郁症", "重度抑郁"],
            "expected_score": 1.0,
        },
        {
            "model_response": "这可能是双相情感障碍。<box>双相障碍</box>",
            "ground_truth": "抑郁症",
            "expected_score": 0.0,
        },
        {
            "model_response": "根据分析，患者的诊断结果如下：焦虑症",
            "ground_truth": "焦虑症",
            "expected_score": 0.0,  # 没有box标签，格式不正确
        },
    ]
    
    print("Testing psy_rewards...")
    for i, test_case in enumerate(test_cases):
        result = compute_diagnosis_score(
            test_case["model_response"], 
            test_case["ground_truth"], 
            return_details=True
        )
        # 确保result是字典类型
        if isinstance(result, dict):
            print(f"Test {i+1}: Score={result['score']}, Expected={test_case['expected_score']}")
            print(f"  Extracted: '{result['extracted_diagnosis']}'")
            print(f"  Ground Truth: {test_case['ground_truth']}")
            print(f"  Correct: {result['score'] == test_case['expected_score']}")
        else:
            print(f"Test {i+1}: Score={result}, Expected={test_case['expected_score']}")
            print(f"  Extracted: 'N/A'")
            print(f"  Ground Truth: {test_case['ground_truth']}")
            print(f"  Correct: {result == test_case['expected_score']}")
        print()


if __name__ == "__main__":
    test_psy_rewards()
