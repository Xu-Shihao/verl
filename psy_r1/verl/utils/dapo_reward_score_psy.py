#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
心理诊断任务的奖励计算函数 - DAPO版本 (简化版)
直接使用GRPO的psy_reward_function，确保完全一致
"""

import sys
import os

# 添加路径以便导入
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# 导入GRPO中的奖励函数
from psy_rewards import psy_reward_function


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
            
            print(f"[DAPO] Reward computed - Score: {result.get('score', 0.0):.3f}, "
                  f"Diagnosis: {result.get('diagnosis_score', 0.0):.3f}, "
                  f"Symptom: {result.get('symptom_coverage', 0.0):.3f}")
                  
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
