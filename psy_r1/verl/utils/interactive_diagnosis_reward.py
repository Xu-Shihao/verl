"""
基于最终诊断结果的交互式问诊奖励函数。

仅使用已有的诊断奖励逻辑（psy_rewards.compute_diagnosis_score），
在多轮问诊结束后对模型输出的 ICD-10 诊断进行评分。
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from psy_r1.verl.utils.psy_rewards import compute_diagnosis_score


def compute_interactive_diagnosis_reward(
    solution_str: str,
    ground_truth: str,
    *,
    patient_id: Optional[str] = None,
    return_details: bool = True,
) -> Dict[str, Any]:
    """
    基于 psy_rewards 的诊断奖励包装器。

    Args:
        solution_str: 模型的最终诊断输出。
        ground_truth: 数据集中标注的 ICD-10 诊断。
        patient_id: 可选的患者 ID，便于日志或调试。
        return_details: 是否返回完整的细节字典。

    Returns:
        包含分数与诊断准确性的字典，兼容 psy_rewards 的返回格式。
    """
    result = compute_diagnosis_score(
        model_response=solution_str,
        ground_truth=ground_truth,
        patient_id=patient_id,
        return_details=True,
        use_symptom_reward=False,
    )

    if not return_details:
        return {"score": float(result.get("score", 0.0))}

    # 补充最小字段，避免调用方缺少关键信息
    return {
        "score": float(result.get("score", 0.0)),
        "acc": bool(result.get("acc", False)),
        "format_score": float(result.get("format_score", 0.0)),
        "extracted_diagnosis": result.get("extracted_diagnosis", ""),
        "ground_truth": ground_truth,
        "patient_id": patient_id,
    }
