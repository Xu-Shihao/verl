"""
基于最终诊断结果的交互式问诊奖励函数。

仅使用已有的诊断奖励逻辑（psy_rewards.compute_diagnosis_score），
在多轮问诊结束后对模型输出的 ICD-10 诊断进行评分。

格式要求（三项全部满足才算格式正确）：
1. 每一轮工具调用前需要包含一个且只有一个 <think>...</think> 区块且闭合
2. 每一个 <tool_call>...</tool_call> JSON 结构合法
3. 最后一个 <tool_call> 的 name 为 do_diagnose，且 diagnosis 里面包含 <box>...</box> 区块
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from psy_r1.verl.utils.psy_rewards import compute_diagnosis_score


def _extract_assistant_content(full_trajectory: str) -> str:
    """
    从完整轨迹中提取 assistant 消息部分的内容，排除 system prompt。
    
    轨迹格式为 "role: content\nrole: content\n..."，
    只提取 role 为 assistant 的部分进行格式检查。
    
    Args:
        full_trajectory: 完整的对话轨迹文本
        
    Returns:
        只包含 assistant 消息内容的文本
    """
    # 方法1：找到第一个 "assistant:" 的位置，只检查从那之后的内容
    # 这样可以排除 system prompt 和 user 消息中的示例标签
    
    # 匹配 "assistant:" 开头的段落（考虑可能的换行）
    # 由于轨迹格式是 "role: content"，我们查找 "assistant:" 标记
    assistant_marker = "assistant:"
    
    # 找到第一个 assistant 消息的位置
    first_assistant_pos = full_trajectory.find(assistant_marker)
    
    if first_assistant_pos == -1:
        # 如果没有找到 assistant: 标记，可能是其他格式，返回原始轨迹
        # 但跳过可能的 system 部分
        return full_trajectory
    
    # 只返回从第一个 assistant 消息开始的部分
    return full_trajectory[first_assistant_pos:]


def check_think_format(full_trajectory: str) -> Tuple[bool, List[str]]:
    """
    检查每一轮工具调用前是否包含一个且只有一个 <think>...</think> 区块。
    
    注意：只检查 assistant 消息部分，排除 system prompt 中的示例标签。
    
    Args:
        full_trajectory: 完整的对话轨迹文本
        
    Returns:
        (is_valid, error_messages): 是否有效以及错误信息列表
    """
    errors = []
    
    # 只检查 assistant 消息部分，排除 system prompt 中的示例标签
    trajectory_to_check = _extract_assistant_content(full_trajectory)
    
    # 按 assistant 回复分割（每次 assistant 回复是一轮）
    # 查找所有包含 <tool_call> 的 assistant 回复段落
    # 使用正则匹配 assistant 回复中的 tool_call 部分
    
    # 方法：找到所有 <tool_call>...</tool_call> 区块，然后检查每个之前是否有且仅有一个 <think>
    tool_call_pattern = re.compile(r'<tool_call>(.*?)</tool_call>', re.DOTALL)
    think_pattern = re.compile(r'<think>(.*?)</think>', re.DOTALL)
    
    # 找到所有 tool_call 的位置
    tool_calls = list(tool_call_pattern.finditer(trajectory_to_check))
    
    if not tool_calls:
        # 没有工具调用，检查是否至少有 think 标签
        thinks = think_pattern.findall(trajectory_to_check)
        if not thinks:
            errors.append("未找到任何 <tool_call> 或 <think> 标签")
        return len(errors) == 0, errors
    
    # 对于每个 tool_call，检查在它之前（但在上一个 tool_response 之后）是否有且仅有一个 think
    prev_end = 0
    for i, tc_match in enumerate(tool_calls):
        tc_start = tc_match.start()
        
        # 找到这个 tool_call 之前的文本段落
        # 如果有上一个 tool_response，从那里开始；否则从头开始
        segment_before_tc = trajectory_to_check[prev_end:tc_start]
        
        # 在这个段落中查找 <think>...</think>
        thinks_in_segment = list(think_pattern.finditer(segment_before_tc))
        
        if len(thinks_in_segment) == 0:
            errors.append(f"第 {i+1} 个 <tool_call> 之前没有找到 <think>...</think> 区块")
        elif len(thinks_in_segment) > 1:
            errors.append(f"第 {i+1} 个 <tool_call> 之前有 {len(thinks_in_segment)} 个 <think> 区块，应该只有 1 个")
        
        # 更新 prev_end 到这个 tool_call 对应的 tool_response 结束位置
        # 查找对应的 </tool_response>
        tool_response_end = trajectory_to_check.find('</tool_response>', tc_match.end())
        if tool_response_end != -1:
            prev_end = tool_response_end + len('</tool_response>')
        else:
            prev_end = tc_match.end()
    
    # 检查是否有未闭合的 think 标签（只检查 assistant 部分）
    open_thinks = trajectory_to_check.count('<think>')
    close_thinks = trajectory_to_check.count('</think>')
    if open_thinks != close_thinks:
        errors.append(f"<think> 标签未正确闭合: 开始标签 {open_thinks} 个，结束标签 {close_thinks} 个")
    
    return len(errors) == 0, errors


def check_tool_call_json_format(full_trajectory: str) -> Tuple[bool, List[str]]:
    """
    检查每一个 <tool_call>...</tool_call> 内的 JSON 结构是否合法。
    
    Args:
        full_trajectory: 完整的对话轨迹文本
        
    Returns:
        (is_valid, error_messages): 是否有效以及错误信息列表
    """
    errors = []
    
    tool_call_pattern = re.compile(r'<tool_call>(.*?)</tool_call>', re.DOTALL)
    tool_calls = tool_call_pattern.findall(full_trajectory)
    
    if not tool_calls:
        errors.append("未找到任何 <tool_call>...</tool_call> 区块")
        return False, errors
    
    for i, tc_content in enumerate(tool_calls):
        tc_content = tc_content.strip()
        if not tc_content:
            errors.append(f"第 {i+1} 个 <tool_call> 内容为空")
            continue
        
        try:
            parsed = json.loads(tc_content)
            # 检查必须的字段
            if not isinstance(parsed, dict):
                errors.append(f"第 {i+1} 个 <tool_call> 内容不是 JSON 对象: {tc_content[:50]}...")
            elif 'name' not in parsed:
                errors.append(f"第 {i+1} 个 <tool_call> 缺少 'name' 字段")
            elif 'arguments' not in parsed:
                errors.append(f"第 {i+1} 个 <tool_call> 缺少 'arguments' 字段")
        except json.JSONDecodeError as e:
            errors.append(f"第 {i+1} 个 <tool_call> JSON 解析失败: {e}")
    
    return len(errors) == 0, errors


def check_final_diagnose_format(full_trajectory: str) -> Tuple[bool, List[str]]:
    """
    检查最后一个 <tool_call> 是否为 do_diagnose，且 diagnosis 参数包含 <box>...</box>。
    
    Args:
        full_trajectory: 完整的对话轨迹文本
        
    Returns:
        (is_valid, error_messages): 是否有效以及错误信息列表
    """
    errors = []
    
    tool_call_pattern = re.compile(r'<tool_call>(.*?)</tool_call>', re.DOTALL)
    tool_calls = tool_call_pattern.findall(full_trajectory)
    
    if not tool_calls:
        errors.append("未找到任何 <tool_call>...</tool_call> 区块")
        return False, errors
    
    # 检查最后一个 tool_call
    last_tc = tool_calls[-1].strip()
    
    try:
        parsed = json.loads(last_tc)
        
        # 检查 name 是否为 do_diagnose
        tool_name = parsed.get('name', '')
        if tool_name != 'do_diagnose':
            errors.append(f"最后一个 <tool_call> 的 name 应为 'do_diagnose'，实际为 '{tool_name}'")
            return False, errors
        
        # 检查 arguments 中的 diagnosis 是否包含 <box>...</box>
        arguments = parsed.get('arguments', {})
        diagnosis = arguments.get('diagnosis', '')
        
        if not diagnosis:
            errors.append("do_diagnose 的 diagnosis 参数为空")
            return False, errors
        
        # 检查 <box>...</box> 标签
        box_pattern = re.compile(r'<box>(.*?)</box>', re.DOTALL)
        box_matches = box_pattern.findall(diagnosis)
        
        if not box_matches:
            errors.append("do_diagnose 的 diagnosis 中未找到 <box>...</box> 区块")
        elif not box_matches[0].strip():
            errors.append("do_diagnose 的 diagnosis 中 <box>...</box> 内容为空")
        
    except json.JSONDecodeError as e:
        errors.append(f"最后一个 <tool_call> JSON 解析失败: {e}")
    
    return len(errors) == 0, errors


def compute_strict_format_score(
    full_trajectory: str,
    debug: bool = False
) -> Dict[str, Any]:
    """
    计算严格的格式分数，三项全部满足才算格式正确。
    
    格式要求：
    1. 每一轮工具调用前需要包含一个且只有一个 <think>...</think> 区块且闭合
    2. 每一个 <tool_call>...</tool_call> JSON 结构合法
    3. 最后一个 <tool_call> 的 name 为 do_diagnose，且 diagnosis 里面包含 <box>...</box> 区块
    
    Args:
        full_trajectory: 完整的对话轨迹文本
        debug: 是否输出调试信息
        
    Returns:
        包含各项格式检查结果的字典
    """
    # 检查三项格式要求
    think_valid, think_errors = check_think_format(full_trajectory)
    json_valid, json_errors = check_tool_call_json_format(full_trajectory)
    diagnose_valid, diagnose_errors = check_final_diagnose_format(full_trajectory)
    
    # 全部满足才算格式正确
    all_valid = think_valid and json_valid and diagnose_valid
    format_score = 1.0 if all_valid else 0.0
    
    result = {
        "format_score": format_score,
        "think_format_valid": think_valid,
        "json_format_valid": json_valid,
        "diagnose_format_valid": diagnose_valid,
        "think_errors": think_errors,
        "json_errors": json_errors,
        "diagnose_errors": diagnose_errors,
    }
    
    if debug:
        print(f"[FORMAT CHECK] think_valid={think_valid}, json_valid={json_valid}, diagnose_valid={diagnose_valid}")
        if think_errors:
            print(f"  Think errors: {think_errors}")
        if json_errors:
            print(f"  JSON errors: {json_errors}")
        if diagnose_errors:
            print(f"  Diagnose errors: {diagnose_errors}")
    
    return result


def compute_length_reward(
    num_turns: int,
    min_turns: int = 10,
    optimal_start: int = 15,
    optimal_end: int = 25,
    max_turns: int = 50,
    debug: bool = False,
) -> Dict[str, Any]:
    """
    计算基于对话轮数的梯形奖励函数。
    
    奖励曲线设计：
    - [0, min_turns)：线性增长 0.0 -> 0.5（鼓励最低限度的信息收集）
    - [min_turns, optimal_start)：线性增长 0.5 -> 1.0（逐步接近最优）
    - [optimal_start, optimal_end]：保持 1.0（最优区间，满分平台期）
    - (optimal_end, max_turns]：线性下降 1.0 -> 0.5（轻微惩罚冗余对话）
    - >max_turns：保持 0.5（避免过度惩罚）
    
    Args:
        num_turns: 实际对话轮数（assistant 回复次数）
        min_turns: 最小合理轮数，默认10轮
        optimal_start: 最优区间开始，默认15轮
        optimal_end: 最优区间结束，默认25轮
        max_turns: 最大合理轮数，默认50轮
        debug: 是否输出调试信息
        
    Returns:
        包含 length_score 和调试信息的字典
    """
    if num_turns < min_turns:
        # 阶段1：线性增长 0.0 -> 0.5
        length_score = 0.5 * (num_turns / min_turns)
        stage = "too_short"
    elif num_turns < optimal_start:
        # 阶段2：线性增长 0.5 -> 1.0
        progress = (num_turns - min_turns) / (optimal_start - min_turns)
        length_score = 0.5 + 0.5 * progress
        stage = "approaching_optimal"
    elif num_turns <= optimal_end:
        # 阶段3：最优区间，满分
        length_score = 1.0
        stage = "optimal"
    elif num_turns <= max_turns:
        # 阶段4：线性下降 1.0 -> 0.5
        progress = (num_turns - optimal_end) / (max_turns - optimal_end)
        length_score = 1.0 - 0.5 * progress
        stage = "slightly_long"
    else:
        # 超过最大轮数，保持0.5
        length_score = 0.5
        stage = "too_long"
    
    result = {
        "length_score": float(length_score),
        "num_turns": num_turns,
        "stage": stage,
        "min_turns": min_turns,
        "optimal_start": optimal_start,
        "optimal_end": optimal_end,
        "max_turns": max_turns,
    }
    
    if debug:
        print(f"[LENGTH REWARD] num_turns={num_turns}, score={length_score:.3f}, stage={stage}")
    
    return result


def compute_simple_format_score(
    solution_str: str,
    debug: bool = False
) -> Dict[str, Any]:
    """
    计算简单的格式分数，只检查 solution_str 中是否有 <box>...</box> 标签。
    
    这是一个宽松的格式检查，适用于不需要严格格式要求的场景。
    
    Args:
        solution_str: 模型的最终诊断输出
        debug: 是否输出调试信息
        
    Returns:
        包含格式检查结果的字典
    """
    has_box = '<box>' in solution_str and '</box>' in solution_str
    format_score = 1.0 if has_box else 0.0
    
    result = {
        "format_score": format_score,
        "think_format_valid": None,  # 简单模式不检查 think
        "json_format_valid": None,   # 简单模式不检查 JSON
        "diagnose_format_valid": has_box,
        "think_errors": [],
        "json_errors": [],
        "diagnose_errors": [] if has_box else ["简单模式：未找到 <box>...</box> 标签"],
    }
    
    if debug:
        print(f"[SIMPLE FORMAT CHECK] has_box={has_box}, format_score={format_score}")
    
    return result


def compute_interactive_diagnosis_reward(
    solution_str: str,
    ground_truth: str,
    *,
    patient_id: Optional[str] = None,
    full_trajectory: Optional[str] = None,
    num_turns: Optional[int] = None,
    use_strict_format_check: bool = False,
    use_length_reward: bool = False,
    length_reward_weight: float = 0.0,
    length_reward_config: Optional[Dict[str, int]] = None,
    return_details: bool = True,
    debug: bool = False,
) -> Dict[str, Any]:
    """
    基于 psy_rewards 的诊断奖励包装器，增加了可配置的格式检查和可选的长度奖励。

    Args:
        solution_str: 模型的最终诊断输出（do_diagnose 的 diagnosis 参数内容）。
        ground_truth: 数据集中标注的 ICD-10 诊断。
        patient_id: 可选的患者 ID，便于日志或调试。
        full_trajectory: 完整的对话轨迹文本，用于格式检查。
        num_turns: 对话轮数（assistant 回复次数），用于长度奖励计算。
        use_strict_format_check: 是否启用严格格式检查（默认 False）。
            - True: 检查 think/tool_call/diagnose 三项格式要求
            - False: 仅检查 <box>...</box> 标签
        use_length_reward: 是否启用长度奖励。
        length_reward_weight: 长度奖励的权重系数。
        length_reward_config: 长度奖励的配置参数（min_turns, optimal_start, optimal_end, max_turns）。
        return_details: 是否返回完整的细节字典。
        debug: 是否输出调试信息。

    Returns:
        包含分数与诊断准确性的字典，兼容 psy_rewards 的返回格式。
    """
    # 计算诊断分数
    result = compute_diagnosis_score(
        model_response=solution_str,
        ground_truth=ground_truth,
        patient_id=patient_id,
        return_details=True,
        use_symptom_reward=False,
    )
    
    # 根据配置选择格式检查方式
    if use_strict_format_check and full_trajectory:
        # 严格格式检查：检查 think/tool_call/diagnose 三项格式要求
        format_result = compute_strict_format_score(full_trajectory, debug=debug)
        format_score = format_result["format_score"]
    else:
        # 简单格式检查：只检查 <box>...</box> 标签
        format_result = compute_simple_format_score(solution_str, debug=debug)
        format_score = format_result["format_score"]
    
    # 计算长度奖励（如果启用）
    length_reward_result = {}
    if use_length_reward and num_turns is not None:
        # 使用配置参数或默认值
        config = length_reward_config or {}
        length_reward_result = compute_length_reward(
            num_turns=num_turns,
            min_turns=config.get("min_turns", 10),
            optimal_start=config.get("optimal_start", 15),
            optimal_end=config.get("optimal_end", 25),
            max_turns=config.get("max_turns", 50),
            debug=debug,
        )
        length_score = length_reward_result["length_score"]
    else:
        length_score = 0.0
        length_reward_result = {"length_score": 0.0, "num_turns": num_turns or 0, "stage": "disabled"}
    
    # 计算最终分数
    base_score = float(result.get("score", 0.0))
    if use_length_reward and length_reward_weight > 0:
        final_score = base_score + length_reward_weight * length_score
    else:
        final_score = base_score
    
    if debug and use_length_reward:
        print(f"[REWARD COMPOSITION] base={base_score:.3f}, length={length_score:.3f}, "
              f"weight={length_reward_weight:.3f}, final={final_score:.3f}")

    if not return_details:
        return {"score": final_score}

    # 补充最小字段，避免调用方缺少关键信息
    return {
        "score": final_score,
        "base_score": base_score,
        "length_score": length_score,
        "length_reward_weight": length_reward_weight,
        "use_length_reward": use_length_reward,
        "use_strict_format_check": use_strict_format_check,
        "acc": bool(result.get("acc", False)),
        "format_score": format_score,
        "extracted_diagnosis": result.get("extracted_diagnosis", ""),
        "ground_truth": ground_truth,
        "patient_id": patient_id,
        # 格式检查详情
        "think_format_valid": format_result.get("think_format_valid"),
        "json_format_valid": format_result.get("json_format_valid"),
        "diagnose_format_valid": format_result.get("diagnose_format_valid"),
        "think_errors": format_result.get("think_errors", []),
        "json_errors": format_result.get("json_errors", []),
        "diagnose_errors": format_result.get("diagnose_errors", []),
        # 长度奖励详情
        "length_reward_details": length_reward_result,
    }
