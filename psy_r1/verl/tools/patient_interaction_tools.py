"""
基于 Patient Agent 的多轮问诊工具：

- AskPatientTool：调用本地 Patient Agent，返回患者回答（无奖励）。
- DiagnoseRewardTool：接收模型最终诊断，复用 psy_rewards 计算奖励。
  - 支持 SIG (Shapley Information Gain) 过程奖励（可选）。
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, Optional, TYPE_CHECKING

import httpx
from psy_r1.verl.utils.interactive_diagnosis_reward import compute_interactive_diagnosis_reward
from verl.tools.base_tool import BaseTool

# SIG reward imports (optional, lazy loaded)
if TYPE_CHECKING:
    from psy_r1.verl.utils.sig_reward import SIGCalculator, SIGConfig

# 简单的 session 存储，按 request_id 复用状态
_SESSION_STORE: dict[str, dict[str, Any]] = {}


def _get_session(instance_id: str) -> dict[str, Any]:
    if instance_id not in _SESSION_STORE:
        _SESSION_STORE[instance_id] = {
            "messages": [],
            "ground_truth": None,
            "patient_id": None,
            "patient_version": None,
            "patient_model_name": None,
            "patient_info": None,  # 新增：患者信息文本（用于SIG）
            "predicted_diagnosis": None,
            "last_reward": None,
        }
    return _SESSION_STORE[instance_id]


class AskPatientTool(BaseTool):
    """向 Patient Agent 提问，获取患者回答。"""

    def __init__(self, config: dict, tool_schema):
        super().__init__(config, tool_schema)
        self.patient_agent_url: str = config.get("patient_agent_url", "http://localhost:8001")
        self.default_patient_version: str = config.get("default_patient_version", "v3")
        self.default_model_name: Optional[str] = config.get("default_model_name", "moonshotai/kimi-k2-0905")
        self.timeout: float = float(config.get("timeout", 15.0))
        self.max_history: int = int(config.get("max_history", 32))
        self.log_payload: bool = bool(config.get("log_payload", False))
        self.log_messages: bool = bool(config.get("log_messages", False))
        # 新增：是否在对话完成时输出完整对话（替代每次请求输出）
        self.log_complete_dialogue: bool = bool(config.get("log_complete_dialogue", True))
        # 重试机制配置
        self.max_retries: int = int(config.get("max_retries", 3))
        self.retry_delay: float = float(config.get("retry_delay", 2.0))
        self.retry_backoff: float = float(config.get("retry_backoff", 1.5))
        # 问题相似度阈值（0.0-1.0，越高越严格）
        self.similarity_threshold: float = float(config.get("similarity_threshold", 0.8))

    def _is_similar_question(self, q1: str, q2: str) -> bool:
        """
        检查两个问题是否相似（使用简单的字符重叠率）。

        Args:
            q1: 新问题
            q2: 已有问题

        Returns:
            True 如果问题相似度超过阈值
        """
        # 简化处理：去除标点和空白，转小写
        import re
        def normalize(s: str) -> str:
            s = re.sub(r'[^\w\u4e00-\u9fff]', '', s.lower())
            return s

        n1, n2 = normalize(q1), normalize(q2)
        if not n1 or not n2:
            return False

        # 如果完全相同
        if n1 == n2:
            return True

        # 使用字符级别的 Jaccard 相似度
        # 对于中文，使用2-gram
        def get_ngrams(s: str, n: int = 2) -> set:
            if len(s) < n:
                return {s}
            return {s[i:i+n] for i in range(len(s) - n + 1)}

        set1, set2 = get_ngrams(n1), get_ngrams(n2)
        intersection = len(set1 & set2)
        union = len(set1 | set2)

        if union == 0:
            return False

        similarity = intersection / union
        return similarity >= self.similarity_threshold

    async def create(
        self,
        instance_id: Optional[str] = None,
        ground_truth: Optional[str] = None,
        patient_id: Optional[str] = None,
        patient_version: Optional[str] = None,
        patient_model_name: Optional[str] = None,
        model_name: Optional[str] = None,  # 兼容数据集中的 model_name 键名
        patient_info: Optional[str] = None,  # 新增：患者信息文本（用于SIG）
        initial_messages: Optional[list[dict[str, Any]]] = None,
        **kwargs,
    ) -> str:
        instance_id = await super().create(instance_id, **kwargs)
        session = _get_session(instance_id)

        # DEBUG: 打印接收到的参数
        if not patient_id or patient_id in ['default', 'cot', 'patient123']:
            print(f"[ASK_PATIENT_CREATE_DEBUG] instance_id={instance_id}")
            print(f"  patient_id={patient_id}")
            print(f"  patient_version={patient_version}")
            print(f"  model_name={model_name}")
            print(f"  patient_model_name={patient_model_name}")
            print(f"  kwargs={kwargs}")

        session["ground_truth"] = ground_truth or session.get("ground_truth")
        session["patient_id"] = patient_id or session.get("patient_id") or "MISSING_PID"
        session["patient_version"] = patient_version or session.get("patient_version") or self.default_patient_version
        # model_name 优先于 patient_model_name（兼容数据集键名）
        session["patient_model_name"] = model_name or patient_model_name or session.get("patient_model_name") or self.default_model_name
        # 存储患者信息（用于SIG奖励计算）
        session["patient_info"] = patient_info or session.get("patient_info")
        if initial_messages:
            session["messages"] = list(initial_messages)[-self.max_history :]
        return instance_id

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[str, float, dict]:
        question = str(parameters.get("question", "")).strip()
        if not question:
            return "问题为空，未调用 patient agent。", 0.0, {}

        session = _get_session(instance_id)
        patient_id = parameters.get("patient_id") or session.get("patient_id") or "0"
        patient_version = parameters.get("patient_version") or session.get("patient_version") or self.default_patient_version
        model_name = parameters.get("model_name") or session.get("patient_model_name") or self.default_model_name

        # === 诊断日志：检查 max_history 和初始消息数量 ===
        initial_msg_count = len(session.get("messages", []))
        if initial_msg_count > self.max_history:
            print(f"[ASK_PATIENT_WARN] 消息超限: {initial_msg_count}/{self.max_history}, patient={patient_id}")

        # === 前置检查：确保对话交替且问题不重复 ===
        messages = session.get("messages", [])

        # 检查1: 确保交替对话 - 上一条消息不能是医生(user)消息
        if messages and messages[-1].get("role") == "user":
            return (
                "错误：上一条消息已经是医生提问，请等待患者回复后再提问。"
                "请调用其他工具或进行诊断。",
                0.0,
                {"error": "consecutive_doctor_message", "patient_id": patient_id}
            )

        # 检查2: 确保问题不重复 - 与之前的医生问题不能相同
        question_key = question[:200].strip().lower()  # 用前200字符比较
        for i, m in enumerate(messages):
            if m.get("role") == "user":
                existing_key = m.get("content", "")[:200].strip().lower()
                # 使用相似度检查（完全匹配或高度相似）
                if question_key == existing_key or self._is_similar_question(question, m.get("content", "")):
                    return (
                        f"错误：您提出的问题与之前的提问（第{i//2 + 1}轮）相似或重复。"
                        "请提出一个新的、不同的问题来进一步了解患者情况。",
                        0.0,
                        {"error": "duplicate_question", "patient_id": patient_id, "duplicate_of": i}
                    )

        # 追加医生提问
        session["messages"].append({"role": "user", "content": question})
        session["messages"] = session["messages"][-self.max_history :]

        # 构建 payload
        payload: Dict[str, Any] = {
            "patient_id": str(patient_id),
            "messages": session["messages"],
            "patient_version": patient_version,
        }
        if model_name:
            payload["model_name"] = model_name

        # === 请求长度和重复消息检测 ===
        messages = session["messages"]
        total_chars = sum(len(m.get("content", "")) for m in messages)
        num_messages = len(messages)

        # 检测重复消息
        seen_contents = {}
        duplicates = []
        for i, m in enumerate(messages):
            content = m.get("content", "")
            content_key = content[:200]  # 用前200字符作为key检测重复
            if content_key in seen_contents:
                duplicates.append((i, seen_contents[content_key], content_key[:50]))
            else:
                seen_contents[content_key] = i

        # 检测连续重复的医生提问
        consecutive_user_duplicates = []
        for i in range(1, len(messages)):
            if messages[i].get("role") == "user" and messages[i-1].get("role") == "user":
                if messages[i].get("content") == messages[i-1].get("content"):
                    consecutive_user_duplicates.append(i)

        # 如果发现问题，打印简洁的调试信息
        CHAR_THRESHOLD = 5000  # 5千字符阈值
        if total_chars > CHAR_THRESHOLD or duplicates or consecutive_user_duplicates:
            issues = []
            if total_chars > CHAR_THRESHOLD:
                issues.append(f"过长({total_chars}字符)")
            if duplicates:
                issues.append(f"重复({len(duplicates)}组)")
            if consecutive_user_duplicates:
                issues.append(f"连续提问({len(consecutive_user_duplicates)}处)")
            print(f"[ASK_PATIENT_WARN] patient={patient_id}, msgs={num_messages}, issues: {', '.join(issues)}")
            print(f"Overlength Message: {messages}")

        start = time.perf_counter()
        if self.log_payload:
            print(f"[ASK_PATIENT] request payload {str(patient_id)}: {payload}")

        # 带重试的请求逻辑
        last_error = None
        last_error_body = ""
        
        for attempt in range(self.max_retries):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    resp = await client.post(f"{self.patient_agent_url}/api/v1/patient/chat", json=payload)
                    resp.raise_for_status()
                    data = resp.json()
                    choices = data.get("choices") or []
                    content = (
                        choices[0]["message"]["content"]
                        if choices and choices[0].get("message", {}).get("content")
                        else data.get("content") or ""
                    )
                    
                    # 请求成功，跳出重试循环
                    latency = time.perf_counter() - start
                    content = str(content).strip()
                    session["messages"].append({"role": "assistant", "content": content})
                    session["messages"] = session["messages"][-self.max_history :]

                    metrics = {
                        "patient_id": patient_id,
                        "patient_version": patient_version,
                        "latency": latency,
                        "request_payload": payload,
                        "retry_count": attempt,
                    }
                    return content, 0.0, metrics
                    
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                try:
                    last_error_body = resp.text  # type: ignore[name-defined]
                    last_status_code = resp.status_code  # type: ignore[name-defined]
                except Exception:
                    last_error_body = ""
                    last_status_code = None

                # 简洁的重试日志
                error_brief = str(exc)[:100]
                print(f"[ASK_PATIENT_RETRY] patient={patient_id}, {attempt + 1}/{self.max_retries}, {error_brief}")

                # 如果不是最后一次重试，等待后继续
                if attempt < self.max_retries - 1:
                    wait_time = self.retry_delay * (self.retry_backoff ** attempt)
                    await asyncio.sleep(wait_time)
        
        # 简洁的失败日志
        error_brief = str(last_error)[:150]
        msg_count = len(payload.get('messages', []))
        msg_chars = sum(len(m.get('content', '')) for m in payload.get('messages', []))
        print(f"[ASK_PATIENT_FAIL] patient={patient_id}, msgs={msg_count}, chars={msg_chars}, error={error_brief}")

        # 移除刚才添加的医生问题，保持状态一致
        if session["messages"] and session["messages"][-1].get("role") == "user":
            session["messages"].pop()

        error_msg = f"调用 patient agent 失败（重试 {self.max_retries} 次后）：{last_error}"
        return (
            error_msg,
            0.0,
            {"error": str(last_error), "payload": payload, "response_text": last_error_body, "retry_count": self.max_retries},
        )

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        # 问诊工具本身不提供奖励
        return 0.0

    async def release(self, instance_id: str, **kwargs) -> None:
        # 不在此处清理，避免影响其他工具复用状态，由 DiagnoseRewardTool 统一回收
        return None


class DiagnoseRewardTool(BaseTool):
    """接收最终诊断并计算奖励。"""

    def __init__(self, config: dict, tool_schema):
        super().__init__(config, tool_schema)
        self.default_patient_version: str = config.get("default_patient_version", "v3")
        # 新增：是否在对话完成时输出完整对话
        self.log_complete_dialogue: bool = bool(config.get("log_complete_dialogue", True))
        # 严格格式检查配置（默认关闭，兼容旧版本行为）
        self.use_strict_format_check: bool = bool(config.get("use_strict_format_check", False))
        # 长度奖励配置
        self.use_length_reward: bool = bool(config.get("use_length_reward", False))
        self.length_reward_weight: float = float(config.get("length_reward_weight", 0.0))
        self.length_reward_config: dict = config.get("length_reward_config", {
            "min_turns": 10,
            "optimal_start": 15,
            "optimal_end": 25,
            "max_turns": 50,
        })
        # v4新增：格式惩罚配置
        self.format_penalty_weight: float = float(config.get("format_penalty_weight", 0.0))

        # SIG (Shapley Information Gain) 奖励配置
        self.use_sig_reward: bool = bool(config.get("use_sig_reward", False))
        self.sig_reward_config: dict = config.get("sig_reward_config", {})
        self._sig_calculator: Optional['SIGCalculator'] = None
        self._sig_initialized: bool = False

    async def create(
        self,
        instance_id: Optional[str] = None,
        ground_truth: Optional[str] = None,
        patient_id: Optional[str] = None,
        patient_version: Optional[str] = None,
        patient_info: Optional[str] = None,  # 新增：患者信息文本（用于SIG）
        model_name: Optional[str] = None,  # 兼容数据集中的 model_name 键名（虽然此工具不用）
        **kwargs,
    ) -> str:
        instance_id = await super().create(instance_id, **kwargs)
        session = _get_session(instance_id)

        # 处理 ground_truth：如果是 numpy array 或 list，转为字符串列表
        if ground_truth is not None:
            import numpy as np
            if isinstance(ground_truth, np.ndarray):
                ground_truth = ground_truth.tolist()
            elif not isinstance(ground_truth, (list, str)):
                ground_truth = str(ground_truth)

        session["ground_truth"] = ground_truth or session.get("ground_truth")
        session["patient_id"] = patient_id or session.get("patient_id")
        session["patient_version"] = patient_version or session.get("patient_version") or self.default_patient_version
        # 存储患者信息（用于SIG奖励计算）
        session["patient_info"] = patient_info or session.get("patient_info")
        return instance_id

    async def _ensure_sig_calculator(self):
        """延迟初始化 SIG 计算器。"""
        if self._sig_initialized:
            return

        if self.use_sig_reward:
            try:
                from psy_r1.verl.utils.sig_reward import SIGCalculator, SIGConfig

                sig_config = SIGConfig.from_dict(self.sig_reward_config)
                sig_config.use_sig_reward = True  # 确保开启
                self._sig_calculator = SIGCalculator(sig_config)
                await self._sig_calculator.initialize()
                print(f"[SIG] Calculator initialized successfully")
            except Exception as e:
                print(f"[SIG] Failed to initialize calculator: {e}")
                self._sig_calculator = None

        self._sig_initialized = True

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[str, float, dict]:
        diagnosis = str(parameters.get("diagnosis", "")).strip()
        session = _get_session(instance_id)
        session["predicted_diagnosis"] = diagnosis
        if not diagnosis:
            return "未收到诊断内容，奖励为 0。", 0.0, {"should_terminate": True}
        # 返回 should_terminate=True 告知 ToolAgentLoop 应该终止对话
        return "诊断已提交，对话结束。", 0.0, {"should_terminate": True}

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        session = _get_session(instance_id)
        diagnosis = session.get("predicted_diagnosis") or ""
        ground_truth = session.get("ground_truth") or ""
        patient_id = session.get("patient_id")
        patient_info = session.get("patient_info") or ""
        messages = session.get("messages", [])

        # 从 kwargs 获取完整轨迹（如果有）
        full_trajectory = kwargs.get("full_trajectory", None)

        # 计算对话轮数（只计算 user 消息数，即医生的提问次数）
        num_turns = sum(1 for msg in messages if msg.get("role") == "user")

        # 输出对话摘要（减少冗余输出，只显示关键信息）
        if self.log_complete_dialogue and messages:
            print("-" * 80)
            print(f"[DIALOGUE_SUMMARY] patient_id={patient_id}, num_turns={num_turns}, total_messages={len(messages)}")
            print(f"  [诊断] {diagnosis[:200]}..." if len(diagnosis) > 200 else f"  [诊断] {diagnosis}")
            print(f"  [真实] {ground_truth}")
            print("-" * 80)

        if not diagnosis or not ground_truth:
            session["last_reward"] = {"score": 0.0, "reason": "missing_prediction_or_ground_truth"}
            return 0.0

        # 计算基础奖励（诊断准确性 + 格式 + 长度 + 格式惩罚）
        result = compute_interactive_diagnosis_reward(
            solution_str=diagnosis,
            ground_truth=ground_truth,
            patient_id=str(patient_id) if patient_id is not None else None,
            full_trajectory=full_trajectory,
            num_turns=num_turns,
            use_strict_format_check=self.use_strict_format_check,
            use_length_reward=self.use_length_reward,
            length_reward_weight=self.length_reward_weight,
            length_reward_config=self.length_reward_config,
            format_penalty_weight=self.format_penalty_weight,  # v4新增
            return_details=True,
            debug=self.log_complete_dialogue,
        )
        base_score = float(result.get("score", 0.0))

        # 计算 SIG (Shapley Information Gain) 过程奖励
        sig_score = 0.0
        sig_details = {}
        sig_result = None  # 初始化，防止异常时 UnboundLocalError
        if self.use_sig_reward and patient_id and patient_info:
            await self._ensure_sig_calculator()
            if self._sig_calculator is not None:
                try:
                    sig_result = await self._sig_calculator.compute_sig_reward(
                        patient_id=str(patient_id),
                        patient_info=patient_info,
                        dialogue_messages=messages,
                        ground_truth_diagnosis=str(ground_truth) if isinstance(ground_truth, list) else ground_truth,
                        predicted_diagnosis=diagnosis,
                    )
                    sig_score = sig_result.total_sig_reward
                    sig_details = sig_result.component_scores
                    result["sig_score"] = sig_score
                    result["sig_details"] = sig_details
                    result["sig_per_turn_reward"] = sig_result.per_turn_reward
                except Exception as e:
                    print(f"[SIG] Error computing SIG reward: {e}")
                    import traceback
                    traceback.print_exc()

        # 合并最终奖励
        final_score = base_score + sig_score
        result["score"] = final_score
        result["base_score_without_sig"] = base_score

        # 如果启用 token-level reward，存储必要的信息供后续计算使用
        if self.use_sig_reward and self.sig_reward_config.get("use_token_level_reward", False):
            result["sig_token_level_info"] = {
                "patient_id": str(patient_id) if patient_id else None,
                "patient_info": patient_info,
                "dialogue_messages": messages,
                "ground_truth_diagnosis": str(ground_truth) if isinstance(ground_truth, list) else ground_truth,
                "predicted_diagnosis": diagnosis,
                "final_diagnosis_correct": sig_result.final_diagnosis_correct if sig_result else False,
                "per_turn_sig": sig_result.per_turn_sig if sig_result else [],
            }

        session["last_reward"] = result

        # 输出奖励分数和格式检查详情
        if self.log_complete_dialogue:
            score = result.get("score", 0.0)
            base_score_display = result.get("base_score", 0.0)
            length_score = result.get("length_score", 0.0)
            format_score = result.get("format_score", 0.0)
            think_valid = result.get("think_format_valid")
            json_valid = result.get("json_format_valid")
            diagnose_valid = result.get("diagnose_format_valid")

            print(f"[REWARD] patient_id={patient_id}, final_score={score:.4f}")
            if self.use_length_reward:
                length_details = result.get("length_reward_details", {})
                print(f"  Base score: {base_score_display:.4f}, Length score: {length_score:.4f} (stage: {length_details.get('stage', 'N/A')})")
                print(f"  Turns: {num_turns}, Weight: {self.length_reward_weight:.3f}")
            if self.use_sig_reward:
                print(f"  SIG score: {sig_score:.4f}")
                if sig_details:
                    print(f"  SIG details: {sig_details}")
            format_mode = "strict" if self.use_strict_format_check else "simple"
            format_penalty = result.get("format_penalty", 0.0)
            print(f"  Format score: {format_score:.1f} (mode: {format_mode})")
            if self.format_penalty_weight > 0:
                print(f"  Format penalty: {format_penalty:.4f} (weight: {self.format_penalty_weight:.3f})")
            print(f"  Format check: think={think_valid}, json={json_valid}, diagnose={diagnose_valid}")

            # 输出格式错误详情
            if result.get("think_errors"):
                print(f"  Think errors: {result['think_errors']}")
            if result.get("json_errors"):
                print(f"  JSON errors: {result['json_errors']}")
            if result.get("diagnose_errors"):
                print(f"  Diagnose errors: {result['diagnose_errors']}")

        return float(result.get("score", 0.0))

    async def release(self, instance_id: str, **kwargs) -> None:
        _SESSION_STORE.pop(instance_id, None)
        # 注意：不在这里关闭 SIG calculator，因为它可能被多个实例复用