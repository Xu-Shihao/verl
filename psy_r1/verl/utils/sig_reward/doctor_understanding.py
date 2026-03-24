# Copyright 2024 Shihao Xu
# SIG Reward Module - Doctor Understanding Generation

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import numpy as np

from .sig_config import SIGConfig


@dataclass
class DoctorUnderstanding:
    """
    Represents the doctor's understanding at turn t.

    This captures what the doctor has learned about the patient
    from the dialogue up to this point.

    Attributes:
        turn_index: The turn number (0-indexed)
        dialogue_history: Messages up to this turn
        understanding_text: Generated summary of understanding
        fact_coverage: Mapping of fact_id -> is_covered
        coverage_vector: Binary numpy array for quick computation
    """
    turn_index: int
    dialogue_history: List[Dict[str, str]]
    understanding_text: str
    fact_coverage: Dict[str, bool] = field(default_factory=dict)
    coverage_vector: Optional[np.ndarray] = None

    def get_coverage_count(self) -> int:
        """Get the number of facts covered."""
        return sum(1 for v in self.fact_coverage.values() if v)

    def get_coverage_ratio(self) -> float:
        """Get the ratio of facts covered."""
        if not self.fact_coverage:
            return 0.0
        return self.get_coverage_count() / len(self.fact_coverage)


# Prompt template for doctor understanding generation
UNDERSTANDING_PROMPT_TEMPLATE = """你是一位精神科医生的助手，正在分析医患对话记录。

请根据以下对话历史，总结医生目前对患者病情的了解。

对话历史：
{dialogue_history}

请从以下方面总结医生当前掌握的信息：

1. **症状信息**：患者报告的症状、症状持续时间、严重程度
2. **否认的症状**：患者明确表示没有的症状
3. **病史信息**：既往病史、家族史、治疗史
4. **心理状态**：情绪、认知功能、行为表现
5. **社会功能**：工作状况、人际关系、日常生活
6. **其他临床相关信息**

医生当前了解的信息："""


class DoctorUnderstandingGenerator:
    """
    Generate doctor understanding from dialogue history.

    This class uses an LLM to summarize what the doctor has learned
    from the dialogue up to each turn.
    """

    def __init__(self, config: SIGConfig):
        """
        Initialize the understanding generator.

        Args:
            config: SIG configuration containing LLM settings
        """
        self.config = config

    def _format_dialogue_history(self, messages: List[Dict[str, str]]) -> str:
        """
        Format dialogue history for the prompt.

        Args:
            messages: List of message dictionaries with 'role' and 'content'

        Returns:
            Formatted dialogue string
        """
        if not messages:
            return "（对话尚未开始）"

        formatted_lines = []
        for i, msg in enumerate(messages):
            role = msg.get('role', 'unknown')
            content = msg.get('content', '')

            # Map roles to readable labels
            role_label = {
                'user': '医生',
                'assistant': '患者',
                'system': '系统',
                'tool': '工具'
            }.get(role, role)

            # Truncate very long content
            if len(content) > 500:
                content = content[:500] + "..."

            formatted_lines.append(f"[{role_label}]: {content}")

        return "\n".join(formatted_lines)

    def _extract_doctor_questions(self, messages: List[Dict[str, str]]) -> List[str]:
        """
        Extract doctor's questions from the dialogue.

        Args:
            messages: List of message dictionaries

        Returns:
            List of doctor's questions
        """
        questions = []
        for msg in messages:
            if msg.get('role') == 'user':
                questions.append(msg.get('content', ''))
        return questions

    async def generate_understanding(
        self,
        dialogue_history: List[Dict[str, str]],
        turn_index: int,
        llm_client: Any,
    ) -> DoctorUnderstanding:
        """
        Generate doctor understanding from dialogue history.

        Args:
            dialogue_history: Messages up to turn t
            turn_index: The turn number (0-indexed)
            llm_client: Async LLM client for making API calls

        Returns:
            DoctorUnderstanding object containing the generated summary
        """
        # Format dialogue for prompt
        formatted_dialogue = self._format_dialogue_history(dialogue_history)

        # Prepare prompt
        prompt = UNDERSTANDING_PROMPT_TEMPLATE.format(
            dialogue_history=formatted_dialogue
        )

        try:
            # Call LLM
            response = await llm_client.complete(
                prompt=prompt,
                model=self.config.understanding_model,
                temperature=0.3,  # Low temperature for consistency
                max_tokens=1024,
            )

            understanding_text = response.strip()

        except Exception as e:
            print(f"[SIG_UNDERSTAND] Error generating understanding at turn {turn_index}: {e}")
            # Fallback: use raw dialogue as understanding
            understanding_text = f"对话轮次 {turn_index}: {formatted_dialogue}"

        return DoctorUnderstanding(
            turn_index=turn_index,
            dialogue_history=dialogue_history.copy(),
            understanding_text=understanding_text,
            fact_coverage={},  # Will be filled by fact checker
            coverage_vector=None
        )

    async def generate_understandings_for_trajectory(
        self,
        full_dialogue: List[Dict[str, str]],
        llm_client: Any,
    ) -> List[DoctorUnderstanding]:
        """
        Generate doctor understandings for each turn in a trajectory.

        This creates U_0, U_1, ..., U_T where U_t represents the
        understanding after turn t.

        Optimized: Uses asyncio.gather for parallel LLM calls (aligned with ProMed).

        Args:
            full_dialogue: Complete dialogue history
            llm_client: Async LLM client

        Returns:
            List of DoctorUnderstanding objects, one per turn
        """
        import asyncio

        # Initial understanding (before any dialogue)
        initial = DoctorUnderstanding(
            turn_index=-1,
            dialogue_history=[],
            understanding_text="医生尚未获得任何患者信息。",
            fact_coverage={},
            coverage_vector=None
        )

        # Identify turn boundaries (each doctor question is a turn)
        turn_boundaries = []
        current_turn = []

        for msg in full_dialogue:
            current_turn.append(msg)
            # A turn ends after patient's response (assistant role)
            if msg.get('role') == 'assistant':
                turn_boundaries.append(current_turn.copy())

        if not turn_boundaries:
            return [initial]

        # Prepare all histories for parallel generation
        all_tasks = []
        for turn_idx in range(len(turn_boundaries)):
            # Accumulate messages up to this turn
            history_up_to_turn = []
            for prev_turn in turn_boundaries[:turn_idx + 1]:
                history_up_to_turn.extend(prev_turn)

            # Create async task for each turn
            task = self.generate_understanding(
                dialogue_history=history_up_to_turn,
                turn_index=turn_idx,
                llm_client=llm_client,
            )
            all_tasks.append(task)

        # Execute all tasks in parallel using asyncio.gather (aligned with ProMed's Pool approach)
        turn_understandings = await asyncio.gather(*all_tasks)

        # Combine initial understanding with generated ones
        understandings = [initial] + list(turn_understandings)

        if self.config.log_sig_details:
            print(f"[SIG_UNDERSTAND] Generated {len(understandings)} understandings "
                  f"(1 initial + {len(turn_boundaries)} turns) [PARALLEL]")

        return understandings

    async def batch_generate_understandings(
        self,
        trajectories: List[List[Dict[str, str]]],
        llm_client: Any,
    ) -> List[List[DoctorUnderstanding]]:
        """
        Batch generate understandings for multiple trajectories.

        This is more efficient than calling generate_understandings_for_trajectory
        multiple times as it can batch LLM calls.

        Args:
            trajectories: List of dialogue trajectories
            llm_client: Async LLM client

        Returns:
            List of understanding lists, one per trajectory
        """
        # For now, just process sequentially
        # TODO: Implement proper batching
        all_understandings = []
        for trajectory in trajectories:
            understandings = await self.generate_understandings_for_trajectory(
                trajectory, llm_client
            )
            all_understandings.append(understandings)
        return all_understandings
