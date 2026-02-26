# Copyright 2024 Shihao Xu
# SIG Reward Module - Fact Entailment Checking

import asyncio
import re
from typing import List, Dict, Any, Tuple, Optional
import numpy as np

from .sig_config import SIGConfig
from .atomic_facts import AtomicFact, FactSet
from .doctor_understanding import DoctorUnderstanding


# Prompt template for fact checking
FACT_CHECK_PROMPT_TEMPLATE = """你是一个医学事实验证系统。

请判断以下医学事实是否被医生的理解所包含（蕴含）。

医生当前的理解：
{understanding}

需要验证的事实：
{fact}

判断标准：
- 如果医生的理解中明确包含或可以推断出该事实，回答"是"
- 如果医生的理解中没有提及或与该事实矛盾，回答"否"
- 只有在有充分证据表明医生已经了解该信息时才回答"是"

请只回答"是"或"否"，不要解释。"""


# Batch fact checking prompt
BATCH_FACT_CHECK_PROMPT_TEMPLATE = """你是一个医学事实验证系统。

请判断以下每个医学事实是否被医生的理解所包含（蕴含）。

医生当前的理解：
{understanding}

需要验证的事实列表：
{facts_list}

判断标准：
- 如果医生的理解中明确包含或可以推断出该事实，标记为"是"
- 如果医生的理解中没有提及或与该事实矛盾，标记为"否"

请以JSON格式输出结果，格式如下：
```json
{{
  "results": [
    {{"id": "f1", "covered": true}},
    {{"id": "f2", "covered": false}},
    ...
  ]
}}
```

只输出JSON，不要输出其他内容。"""


class FactChecker:
    """
    LLM-based entailment checking for atomic facts.

    This class checks whether each atomic fact is covered (entailed)
    by the doctor's current understanding.

    Aligned with ProMed: supports string matching fallback when LLM fails.
    """

    def __init__(self, config: SIGConfig):
        """
        Initialize the fact checker.

        Args:
            config: SIG configuration containing LLM settings
        """
        self.config = config

    def _string_match_check(self, text: str, fact_content: str) -> bool:
        """
        Check if a fact is covered using string matching (aligned with ProMed).

        Uses keyword matching with configurable threshold.

        Args:
            text: Text to check against (doctor's understanding)
            fact_content: The fact content to check

        Returns:
            True if keywords match above threshold
        """
        text_lower = text.lower()
        fact_lower = fact_content.lower()

        # Direct substring match
        if fact_lower in text_lower:
            return True

        # Keyword matching (aligned with ProMed: only consider words > 2 chars)
        fact_keywords = [word for word in fact_lower.split() if len(word) > 2]

        if len(fact_keywords) == 0:
            return False

        match_count = sum(1 for keyword in fact_keywords if keyword in text_lower)
        match_ratio = match_count / len(fact_keywords)

        if self.config.log_sig_details:
            print(f"[SIG_FACTCHECK] String match: keywords={fact_keywords}, "
                  f"matched={match_count}/{len(fact_keywords)}, ratio={match_ratio:.2f}")

        return match_ratio >= self.config.string_match_threshold

    async def check_single_fact(
        self,
        understanding: str,
        fact: AtomicFact,
        llm_client: Any,
    ) -> bool:
        """
        Check if a single fact is covered by the understanding.

        Aligned with ProMed: uses string matching fallback when LLM fails.

        Args:
            understanding: Doctor's understanding text
            fact: The atomic fact to check
            llm_client: Async LLM client

        Returns:
            True if the fact is covered, False otherwise
        """
        prompt = FACT_CHECK_PROMPT_TEMPLATE.format(
            understanding=understanding,
            fact=f"[{fact.category}] {fact.content}"
        )

        try:
            response = await llm_client.complete(
                prompt=prompt,
                model=self.config.fact_checker_model,
                temperature=0.1,  # Very low temperature for consistency
                max_tokens=10,
            )

            # Parse response
            response_lower = response.strip().lower()
            if '是' in response_lower or 'yes' in response_lower:
                return True
            elif '否' in response_lower or 'no' in response_lower:
                return False
            else:
                # Default to False if unclear
                return False

        except Exception as e:
            print(f"[SIG_FACTCHECK] Error checking fact {fact.id}: {e}")

            # Fallback to string matching (aligned with ProMed)
            if self.config.use_string_matching_fallback:
                print(f"[SIG_FACTCHECK] Using string matching fallback for {fact.id}")
                return self._string_match_check(understanding, fact.content)

            return False

    async def check_facts_batch(
        self,
        understanding: str,
        facts: List[AtomicFact],
        llm_client: Any,
    ) -> Dict[str, bool]:
        """
        Batch check multiple facts against one understanding.

        This is more efficient than checking facts one by one.

        Args:
            understanding: Doctor's understanding text
            facts: List of atomic facts to check
            llm_client: Async LLM client

        Returns:
            Dictionary mapping fact_id -> is_covered
        """
        if not facts:
            return {}

        # Format facts list
        facts_list_str = "\n".join([
            f"- [{f.id}] [{f.category}] {f.content}"
            for f in facts
        ])

        prompt = BATCH_FACT_CHECK_PROMPT_TEMPLATE.format(
            understanding=understanding,
            facts_list=facts_list_str
        )

        try:
            response = await llm_client.complete(
                prompt=prompt,
                model=self.config.fact_checker_model,
                temperature=0.1,
                max_tokens=1024,
            )

            # Parse JSON response
            results = self._parse_batch_response(response, facts)

        except Exception as e:
            print(f"[SIG_FACTCHECK] Error in batch check: {e}")

            # Fallback strategy (aligned with ProMed)
            results = {}
            if self.config.use_string_matching_fallback:
                # Use string matching for all facts
                print(f"[SIG_FACTCHECK] Using string matching fallback for batch")
                for fact in facts:
                    results[fact.id] = self._string_match_check(
                        understanding, fact.content
                    )
            else:
                # Try individual LLM checks
                for fact in facts:
                    results[fact.id] = await self.check_single_fact(
                        understanding, fact, llm_client
                    )

        return results

    def _parse_batch_response(
        self,
        response: str,
        facts: List[AtomicFact]
    ) -> Dict[str, bool]:
        """
        Parse batch fact checking response.

        Args:
            response: LLM response text
            facts: List of facts that were checked

        Returns:
            Dictionary mapping fact_id -> is_covered
        """
        import json

        results = {f.id: False for f in facts}  # Default to False

        try:
            # Try to extract JSON
            json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_match = re.search(r'\{.*"results".*\}', response, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                else:
                    json_str = response

            data = json.loads(json_str)

            if isinstance(data, dict) and 'results' in data:
                for item in data['results']:
                    if isinstance(item, dict):
                        fact_id = str(item.get('id', ''))
                        covered = bool(item.get('covered', False))
                        if fact_id in results:
                            results[fact_id] = covered

        except json.JSONDecodeError:
            # Fallback: parse line by line
            for fact in facts:
                # Look for patterns like "f1: 是" or "f1: true"
                pattern = rf'{fact.id}[:\s]*(是|yes|true|否|no|false)'
                match = re.search(pattern, response, re.IGNORECASE)
                if match:
                    result_str = match.group(1).lower()
                    results[fact.id] = result_str in ['是', 'yes', 'true']

        return results

    async def build_coverage_matrix(
        self,
        understandings: List[DoctorUnderstanding],
        facts: FactSet,
        llm_client: Any,
    ) -> np.ndarray:
        """
        Build coverage matrix C[t, i] = 1(f_i ⊆ U_t).

        This matrix has shape (num_turns, num_facts) where each entry
        indicates whether fact i is covered by understanding t.

        Args:
            understandings: List of doctor understandings (including initial)
            facts: FactSet containing all atomic facts
            llm_client: Async LLM client

        Returns:
            Coverage matrix of shape (len(understandings), len(facts))
        """
        num_turns = len(understandings)
        num_facts = len(facts)

        coverage_matrix = np.zeros((num_turns, num_facts), dtype=bool)

        # First row (initial understanding) is all False
        # since doctor knows nothing at the start

        # Check facts for each understanding
        for t, understanding in enumerate(understandings):
            if t == 0 and understanding.turn_index == -1:
                # Initial state - no facts covered
                continue

            # Batch check all facts
            coverage_dict = await self.check_facts_batch(
                understanding.understanding_text,
                facts.facts,
                llm_client
            )

            # Fill matrix row
            for i, fact in enumerate(facts.facts):
                coverage_matrix[t, i] = coverage_dict.get(fact.id, False)

            # Store in understanding object
            understanding.fact_coverage = coverage_dict
            understanding.coverage_vector = coverage_matrix[t].copy()

            if self.config.log_sig_details:
                covered_count = np.sum(coverage_matrix[t])
                print(f"[SIG_FACTCHECK] Turn {understanding.turn_index}: "
                      f"{covered_count}/{num_facts} facts covered")

        return coverage_matrix

    async def check_coverage_delta(
        self,
        understanding_prev: DoctorUnderstanding,
        understanding_curr: DoctorUnderstanding,
        facts: FactSet,
        llm_client: Any,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Check coverage change between two consecutive understandings.

        Returns the coverage vectors and delta vector.

        Args:
            understanding_prev: Previous understanding (U_{t-1})
            understanding_curr: Current understanding (U_t)
            facts: FactSet containing all atomic facts
            llm_client: Async LLM client

        Returns:
            Tuple of (coverage_prev, coverage_curr, delta)
            where delta[i] = coverage_curr[i] - coverage_prev[i]
        """
        # Check if coverage vectors are already computed
        if understanding_prev.coverage_vector is None:
            if understanding_prev.turn_index == -1:
                coverage_prev = np.zeros(len(facts), dtype=bool)
            else:
                prev_dict = await self.check_facts_batch(
                    understanding_prev.understanding_text,
                    facts.facts,
                    llm_client
                )
                coverage_prev = np.array([
                    prev_dict.get(f.id, False) for f in facts.facts
                ], dtype=bool)
            understanding_prev.coverage_vector = coverage_prev
        else:
            coverage_prev = understanding_prev.coverage_vector

        if understanding_curr.coverage_vector is None:
            curr_dict = await self.check_facts_batch(
                understanding_curr.understanding_text,
                facts.facts,
                llm_client
            )
            coverage_curr = np.array([
                curr_dict.get(f.id, False) for f in facts.facts
            ], dtype=bool)
            understanding_curr.coverage_vector = coverage_curr
        else:
            coverage_curr = understanding_curr.coverage_vector

        # Compute delta (new facts discovered this turn)
        delta = coverage_curr.astype(float) - coverage_prev.astype(float)

        return coverage_prev, coverage_curr, delta
