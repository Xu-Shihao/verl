# Copyright 2024 Shihao Xu
# SIG Reward Module - Atomic Facts Extraction

import json
import re
import hashlib
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple, Union
import numpy as np

try:
    from pydantic import BaseModel, Field, ValidationError
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False

from .sig_config import SIGConfig


# Pydantic models for structured JSON parsing
if PYDANTIC_AVAILABLE:
    class FactItem(BaseModel):
        """Single fact item in the extraction result."""
        id: str = Field(default="f1")
        category: str = Field(default="other")
        content: str

    class FactExtractionResult(BaseModel):
        """Result of fact extraction from LLM."""
        facts: List[FactItem] = Field(default_factory=list)


@dataclass
class AtomicFact:
    """
    Represents a single atomic fact about the patient.

    An atomic fact is an independent, verifiable statement about the patient's
    condition that can be checked against the doctor's understanding.

    Attributes:
        id: Unique identifier for the fact
        content: Natural language description of the fact
        category: Category of the fact (symptom, history, demographic, etc.)
        source: Source of the fact (e.g., "patient_info", "llm_extraction")
    """
    id: str
    content: str
    category: str
    source: str = "llm_extraction"

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        if isinstance(other, AtomicFact):
            return self.id == other.id
        return False


@dataclass
class FactSet:
    """
    Collection of atomic facts for a patient.

    Attributes:
        patient_id: Unique identifier for the patient
        facts: List of atomic facts
        ground_truth_diagnosis: The correct ICD-10 diagnosis code(s), supports multiple codes
        patient_info_hash: Hash of the patient info text (for cache validation)
    """
    patient_id: str
    facts: List[AtomicFact]
    ground_truth_diagnosis: Union[str, List[str]]
    patient_info_hash: str = ""

    def __len__(self):
        return len(self.facts)

    def get_fact_by_id(self, fact_id: str) -> Optional[AtomicFact]:
        """Get a fact by its ID."""
        for fact in self.facts:
            if fact.id == fact_id:
                return fact
        return None

    def get_fact_ids(self) -> List[str]:
        """Get list of all fact IDs."""
        return [f.id for f in self.facts]

    def get_fact_contents(self) -> List[str]:
        """Get list of all fact contents."""
        return [f.content for f in self.facts]

    def to_subset(self, indices: List[int]) -> 'FactSet':
        """
        Create a subset of facts for Shapley computation.

        Args:
            indices: List of indices to include in the subset

        Returns:
            New FactSet containing only the specified facts
        """
        subset_facts = [self.facts[i] for i in indices if i < len(self.facts)]
        return FactSet(
            patient_id=self.patient_id,
            facts=subset_facts,
            ground_truth_diagnosis=self.ground_truth_diagnosis,
            patient_info_hash=self.patient_info_hash
        )

    def to_state_string(self) -> str:
        """
        Convert facts to a state string for LLM input.

        Returns:
            Formatted string containing all facts
        """
        if not self.facts:
            return "无已知信息"

        fact_strings = []
        for i, fact in enumerate(self.facts, 1):
            fact_strings.append(f"{i}. [{fact.category}] {fact.content}")
        return "\n".join(fact_strings)


# Prompt template for LLM-based fact extraction
# 注意：明确要求不要输出思考过程，直接输出JSON以加速提取
FACT_EXTRACTION_PROMPT = """分析以下患者病例信息，提取所有临床相关的原子事实。

病例信息：
{patient_info}

请提取所有独立的、可验证的临床事实，包括但不限于：
- 症状（当前症状、症状持续时间、症状严重程度）
- 病史（既往病史、家族史）
- 人口学信息（年龄、性别、职业）
- 生活习惯（睡眠、饮食、运动）
- 心理状态（情绪、认知、行为）
- 社会功能（工作、人际关系）

每个事实应该是独立的、可以单独验证的陈述。

重要：直接输出JSON，不要输出任何思考过程或解释。

JSON格式如下：
{{"facts": [{{"id": "f1", "category": "symptom", "content": "患者存在睡眠障碍"}}, {{"id": "f2", "category": "history", "content": "病程持续2周"}}]}}

category可选值：symptom, history, demographic, lifestyle, mood, cognition, behavior, social, medication, other"""


class AtomicFactExtractor:
    """
    Extract atomic facts from patient information using LLM.

    This class handles the extraction of atomic facts from unstructured
    patient information text using an LLM.
    """

    def __init__(self, config: SIGConfig):
        """
        Initialize the fact extractor.

        Args:
            config: SIG configuration containing LLM settings
        """
        self.config = config
        self._cache: Dict[str, FactSet] = {}  # patient_info_hash -> FactSet

    def _compute_hash(self, text: str) -> str:
        """Compute hash of text for caching."""
        return hashlib.md5(text.encode('utf-8')).hexdigest()

    def _parse_llm_response(self, response_text: str, patient_id: str) -> List[AtomicFact]:
        """
        Parse LLM response to extract atomic facts using Pydantic for validation.

        Args:
            response_text: Raw LLM response text
            patient_id: Patient identifier for error logging

        Returns:
            List of extracted AtomicFact objects
        """
        facts = []

        try:
            # Try to extract JSON from the response
            # Handle cases where the response might have markdown code blocks
            json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # Try to find raw JSON - use non-greedy match to avoid extra data issues
                json_match = re.search(r'\{"facts"\s*:\s*\[.*?\]\s*\}', response_text, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                else:
                    # Fallback: find first { to last }
                    first_brace = response_text.find('{')
                    last_brace = response_text.rfind('}')
                    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
                        json_str = response_text[first_brace:last_brace+1]
                    else:
                        json_str = response_text

            # Use Pydantic for robust parsing if available
            if PYDANTIC_AVAILABLE:
                try:
                    result = FactExtractionResult.model_validate_json(json_str)
                    for item in result.facts:
                        if item.content.strip():
                            facts.append(AtomicFact(
                                id=item.id,
                                content=item.content.strip(),
                                category=item.category.lower(),
                                source='llm_extraction'
                            ))
                except ValidationError as ve:
                    # Pydantic validation failed, try manual parsing
                    data = json.loads(json_str)
                    if isinstance(data, dict) and 'facts' in data:
                        for item in data['facts']:
                            if isinstance(item, dict):
                                content = str(item.get('content', '')).strip()
                                if content:
                                    facts.append(AtomicFact(
                                        id=str(item.get('id', f'f{len(facts)+1}')),
                                        content=content,
                                        category=str(item.get('category', 'other')).lower(),
                                        source='llm_extraction'
                                    ))
            else:
                # Fallback to manual JSON parsing
                data = json.loads(json_str)
                if isinstance(data, dict) and 'facts' in data:
                    for item in data['facts']:
                        if isinstance(item, dict):
                            fact_id = str(item.get('id', f'f{len(facts)+1}'))
                            content = str(item.get('content', '')).strip()
                            category = str(item.get('category', 'other')).lower()

                            if content:  # Only add facts with content
                                facts.append(AtomicFact(
                                    id=fact_id,
                                    content=content,
                                    category=category,
                                    source='llm_extraction'
                                ))

        except json.JSONDecodeError as e:
            print(f"[SIG_FACTS] JSON parse error for patient {patient_id}: {e}")
            # Fallback: try to extract facts using regex
            facts = self._fallback_extraction(response_text)
        except Exception as e:
            print(f"[SIG_FACTS] Unexpected error parsing response for patient {patient_id}: {e}")
            facts = self._fallback_extraction(response_text)

        # Limit number of facts
        if len(facts) > self.config.max_facts_per_patient:
            facts = facts[:self.config.max_facts_per_patient]

        # Ensure unique IDs
        seen_ids = set()
        for i, fact in enumerate(facts):
            if fact.id in seen_ids:
                fact.id = f"f{i+1}"
            seen_ids.add(fact.id)

        return facts

    def _fallback_extraction(self, text: str) -> List[AtomicFact]:
        """
        Fallback extraction when JSON parsing fails.

        Attempts to extract facts from plain text.
        """
        facts = []
        # Simple pattern matching for numbered lists
        patterns = [
            r'\d+\.\s*\[(\w+)\]\s*(.+?)(?=\d+\.|$)',
            r'"content":\s*"([^"]+)"',
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text, re.MULTILINE)
            for i, match in enumerate(matches):
                if isinstance(match, tuple):
                    category, content = match[0], match[1] if len(match) > 1 else match[0]
                else:
                    category, content = 'other', match

                facts.append(AtomicFact(
                    id=f"f{len(facts)+1}",
                    content=content.strip(),
                    category=category.lower() if category else 'other',
                    source='fallback_extraction'
                ))

        return facts

    async def extract_facts(
        self,
        patient_id: str,
        patient_info: str,
        ground_truth_diagnosis: Union[str, List[str]],
        llm_client: Any,
    ) -> FactSet:
        """
        Extract atomic facts from patient information using LLM.

        Args:
            patient_id: Unique patient identifier
            patient_info: Unstructured patient information text
            ground_truth_diagnosis: The correct ICD-10 diagnosis (single string or list of codes)
            llm_client: Async LLM client for making API calls

        Returns:
            FactSet containing extracted atomic facts
        """
        # Check cache
        info_hash = self._compute_hash(patient_info)
        cache_key = f"{patient_id}_{info_hash}"

        if self.config.enable_caching and cache_key in self._cache:
            cached = self._cache[cache_key]
            # Update diagnosis if needed
            if cached.ground_truth_diagnosis != ground_truth_diagnosis:
                cached.ground_truth_diagnosis = ground_truth_diagnosis
            return cached

        # Prepare prompt
        prompt = FACT_EXTRACTION_PROMPT.format(patient_info=patient_info)

        try:
            # Call LLM with thinking disabled for faster extraction
            response = await llm_client.complete(
                prompt=prompt,
                model=self.config.understanding_model,  # Reuse understanding model
                temperature=self.config.fact_extraction_temperature,
                max_tokens=2048,
                enable_thinking=False,  # 禁用思考模式加速提取
            )

            # Parse response
            facts = self._parse_llm_response(response, patient_id)

            if not facts:
                print(f"[SIG_FACTS] Warning: No facts extracted for patient {patient_id}")
                # Create a minimal fact set
                facts = [AtomicFact(
                    id="f1",
                    content="患者信息不足",
                    category="other",
                    source="default"
                )]

        except Exception as e:
            print(f"[SIG_FACTS] Error extracting facts for patient {patient_id}: {e}")
            facts = [AtomicFact(
                id="f1",
                content="事实提取失败",
                category="error",
                source="error"
            )]

        # Create FactSet
        fact_set = FactSet(
            patient_id=patient_id,
            facts=facts,
            ground_truth_diagnosis=ground_truth_diagnosis,
            patient_info_hash=info_hash
        )

        # Cache result
        if self.config.enable_caching:
            self._cache[cache_key] = fact_set

        if self.config.log_sig_details:
            print(f"[SIG_FACTS] Extracted {len(facts)} facts for patient {patient_id}")
            for fact in facts[:5]:  # Log first 5 facts
                print(f"  - [{fact.category}] {fact.content[:50]}...")

        return fact_set

    def clear_cache(self):
        """Clear the fact extraction cache."""
        self._cache.clear()

    def get_cache_size(self) -> int:
        """Get the number of cached fact sets."""
        return len(self._cache)
