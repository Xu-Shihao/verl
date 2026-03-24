# Copyright 2024 Shihao Xu
# SIG Reward Module - Main Calculator

import asyncio
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np

from .sig_config import SIGConfig
from .atomic_facts import AtomicFact, FactSet, AtomicFactExtractor
from .doctor_understanding import DoctorUnderstanding, DoctorUnderstandingGenerator
from .fact_checker import FactChecker
from .shapley_estimator import ShapleyValues, MonteCarloShapleyEstimator


@dataclass
class TokenLevelReward:
    """
    Token-level reward allocation (aligned with ProMed).

    Attributes:
        token_rewards: Reward for each token
        question_boundaries: List of (start_idx, end_idx) for question tokens
        answer_boundaries: List of (start_idx, end_idx) for answer tokens
        question_shapley_gains: Shapley gain for each question
        format_rewards: Format reward for each token
    """
    token_rewards: List[float]
    question_boundaries: List[Tuple[int, int]]
    answer_boundaries: List[Tuple[int, int]]
    question_shapley_gains: List[float]
    format_rewards: List[float]

    def get_mean_reward(self) -> float:
        """Get mean token reward."""
        if not self.token_rewards:
            return 0.0
        return sum(self.token_rewards) / len(self.token_rewards)

    def get_question_mean_reward(self) -> float:
        """Get mean reward for question tokens."""
        rewards = []
        for start, end in self.question_boundaries:
            rewards.extend(self.token_rewards[start:end])
        return sum(rewards) / len(rewards) if rewards else 0.0

    def get_answer_mean_reward(self) -> float:
        """Get mean reward for answer tokens."""
        rewards = []
        for start, end in self.answer_boundaries:
            rewards.extend(self.token_rewards[start:end])
        return sum(rewards) / len(rewards) if rewards else 0.0


@dataclass
class SIGResult:
    """
    Complete SIG computation result for a dialogue trajectory.

    Attributes:
        patient_id: Patient identifier
        total_sig_reward: Total SIG reward for the trajectory
        per_turn_sig: SIG(q_t) for each turn
        per_turn_reward: Final reward R(q_t) for each turn
        shapley_values: Computed Shapley values
        understandings: Doctor understandings for each turn
        coverage_matrix: Fact coverage matrix
        final_diagnosis_correct: Whether final diagnosis was correct
        component_scores: Breakdown of component scores
        token_level_reward: Token-level reward allocation (optional)
    """
    patient_id: str
    total_sig_reward: float
    per_turn_sig: List[float]
    per_turn_reward: List[float]
    shapley_values: Optional[ShapleyValues]
    understandings: List[DoctorUnderstanding]
    coverage_matrix: Optional[np.ndarray]
    final_diagnosis_correct: bool
    component_scores: Dict[str, Any] = field(default_factory=dict)
    token_level_reward: Optional[TokenLevelReward] = None

    def get_turn_rewards_for_distribution(self) -> List[Tuple[int, float]]:
        """
        Get rewards for token-level distribution.

        Returns list of (turn_index, reward) tuples.
        """
        return [(i, r) for i, r in enumerate(self.per_turn_reward)]


class AsyncLLMClient:
    """
    Async LLM client wrapper for SIG computations.

    This provides a unified interface for different LLM backends.
    """

    def __init__(
        self,
        base_url: str,
        timeout: float = 60.0,
        max_concurrent: int = 32,
    ):
        """
        Initialize the async LLM client.

        Args:
            base_url: Base URL for the LLM API
            timeout: Request timeout in seconds
            max_concurrent: Maximum concurrent requests
        """
        self.base_url = base_url
        self.timeout = timeout
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self._client = None

    async def _ensure_client(self):
        """Lazily initialize the HTTP client."""
        if self._client is None:
            try:
                import httpx
                self._client = httpx.AsyncClient(timeout=self.timeout)
            except ImportError:
                # Fallback to aiohttp if httpx not available
                import aiohttp
                self._client = aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout(total=self.timeout)
                )

    async def complete(
        self,
        prompt: str,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 256,
        enable_thinking: bool = False,  # 新增：是否启用思考模式（Qwen3等支持）
    ) -> str:
        """
        Complete a prompt using the LLM.

        Args:
            prompt: The prompt text
            model: Model name to use
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            enable_thinking: Whether to enable thinking mode (for Qwen3, etc.)

        Returns:
            Generated text response
        """
        await self._ensure_client()

        async with self.semaphore:
            try:
                import httpx
                request_body = {
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                }
                # 直接在请求体中添加 chat_template_kwargs 来控制 thinking 模式（vLLM + Qwen3 支持）
                # 注意：vLLM 直接接受此字段，无需通过 extra_body 包装
                if not enable_thinking:
                    request_body["chat_template_kwargs"] = {"enable_thinking": False}

                response = await self._client.post(
                    f"{self.base_url}/chat/completions",
                    json=request_body,
                    timeout=self.timeout
                )
                data = response.json()
                return data["choices"][0]["message"]["content"]

            except Exception as e:
                print(f"[SIG_LLM] Error in completion: {e}")
                raise

    async def compute_logprob(
        self,
        prompt: str,
        target: str,
        model: str,
    ) -> float:
        """
        Compute log probability of target given prompt.

        Args:
            prompt: The prompt text
            target: Target text to compute probability for
            model: Model name to use

        Returns:
            Log probability of target
        """
        await self._ensure_client()

        async with self.semaphore:
            try:
                import httpx
                # Use logprobs API (vLLM compatible)
                response = await self._client.post(
                    f"{self.base_url}/chat/completions",
                    json={
                        "model": model,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": len(target.split()) + 10,
                        "temperature": 1.0,  # Required for logprobs
                        "logprobs": True,
                        "top_logprobs": 5,
                    },
                    timeout=self.timeout
                )
                data = response.json()

                # Extract logprobs
                if "choices" in data and data["choices"]:
                    logprobs_data = data["choices"][0].get("logprobs", {})
                    if logprobs_data and "content" in logprobs_data:
                        total_logprob = sum(
                            token_info.get("logprob", -10)
                            for token_info in logprobs_data["content"]
                        )
                        return total_logprob

                # Fallback: use alternative scoring
                return await self._alternative_logprob(prompt, target, model)

            except Exception as e:
                print(f"[SIG_LLM] Error computing logprob: {e}")
                return -10.0  # Return low probability on error

    async def _alternative_logprob(
        self,
        prompt: str,
        target: str,
        model: str,
    ) -> float:
        """
        Alternative logprob estimation when API doesn't support logprobs.

        Uses perplexity-like scoring.
        """
        try:
            response = await self.complete(
                prompt=prompt,
                model=model,
                temperature=0.1,
                max_tokens=50
            )

            # Simple scoring based on match
            response_clean = response.strip().upper()
            target_clean = target.strip().upper()

            if target_clean in response_clean:
                return 0.0  # Perfect match
            elif response_clean[:3] == target_clean[:3]:
                return -1.0  # Partial match
            else:
                return -5.0  # No match

        except Exception:
            return -10.0

    async def close(self):
        """Close the HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None


class SIGCalculator:
    """
    Main orchestrator for SIG (Shapley Information Gain) reward computation.

    This class coordinates all components to compute SIG rewards:
    1. Extract atomic facts from patient info
    2. Compute Shapley values (with caching)
    3. Generate doctor understanding for each turn
    4. Check fact coverage
    5. Compute per-turn SIG
    6. Compute final rewards with correctness bonus
    """

    def __init__(self, config: SIGConfig):
        """
        Initialize the SIG calculator.

        Args:
            config: SIG configuration
        """
        self.config = config
        self.fact_extractor = AtomicFactExtractor(config)
        self.understanding_generator: Optional[DoctorUnderstandingGenerator] = None
        self.fact_checker: Optional[FactChecker] = None
        self.shapley_estimator: Optional[MonteCarloShapleyEstimator] = None
        self._llm_client: Optional[AsyncLLMClient] = None
        self._initialized = False

    async def initialize(self):
        """Initialize all components and LLM client."""
        if self._initialized:
            return

        # Create LLM client
        self._llm_client = AsyncLLMClient(
            base_url=self.config.llm_base_url,
            timeout=self.config.timeout,
            max_concurrent=self.config.max_concurrent_requests,
        )

        # Initialize components
        self.understanding_generator = DoctorUnderstandingGenerator(self.config)
        self.fact_checker = FactChecker(self.config)
        self.shapley_estimator = MonteCarloShapleyEstimator(self.config)

        self._initialized = True

        if self.config.log_sig_details:
            print(f"[SIG] Calculator initialized with LLM at {self.config.llm_base_url}")

    async def compute_sig_reward(
        self,
        patient_id: str,
        patient_info: str,
        dialogue_messages: List[Dict[str, str]],
        ground_truth_diagnosis: Union[str, List[str]],
        predicted_diagnosis: str,
        model_version: str = "",
    ) -> SIGResult:
        """
        Compute SIG reward for a complete dialogue trajectory.

        This is the main entry point for SIG computation.

        Args:
            patient_id: Unique patient identifier
            patient_info: Patient information text (for fact extraction)
            dialogue_messages: Complete dialogue history
            ground_truth_diagnosis: The correct ICD-10 diagnosis (single string or list of codes)
            predicted_diagnosis: The model's predicted diagnosis
            model_version: Model version hash (for cache invalidation)

        Returns:
            SIGResult containing all computed values
        """
        await self.initialize()

        if self.config.log_sig_details:
            print(f"[SIG] Computing reward for patient {patient_id}")
            print(f"[SIG] Dialogue has {len(dialogue_messages)} messages")

        # Step 1: Extract atomic facts
        fact_set = await self.fact_extractor.extract_facts(
            patient_id=patient_id,
            patient_info=patient_info,
            ground_truth_diagnosis=ground_truth_diagnosis,
            llm_client=self._llm_client,
        )

        if len(fact_set) == 0:
            print(f"[SIG] Warning: No facts extracted for patient {patient_id}")
            return self._create_empty_result(
                patient_id,
                self._check_diagnosis_correct(predicted_diagnosis, ground_truth_diagnosis)
            )

        # Step 2: Compute/retrieve Shapley values
        shapley_values = await self.shapley_estimator.estimate_shapley_values(
            fact_set=fact_set,
            llm_client=self._llm_client,
            model_version=model_version,
        )

        # Step 3: Generate doctor understandings for each turn
        understandings = await self.understanding_generator.generate_understandings_for_trajectory(
            full_dialogue=dialogue_messages,
            llm_client=self._llm_client,
        )

        # Step 4: Build coverage matrix
        coverage_matrix = await self.fact_checker.build_coverage_matrix(
            understandings=understandings,
            facts=fact_set,
            llm_client=self._llm_client,
        )

        # Step 5: Compute per-turn SIG
        per_turn_sig = self._compute_per_turn_sig(
            shapley_values=shapley_values,
            coverage_matrix=coverage_matrix,
        )

        # Step 6: Compute final rewards with correctness bonus
        final_diagnosis_correct = self._check_diagnosis_correct(
            predicted_diagnosis,
            ground_truth_diagnosis
        )

        per_turn_reward = self._compute_per_turn_rewards(
            per_turn_sig=per_turn_sig,
            diagnosis_correct=final_diagnosis_correct,
        )

        # Total SIG reward
        total_sig_reward = sum(per_turn_reward)

        # Create result
        result = SIGResult(
            patient_id=patient_id,
            total_sig_reward=total_sig_reward,
            per_turn_sig=per_turn_sig,
            per_turn_reward=per_turn_reward,
            shapley_values=shapley_values,
            understandings=understandings,
            coverage_matrix=coverage_matrix,
            final_diagnosis_correct=final_diagnosis_correct,
            component_scores={
                "num_facts": len(fact_set),
                "num_turns": len(per_turn_sig),
                "total_sig": sum(per_turn_sig),
                "shapley_converged": shapley_values.convergence_achieved,
                "shapley_iterations": shapley_values.num_iterations,
                "final_coverage": float(np.mean(coverage_matrix[-1])) if len(coverage_matrix) > 0 else 0.0,
            }
        )

        if self.config.log_sig_details:
            print(f"[SIG] Result: total_reward={total_sig_reward:.4f}, "
                  f"turns={len(per_turn_sig)}, diagnosis_correct={final_diagnosis_correct}")
            print(f"[SIG] Per-turn SIG: {[f'{s:.4f}' for s in per_turn_sig]}")

        return result

    def _compute_per_turn_sig(
        self,
        shapley_values: ShapleyValues,
        coverage_matrix: np.ndarray,
    ) -> List[float]:
        """
        Compute SIG for each turn.

        SIG(q_t) = Σ φ̃_i × [1(f_i ⊆ U_t) - 1(f_i ⊆ U_{t-1})]

        Args:
            shapley_values: Normalized Shapley values
            coverage_matrix: Coverage matrix C[t, i]

        Returns:
            List of SIG values, one per turn (excluding initial state)
        """
        per_turn_sig = []
        phi = shapley_values.normalized_values

        # 检查维度是否匹配
        num_shapley_facts = len(phi)
        num_coverage_facts = coverage_matrix.shape[1] if len(coverage_matrix.shape) > 1 else 0

        if num_shapley_facts != num_coverage_facts:
            print(f"[SIG] Warning: Dimension mismatch - shapley={num_shapley_facts}, coverage={num_coverage_facts}")
            # 使用较小的维度，截断或填充
            min_facts = min(num_shapley_facts, num_coverage_facts)
            if min_facts == 0:
                return [0.0] * (len(coverage_matrix) - 1)
            phi = phi[:min_facts]
            coverage_matrix = coverage_matrix[:, :min_facts]

        # Coverage matrix includes initial state at index 0
        for t in range(1, len(coverage_matrix)):
            coverage_prev = coverage_matrix[t - 1].astype(float)
            coverage_curr = coverage_matrix[t].astype(float)
            delta = coverage_curr - coverage_prev

            sig_t = float(np.dot(phi, delta))
            per_turn_sig.append(sig_t)

        return per_turn_sig

    def _compute_per_turn_rewards(
        self,
        per_turn_sig: List[float],
        diagnosis_correct: bool,
    ) -> List[float]:
        """
        Compute final reward for each turn.

        R(q_t) = β × SIG(q_t) + λ_q × (SIG(q_t)/Σ SIG) × 1(A' = A*)

        Args:
            per_turn_sig: SIG values for each turn
            diagnosis_correct: Whether final diagnosis was correct

        Returns:
            List of rewards, one per turn
        """
        beta = self.config.sig_reward_weight
        lambda_q = self.config.correctness_bonus_weight

        total_sig = sum(per_turn_sig) if per_turn_sig else 0.0
        correctness_indicator = 1.0 if diagnosis_correct else 0.0

        per_turn_reward = []
        for sig_t in per_turn_sig:
            # Base SIG reward
            r_t = beta * sig_t

            # Correctness bonus distributed by SIG contribution
            if total_sig > 0:
                r_t += lambda_q * (sig_t / total_sig) * correctness_indicator

            per_turn_reward.append(r_t)

        return per_turn_reward

    def _check_diagnosis_correct(
        self,
        predicted: str,
        ground_truth: Union[str, List[str]],
    ) -> bool:
        """
        Check if predicted diagnosis matches ground truth.

        Supports multiple ground truth ICD codes and two matching modes:
        - strict: Exact match (F32.1 must match F32.1)
        - soft: Major category match (F32.1 matches F32.x, only compare first 3 chars)

        Args:
            predicted: Predicted diagnosis text
            ground_truth: Ground truth diagnosis (single string or list of ICD codes)

        Returns:
            True if diagnosis is correct (any predicted code matches any ground truth)
        """
        import re

        # Look for ICD-10 pattern in predicted
        pred_codes = set(re.findall(r'[A-Z]\d{2}(?:\.\d+)?', predicted.upper()))

        # Handle both single string and list of ground truth codes
        gt_codes = set()
        if isinstance(ground_truth, str):
            gt_codes = set(re.findall(r'[A-Z]\d{2}(?:\.\d+)?', ground_truth.upper()))
        elif isinstance(ground_truth, (list, tuple)):
            for gt in ground_truth:
                gt_codes.update(re.findall(r'[A-Z]\d{2}(?:\.\d+)?', gt.upper()))

        if not gt_codes:
            # No valid ground truth codes
            return False

        if not pred_codes:
            # No predicted codes found
            return False

        # Check matching based on mode
        if self.config.diagnosis_match_mode == "soft":
            # Soft match: only compare major category (first 3 characters, e.g., F32)
            pred_major = {code[:3] for code in pred_codes}
            gt_major = {code[:3] for code in gt_codes}
            return bool(pred_major & gt_major)
        else:
            # Strict match: exact ICD code match
            return bool(pred_codes & gt_codes)

    def _create_empty_result(
        self,
        patient_id: str,
        diagnosis_correct: bool,
    ) -> SIGResult:
        """Create an empty result when computation fails."""
        return SIGResult(
            patient_id=patient_id,
            total_sig_reward=0.0,
            per_turn_sig=[],
            per_turn_reward=[],
            shapley_values=None,
            understandings=[],
            coverage_matrix=None,
            final_diagnosis_correct=diagnosis_correct,
            component_scores={"error": "empty_result"}
        )

    async def close(self):
        """Clean up resources."""
        if self._llm_client is not None:
            await self._llm_client.close()
            self._llm_client = None
        self._initialized = False

    def clear_caches(self):
        """Clear all caches."""
        self.fact_extractor.clear_cache()
        if self.shapley_estimator is not None:
            self.shapley_estimator.clear_cache()

    def compute_token_level_reward(
        self,
        completion_text: str,
        tokenizer: Any,
        per_turn_sig: List[float],
        diagnosis_correct: bool,
    ) -> TokenLevelReward:
        """
        Compute token-level reward allocation (aligned with ProMed).

        Token-level reward system:
        1. Question tokens: Shapley reward (α × gain) + result reward (β × gain × correct)
        2. Answer tokens: correctness reward (γ × correct)
        3. Format reward: 1.0 for correct format, 0.5 for partial, 0 otherwise

        Args:
            completion_text: Complete dialogue text
            tokenizer: Tokenizer for token boundary identification
            per_turn_sig: SIG values for each turn
            diagnosis_correct: Whether final diagnosis was correct

        Returns:
            TokenLevelReward containing per-token rewards
        """
        import re

        alpha = self.config.token_reward_alpha
        beta = self.config.token_reward_beta
        gamma = self.config.token_reward_gamma
        format_weight = self.config.format_reward_weight
        correct_indicator = 1.0 if diagnosis_correct else 0.0

        # Tokenize text
        tokens = tokenizer.encode(completion_text, add_special_tokens=False)
        full_text = tokenizer.decode(tokens, skip_special_tokens=False)
        num_tokens = len(tokens)

        # Initialize rewards
        token_rewards = [0.0] * num_tokens
        format_rewards = [0.0] * num_tokens

        # Find question and answer boundaries
        question_boundaries = []
        answer_boundaries = []

        # Parse assistant turns using pattern matching
        assistant_pattern = r'<\|im_start\|>assistant(.*?)(?=<\|im_start\||$)'
        assistant_matches = list(re.finditer(assistant_pattern, full_text, re.DOTALL))

        question_gains = []

        for i, match in enumerate(assistant_matches):
            content = match.group(1).strip()
            start_char = match.start() + len('<|im_start|>assistant')
            end_char = match.end()

            # Convert to token positions
            prefix_tokens = tokenizer.encode(full_text[:start_char], add_special_tokens=False)
            segment_tokens = tokenizer.encode(full_text[start_char:end_char], add_special_tokens=False)
            start_token = len(prefix_tokens)
            end_token = min(start_token + len(segment_tokens), num_tokens)

            # Determine if this is a question or answer turn
            is_question = 'question:' in content.lower()
            is_answer = 'answer:' in content.lower()

            # Calculate format reward
            format_score = 0.0
            if content.lower().startswith('question:') or content.lower().startswith('answer:'):
                format_score = 1.0
            elif is_question or is_answer:
                format_score = 0.5

            # Assign format rewards
            for idx in range(start_token, end_token):
                format_rewards[idx] = format_score * format_weight
                token_rewards[idx] += format_rewards[idx]

            if is_question and not is_answer:
                question_boundaries.append((start_token, end_token))
                # Get Shapley gain for this question
                question_idx = len(question_boundaries) - 1
                shapley_gain = per_turn_sig[question_idx] if question_idx < len(per_turn_sig) else 0.0
                question_gains.append(shapley_gain)

                # Question content reward: α × gain + β × gain × correct
                question_reward = alpha * shapley_gain + beta * shapley_gain * correct_indicator
                for idx in range(start_token, end_token):
                    token_rewards[idx] += question_reward

            elif is_answer:
                answer_boundaries.append((start_token, end_token))
                # Answer content reward: γ × correct
                answer_reward = gamma * correct_indicator
                for idx in range(start_token, end_token):
                    token_rewards[idx] += answer_reward

        if self.config.log_sig_details:
            print(f"[SIG_TOKEN] Token-level rewards computed: "
                  f"{len(question_boundaries)} questions, {len(answer_boundaries)} answers")
            print(f"[SIG_TOKEN] Mean token reward: {sum(token_rewards)/len(token_rewards):.4f}")

        return TokenLevelReward(
            token_rewards=token_rewards,
            question_boundaries=question_boundaries,
            answer_boundaries=answer_boundaries,
            question_shapley_gains=question_gains,
            format_rewards=format_rewards,
        )

    async def compute_sig_reward_with_tokens(
        self,
        patient_id: str,
        patient_info: str,
        dialogue_messages: List[Dict[str, str]],
        ground_truth_diagnosis: Union[str, List[str]],
        predicted_diagnosis: str,
        completion_text: str,
        tokenizer: Any,
        model_version: str = "",
    ) -> SIGResult:
        """
        Compute SIG reward with token-level allocation (aligned with ProMed).

        This extends compute_sig_reward to include token-level rewards.

        Args:
            patient_id: Unique patient identifier
            patient_info: Patient information text
            dialogue_messages: Complete dialogue history
            ground_truth_diagnosis: The correct ICD-10 diagnosis (single string or list of codes)
            predicted_diagnosis: The model's predicted diagnosis
            completion_text: Raw completion text for tokenization
            tokenizer: Tokenizer for token boundary identification
            model_version: Model version hash

        Returns:
            SIGResult with token_level_reward populated
        """
        # First compute standard SIG reward
        result = await self.compute_sig_reward(
            patient_id=patient_id,
            patient_info=patient_info,
            dialogue_messages=dialogue_messages,
            ground_truth_diagnosis=ground_truth_diagnosis,
            predicted_diagnosis=predicted_diagnosis,
            model_version=model_version,
        )

        # If token-level reward is enabled, compute it
        if self.config.use_token_level_reward:
            token_reward = self.compute_token_level_reward(
                completion_text=completion_text,
                tokenizer=tokenizer,
                per_turn_sig=result.per_turn_sig,
                diagnosis_correct=result.final_diagnosis_correct,
            )
            result.token_level_reward = token_reward

            if self.config.log_sig_details:
                print(f"[SIG_TOKEN] Token-level reward computed: "
                      f"mean={token_reward.get_mean_reward():.4f}")

        return result


# Convenience function for sync usage
def compute_sig_reward_sync(
    patient_id: str,
    patient_info: str,
    dialogue_messages: List[Dict[str, str]],
    ground_truth_diagnosis: Union[str, List[str]],
    predicted_diagnosis: str,
    config: Optional[SIGConfig] = None,
    model_version: str = "",
) -> SIGResult:
    """
    Synchronous wrapper for SIG reward computation.

    This is a convenience function for use in synchronous contexts.

    Args:
        patient_id: Unique patient identifier
        patient_info: Patient information text
        dialogue_messages: Complete dialogue history
        ground_truth_diagnosis: The correct ICD-10 diagnosis (single string or list of codes)
        predicted_diagnosis: The model's predicted diagnosis
        config: SIG configuration (optional)
        model_version: Model version hash

    Returns:
        SIGResult containing all computed values
    """
    if config is None:
        config = SIGConfig()

    calculator = SIGCalculator(config)

    async def _compute():
        try:
            return await calculator.compute_sig_reward(
                patient_id=patient_id,
                patient_info=patient_info,
                dialogue_messages=dialogue_messages,
                ground_truth_diagnosis=ground_truth_diagnosis,
                predicted_diagnosis=predicted_diagnosis,
                model_version=model_version,
            )
        finally:
            await calculator.close()

    return asyncio.run(_compute())
