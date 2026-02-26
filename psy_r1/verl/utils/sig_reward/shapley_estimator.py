# Copyright 2024 Shihao Xu
# SIG Reward Module - Monte Carlo Shapley Value Estimation

import asyncio
import os
import pickle
import hashlib
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np

from .sig_config import SIGConfig
from .atomic_facts import AtomicFact, FactSet


@dataclass
class ShapleyValues:
    """
    Shapley values for all facts of a patient.

    The Shapley value φ(f_i) measures the contribution of fact f_i
    to the model's ability to predict the correct diagnosis.

    Attributes:
        patient_id: Patient identifier
        fact_ids: List of fact IDs in order
        raw_values: Raw Shapley values φ(f_i) for each fact
        normalized_values: Softmax normalized values φ̃_i
        convergence_achieved: Whether MC estimation converged
        num_iterations: Number of MC iterations performed
        model_version: Hash of model weights (for cache invalidation)
    """
    patient_id: str
    fact_ids: List[str]
    raw_values: np.ndarray
    normalized_values: np.ndarray
    convergence_achieved: bool
    num_iterations: int
    model_version: str = ""

    def get_value(self, fact_id: str) -> float:
        """Get normalized Shapley value for a fact."""
        try:
            idx = self.fact_ids.index(fact_id)
            return float(self.normalized_values[idx])
        except ValueError:
            return 0.0

    def get_raw_value(self, fact_id: str) -> float:
        """Get raw Shapley value for a fact."""
        try:
            idx = self.fact_ids.index(fact_id)
            return float(self.raw_values[idx])
        except ValueError:
            return 0.0

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary mapping fact_id -> normalized value."""
        return {
            fid: float(self.normalized_values[i])
            for i, fid in enumerate(self.fact_ids)
        }


# Prompt template for computing v(S) = log P(A* | Q, S)
DIAGNOSIS_PROMPT_TEMPLATE = """你是一位精神科医生。请根据以下患者信息做出诊断。

已知患者信息：
{state_info}

请根据以上信息，给出最可能的ICD-10精神障碍诊断代码。

只输出诊断代码（如F32.1），不要输出其他内容。"""


class ShapleyValueCache:
    """
    Cache for Shapley values to avoid recomputation.

    Note: When using training model for dynamic rewards, the cache
    should be invalidated when model weights change significantly.
    """

    def __init__(self, cache_dir: str = "~/.cache/sig_reward"):
        """
        Initialize the cache.

        Args:
            cache_dir: Directory for persistent cache storage
        """
        self.cache_dir = os.path.expanduser(cache_dir)
        self._memory_cache: Dict[str, ShapleyValues] = {}

    def _get_cache_key(self, patient_id: str, model_version: str) -> str:
        """Generate cache key from patient_id and model version."""
        return f"{patient_id}_{model_version}"

    def get(
        self,
        patient_id: str,
        model_version: str = ""
    ) -> Optional[ShapleyValues]:
        """
        Get cached Shapley values for a patient.

        Args:
            patient_id: Patient identifier
            model_version: Model version hash (for cache invalidation)

        Returns:
            Cached ShapleyValues or None if not found
        """
        cache_key = self._get_cache_key(patient_id, model_version)

        # Check memory cache first
        if cache_key in self._memory_cache:
            return self._memory_cache[cache_key]

        # Try disk cache
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    values = pickle.load(f)
                    self._memory_cache[cache_key] = values
                    return values
            except Exception as e:
                print(f"[SIG_CACHE] Error loading cache: {e}")
                return None

        return None

    def set(
        self,
        patient_id: str,
        values: ShapleyValues,
        model_version: str = ""
    ):
        """
        Cache Shapley values.

        Args:
            patient_id: Patient identifier
            values: ShapleyValues to cache
            model_version: Model version hash
        """
        cache_key = self._get_cache_key(patient_id, model_version)

        # Store in memory
        self._memory_cache[cache_key] = values

        # Store on disk
        try:
            os.makedirs(self.cache_dir, exist_ok=True)
            cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")
            with open(cache_file, 'wb') as f:
                pickle.dump(values, f)
        except Exception as e:
            print(f"[SIG_CACHE] Error saving cache: {e}")

    def clear(self):
        """Clear all cached values."""
        self._memory_cache.clear()

    def clear_for_patient(self, patient_id: str):
        """Clear cache for a specific patient."""
        keys_to_remove = [
            k for k in self._memory_cache.keys()
            if k.startswith(f"{patient_id}_")
        ]
        for key in keys_to_remove:
            del self._memory_cache[key]


class MonteCarloShapleyEstimator:
    """
    Monte Carlo estimation of Shapley values for atomic facts.

    This implements Algorithm 1 from the SIG paper:
    1. For k = 1 to K:
       a. Generate random permutation π of facts
       b. For each fact f_i:
          - S = facts before f_i in permutation
          - Marginal contribution = v(S ∪ {f_i}) - v(S)
          - Update running average for φ(f_i)
       c. Check convergence (optional early stopping)
    2. Normalize via softmax: φ̃_i = exp(φ_i) / Σexp(φ_j)
    """

    def __init__(self, config: SIGConfig):
        """
        Initialize the Shapley estimator.

        Args:
            config: SIG configuration
        """
        self.config = config
        self._cache = ShapleyValueCache(config.cache_dir) if config.cache_shapley_values else None

    async def estimate_shapley_values(
        self,
        fact_set: FactSet,
        llm_client: Any,
        model_version: str = "",
    ) -> ShapleyValues:
        """
        Estimate Shapley values using Monte Carlo sampling.

        Args:
            fact_set: Set of atomic facts for a patient
            llm_client: Async LLM client for computing v(S)
            model_version: Model version hash for cache invalidation

        Returns:
            ShapleyValues containing estimated values
        """
        patient_id = fact_set.patient_id
        n = len(fact_set.facts)

        if n == 0:
            return ShapleyValues(
                patient_id=patient_id,
                fact_ids=[],
                raw_values=np.array([]),
                normalized_values=np.array([]),
                convergence_achieved=True,
                num_iterations=0,
                model_version=model_version
            )

        # Check cache
        if self._cache is not None:
            cached = self._cache.get(patient_id, model_version)
            if cached is not None:
                if self.config.log_sig_details:
                    print(f"[SIG_SHAPLEY] Using cached values for patient {patient_id}")
                return cached

        # Initialize Shapley values
        phi = np.zeros(n, dtype=np.float64)
        phi_history = []  # For convergence checking

        K = self.config.monte_carlo_k
        ground_truth = fact_set.ground_truth_diagnosis

        if self.config.log_sig_details:
            print(f"[SIG_SHAPLEY] Estimating Shapley values for {n} facts, K={K}")

        # Monte Carlo sampling
        converged = False
        for k in range(1, K + 1):
            # Generate random permutation
            perm = np.random.permutation(n)

            # Track marginal contributions for this iteration
            marginal_contributions = np.zeros(n, dtype=np.float64)

            # Start with empty set
            S_indices = []
            prev_v = await self._compute_value_function(
                fact_set.to_subset(S_indices),
                ground_truth,
                llm_client
            )

            # Add facts one by one according to permutation
            for i in perm:
                S_indices.append(int(i))
                curr_v = await self._compute_value_function(
                    fact_set.to_subset(S_indices),
                    ground_truth,
                    llm_client
                )

                # Marginal contribution
                delta = curr_v - prev_v
                marginal_contributions[i] = delta

                prev_v = curr_v

            # Online update of Shapley values
            # φ_i = (k-1)/k * φ_i + 1/k * Δ_i
            phi = ((k - 1) / k) * phi + (1 / k) * marginal_contributions

            # Store history for convergence check
            phi_history.append(phi.copy())

            # Check convergence after minimum iterations
            if k >= self.config.min_iterations:
                converged = self._check_convergence(
                    phi_history,
                    self.config.convergence_epsilon
                )
                if converged:
                    if self.config.log_sig_details:
                        print(f"[SIG_SHAPLEY] Converged after {k} iterations")
                    break

            if k % 10 == 0 and self.config.log_sig_details:
                print(f"[SIG_SHAPLEY] Iteration {k}/{K}, "
                      f"max change: {self._get_max_change(phi_history):.4f}")

        # Normalize using softmax
        phi_normalized = self._normalize_shapley_values(phi)

        # Create result
        result = ShapleyValues(
            patient_id=patient_id,
            fact_ids=[f.id for f in fact_set.facts],
            raw_values=phi,
            normalized_values=phi_normalized,
            convergence_achieved=converged,
            num_iterations=len(phi_history),
            model_version=model_version
        )

        # Cache result
        if self._cache is not None:
            self._cache.set(patient_id, result, model_version)

        if self.config.log_sig_details:
            print(f"[SIG_SHAPLEY] Completed: converged={converged}, "
                  f"iterations={len(phi_history)}")
            # Log top 5 facts by Shapley value
            sorted_indices = np.argsort(phi_normalized)[::-1]
            print("[SIG_SHAPLEY] Top 5 facts by contribution:")
            for i in sorted_indices[:5]:
                fact = fact_set.facts[i]
                print(f"  - {fact.id} ({phi_normalized[i]:.4f}): {fact.content[:50]}...")

        return result

    async def _compute_value_function(
        self,
        fact_subset: FactSet,
        ground_truth: Union[str, List[str]],
        llm_client: Any,
    ) -> float:
        """
        Compute v(S) = log P(A* | Q, S).

        This is the value function for Shapley computation.
        It measures how well the model can predict the correct
        diagnosis given a subset of facts.

        Supports multiple ground truth diagnoses - returns max logprob
        across all valid diagnoses.

        Args:
            fact_subset: Subset of facts (the state S)
            ground_truth: The correct diagnosis (single string or list of ICD codes)
            llm_client: Async LLM client

        Returns:
            Log probability of correct diagnosis (max across all valid codes)
        """
        # Convert facts to state string
        state_info = fact_subset.to_state_string()

        # Prepare prompt
        prompt = DIAGNOSIS_PROMPT_TEMPLATE.format(state_info=state_info)

        # Normalize ground_truth to list
        gt_list = [ground_truth] if isinstance(ground_truth, str) else list(ground_truth)

        try:
            # Get log probability for each ground truth code
            logprobs = []
            for gt in gt_list:
                logprob = await llm_client.compute_logprob(
                    prompt=prompt,
                    target=gt,
                    model=self.config.diagnosis_model,
                )
                logprobs.append(logprob)

            # Return max logprob (best matching diagnosis)
            return max(logprobs) if logprobs else -10.0

        except Exception as e:
            # If logprob computation fails, use alternative scoring
            if self.config.log_sig_details:
                print(f"[SIG_SHAPLEY] Logprob failed, using alternative: {e}")

            try:
                # Alternative: check if model generates correct diagnosis
                response = await llm_client.complete(
                    prompt=prompt,
                    model=self.config.diagnosis_model,
                    temperature=0.1,
                    max_tokens=50,
                )

                # Extract ICD codes from response
                import re
                response_codes = set(re.findall(r'[A-Z]\d{2}(?:\.\d+)?', response.upper()))

                # Check matching based on mode
                if self.config.diagnosis_match_mode == "soft":
                    # Soft match: only compare major category (first 3 chars)
                    response_major = {code[:3] for code in response_codes}
                    gt_major = {gt[:3].upper() for gt in gt_list}
                    if response_major & gt_major:
                        return 0.0  # Correct prediction (major category match)
                else:
                    # Strict match: exact code match
                    gt_codes_set = {gt.strip().upper() for gt in gt_list}
                    if response_codes & gt_codes_set:
                        return 0.0  # Correct prediction (exact match)

                # Fallback: check for partial match in text
                response_clean = response.strip().upper()
                best_score = -5.0
                for gt in gt_list:
                    gt_clean = gt.strip().upper()
                    if gt_clean in response_clean:
                        return 0.0  # Correct (found in text)
                    elif len(gt_clean) >= 3 and response_clean[:3] == gt_clean[:3]:
                        best_score = max(best_score, -1.0)  # Partial match

                return best_score

            except Exception as e2:
                print(f"[SIG_SHAPLEY] Alternative scoring failed: {e2}")
                return -10.0  # Error case

    def _check_convergence(
        self,
        phi_history: List[np.ndarray],
        epsilon: float,
        window_size: int = 10
    ) -> bool:
        """
        Check if Shapley values have converged.

        Uses relative change between recent iterations.

        Args:
            phi_history: History of Shapley value estimates
            epsilon: Convergence threshold
            window_size: Window size for averaging

        Returns:
            True if converged, False otherwise
        """
        if len(phi_history) < window_size * 2:
            return False

        # Compare recent average to earlier average
        recent = np.mean(phi_history[-window_size:], axis=0)
        earlier = np.mean(phi_history[-2 * window_size:-window_size], axis=0)

        # Compute relative change
        denominator = np.abs(earlier) + 1e-8
        relative_change = np.abs(recent - earlier) / denominator

        return np.max(relative_change) < epsilon

    def _get_max_change(self, phi_history: List[np.ndarray]) -> float:
        """Get maximum change between last two iterations."""
        if len(phi_history) < 2:
            return float('inf')
        return float(np.max(np.abs(phi_history[-1] - phi_history[-2])))

    def _normalize_shapley_values(
        self,
        raw_values: np.ndarray,
        method: str = None,
        temperature: float = None
    ) -> np.ndarray:
        """
        Normalize Shapley values using specified method.

        Supports two methods (aligned with ProMed):
        1. softmax: φ̃_i = exp(φ_i / τ) / Σexp(φ_j / τ)
        2. z_score: Z-score standardization then softmax on absolute values

        Args:
            raw_values: Raw Shapley values
            method: Normalization method ("softmax" or "z_score"), defaults to config
            temperature: Temperature parameter for softmax, defaults to config

        Returns:
            Normalized values summing to 1
        """
        if len(raw_values) == 0:
            return np.array([])

        # Use config defaults if not specified
        if method is None:
            method = self.config.shapley_normalize_method
        if temperature is None:
            temperature = self.config.shapley_temperature

        # Handle single fact case
        if len(raw_values) == 1:
            return np.array([1.0])

        # Handle case where all values are the same
        if np.std(raw_values) < 1e-8:
            return np.ones(len(raw_values)) / len(raw_values)

        try:
            if method == "z_score":
                # Z-score normalization (aligned with ProMed)
                # Step 1: Z-score standardization
                mean_score = np.mean(raw_values)
                std_score = np.std(raw_values)
                z_scores = (raw_values - mean_score) / std_score

                # Step 2: Take absolute value (importance regardless of sign)
                abs_z_scores = np.abs(z_scores)

                # Step 3: Softmax on absolute values
                shifted = abs_z_scores - np.max(abs_z_scores)
                exp_values = np.exp(shifted)
                weights = exp_values / np.sum(exp_values)

                if self.config.log_sig_details:
                    print(f"[SIG_SHAPLEY] Z-score normalization: "
                          f"mean={mean_score:.4f}, std={std_score:.4f}")

            elif method == "softmax":
                # Softmax with temperature (aligned with ProMed)
                # φ̃_i = exp(φ_i / τ) / Σexp(φ_j / τ)
                shifted = raw_values - np.max(raw_values)
                exp_values = np.exp(shifted / temperature)
                weights = exp_values / np.sum(exp_values)

                if self.config.log_sig_details:
                    print(f"[SIG_SHAPLEY] Softmax normalization: temperature={temperature}")

            else:
                raise ValueError(f"Unknown normalization method: {method}")

            # Verify validity
            if np.any(np.isnan(weights)) or np.any(np.isinf(weights)):
                print(f"[SIG_SHAPLEY] {method} produced invalid values, falling back to uniform")
                weights = np.ones(len(raw_values)) / len(raw_values)

            return weights

        except (OverflowError, FloatingPointError, ZeroDivisionError) as e:
            print(f"[SIG_SHAPLEY] {method} calculation error: {e}, using uniform weights")
            return np.ones(len(raw_values)) / len(raw_values)

    def clear_cache(self):
        """Clear the Shapley value cache."""
        if self._cache is not None:
            self._cache.clear()

    def clear_cache_for_patient(self, patient_id: str):
        """Clear cache for a specific patient."""
        if self._cache is not None:
            self._cache.clear_for_patient(patient_id)
