# Copyright 2024 Shihao Xu
# SIG (Shapley Information Gain) Reward Module - Configuration

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


@dataclass
class SIGConfig:
    """
    Configuration for SIG (Shapley Information Gain) reward computation.

    This module implements process rewards based on information gain of each question,
    weighted by Shapley values as described in the SIG paper.

    Attributes:
        use_sig_reward: Master toggle for SIG reward (default False)
        sig_reward_weight: Weight coefficient β for SIG score
        correctness_bonus_weight: Weight λ_q for correctness bonus distribution
        monte_carlo_k: Number of Monte Carlo sampling iterations for Shapley estimation
        convergence_epsilon: Early stopping threshold for Shapley convergence
        min_iterations: Minimum MC iterations before checking convergence
        llm_base_url: Base URL for LLM API calls
        understanding_model: Model name for doctor understanding generation
        fact_checker_model: Model name for fact entailment checking
        diagnosis_model: Model name for computing v(S) = log P(A* | Q, S)
        use_training_model: If True, use the training model itself for all LLM calls
        batch_size: Batch size for LLM inference
        max_concurrent_requests: Maximum concurrent async requests
        timeout: Timeout for LLM API calls in seconds
        enable_caching: Enable caching for expensive computations
        cache_shapley_values: Cache Shapley values per patient (Note: should consider model version)
        cache_dir: Directory for persistent cache storage
        log_sig_details: Log detailed SIG computation information
        return_component_scores: Return breakdown of component scores
    """

    # Master toggle
    use_sig_reward: bool = False

    # Weight parameters
    sig_reward_weight: float = 0.5  # β in the formula
    correctness_bonus_weight: float = 0.3  # λ_q in the formula

    # Monte Carlo Shapley estimation parameters
    monte_carlo_k: int = 50  # Number of permutation samples (K)
    convergence_epsilon: float = 0.01  # Early stopping threshold
    min_iterations: int = 20  # Minimum iterations before convergence check

    # Shapley normalization parameters (aligned with ProMed)
    shapley_normalize_method: str = "softmax"  # "softmax" or "z_score"
    shapley_temperature: float = 1.0  # Temperature for softmax normalization (ProMed default: 2.0)

    # Fact checking fallback (aligned with ProMed)
    use_string_matching_fallback: bool = True  # Use string matching when LLM fails
    string_match_threshold: float = 0.5  # Keyword match ratio threshold (ProMed: 0.5)

    # Token-level reward parameters (aligned with ProMed)
    use_token_level_reward: bool = False  # Enable token-level reward allocation
    token_reward_alpha: float = 1.0  # Question Shapley reward weight
    token_reward_beta: float = 1.0  # Question result reward weight
    token_reward_gamma: float = 3.0  # Answer correctness reward weight
    format_reward_weight: float = 1.0  # Format reward weight

    # Diagnosis matching mode
    # "strict": Exact ICD code match (F32.1 must match F32.1)
    # "soft": Only compare major category (F32.1 matches F32.x, F32 matches F32.x)
    diagnosis_match_mode: str = "strict"

    # LLM configuration
    llm_base_url: str = "http://localhost:8000/v1"
    understanding_model: str = "Qwen3-32B"
    fact_checker_model: str = "Qwen3-32B"
    diagnosis_model: str = "Qwen3-32B"
    use_training_model: bool = True  # Use training model for dynamic rewards

    # Batch inference settings
    batch_size: int = 16
    max_concurrent_requests: int = 32
    timeout: float = 60.0

    # Caching configuration
    enable_caching: bool = True
    cache_shapley_values: bool = True
    cache_dir: str = "~/.cache/sig_reward"

    # Logging and debugging
    log_sig_details: bool = True
    return_component_scores: bool = True

    # Fact extraction configuration
    max_facts_per_patient: int = 50  # Maximum number of atomic facts to extract
    fact_extraction_temperature: float = 0.1  # Low temperature for consistent extraction

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.monte_carlo_k < 1:
            raise ValueError("monte_carlo_k must be at least 1")
        if self.convergence_epsilon <= 0:
            raise ValueError("convergence_epsilon must be positive")
        if self.sig_reward_weight < 0:
            raise ValueError("sig_reward_weight must be non-negative")
        if self.min_iterations < 1:
            raise ValueError("min_iterations must be at least 1")
        if self.min_iterations > self.monte_carlo_k:
            self.min_iterations = self.monte_carlo_k
        # Validate new parameters (aligned with ProMed)
        if self.shapley_normalize_method not in ["softmax", "z_score"]:
            raise ValueError("shapley_normalize_method must be 'softmax' or 'z_score'")
        if self.shapley_temperature <= 0:
            raise ValueError("shapley_temperature must be positive")
        if not 0 < self.string_match_threshold <= 1:
            raise ValueError("string_match_threshold must be in (0, 1]")
        if self.diagnosis_match_mode not in ["strict", "soft"]:
            raise ValueError("diagnosis_match_mode must be 'strict' or 'soft'")

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'SIGConfig':
        """Create SIGConfig from a dictionary, ignoring unknown keys."""
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_dict = {k: v for k, v in config_dict.items() if k in valid_keys}
        return cls(**filtered_dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'use_sig_reward': self.use_sig_reward,
            'sig_reward_weight': self.sig_reward_weight,
            'correctness_bonus_weight': self.correctness_bonus_weight,
            'monte_carlo_k': self.monte_carlo_k,
            'convergence_epsilon': self.convergence_epsilon,
            'min_iterations': self.min_iterations,
            'shapley_normalize_method': self.shapley_normalize_method,
            'shapley_temperature': self.shapley_temperature,
            'use_string_matching_fallback': self.use_string_matching_fallback,
            'string_match_threshold': self.string_match_threshold,
            'use_token_level_reward': self.use_token_level_reward,
            'token_reward_alpha': self.token_reward_alpha,
            'token_reward_beta': self.token_reward_beta,
            'token_reward_gamma': self.token_reward_gamma,
            'format_reward_weight': self.format_reward_weight,
            'diagnosis_match_mode': self.diagnosis_match_mode,
            'llm_base_url': self.llm_base_url,
            'understanding_model': self.understanding_model,
            'fact_checker_model': self.fact_checker_model,
            'diagnosis_model': self.diagnosis_model,
            'use_training_model': self.use_training_model,
            'batch_size': self.batch_size,
            'max_concurrent_requests': self.max_concurrent_requests,
            'timeout': self.timeout,
            'enable_caching': self.enable_caching,
            'cache_shapley_values': self.cache_shapley_values,
            'cache_dir': self.cache_dir,
            'log_sig_details': self.log_sig_details,
            'return_component_scores': self.return_component_scores,
            'max_facts_per_patient': self.max_facts_per_patient,
            'fact_extraction_temperature': self.fact_extraction_temperature,
        }
