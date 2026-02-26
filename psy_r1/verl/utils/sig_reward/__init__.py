# Copyright 2024 Shihao Xu
# SIG (Shapley Information Gain) Reward Module
#
# This module implements process rewards for multi-turn dialogue training
# based on Shapley Information Gain as described in the SIG paper.
#
# Usage:
#   from psy_r1.verl.utils.sig_reward import SIGCalculator, SIGConfig
#
#   config = SIGConfig(use_sig_reward=True, sig_reward_weight=0.5)
#   calculator = SIGCalculator(config)
#   await calculator.initialize()
#   result = await calculator.compute_sig_reward(...)

from .sig_config import SIGConfig
from .atomic_facts import AtomicFact, FactSet, AtomicFactExtractor
from .doctor_understanding import DoctorUnderstanding, DoctorUnderstandingGenerator
from .fact_checker import FactChecker
from .shapley_estimator import ShapleyValues, MonteCarloShapleyEstimator
from .sig_calculator import (
    SIGCalculator, SIGResult, TokenLevelReward, AsyncLLMClient, compute_sig_reward_sync
)


__all__ = [
    # Configuration
    'SIGConfig',

    # Data structures
    'AtomicFact',
    'FactSet',
    'DoctorUnderstanding',
    'ShapleyValues',
    'SIGResult',
    'TokenLevelReward',  # Token-level reward (aligned with ProMed)

    # Components
    'AtomicFactExtractor',
    'DoctorUnderstandingGenerator',
    'FactChecker',
    'MonteCarloShapleyEstimator',

    # Main calculator
    'SIGCalculator',
    'AsyncLLMClient',

    # Sync wrapper
    'compute_sig_reward_sync',
]
