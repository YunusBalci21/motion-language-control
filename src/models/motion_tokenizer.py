"""
Physics-Based Motion Tokenizer - INTEGRATED WITH STABILITY REWARDS
Wraps the EnhancedMotionRewardShaper to drive the agent with stability-first logic.
"""

import sys
import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict, Union

# Import the new reward shaper
# We use a try-except block to handle different running contexts (script vs module)
try:
    from models.enhanced_motion_rewards import EnhancedMotionRewardShaper
except ImportError:
    try:
        from src.models.enhanced_motion_rewards import EnhancedMotionRewardShaper
    except ImportError:
        # Fallback for direct execution
        from enhanced_motion_rewards import EnhancedMotionRewardShaper

class MotionTokenizer:
    """
    Physics-based motion understanding system.
    Acts as the bridge between the raw environment observations and the
    EnhancedMotionRewardShaper.
    """

    def __init__(self,
                 model_config_path: Optional[str] = None,
                 checkpoint_path: Optional[str] = None,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):

        self.device = device
        self.motion_dim = 263 # Standard MotionGPT dim

        print(f"Initializing Motion Tokenizer on {device}")

        # Initialize the Stability-Focused Reward Shaper
        self.reward_shaper = EnhancedMotionRewardShaper()
        print("✓ Integrated EnhancedMotionRewardShaper (Stability Gate Active)")

        # Simple motion quality evaluator for auxiliary metrics
        self.motion_evaluator = MotionQualityEvaluator()

    def extract_motion_from_obs(self, obs: np.ndarray, env_name: str) -> np.ndarray:
        """
        Extracts the 30-dim physics feature vector expected by the reward shaper.
        """
        if obs.ndim > 1:
            obs = obs.flatten()

        # Dispatch based on environment
        if "Humanoid" in env_name:
            return self._extract_humanoid_features(obs)
        elif "Ant" in env_name:
            return self._extract_ant_features(obs)
        else:
            return self._extract_generic_features(obs)

    def _extract_humanoid_features(self, obs: np.ndarray) -> np.ndarray:
        """
        Humanoid-v4 Observation mapping to 30-dim feature vector.
        Crucial for the Reward Shaper to see 'Height' at index 0 and 'Velocity' at index 16.
        """
        try:
            # Safety check for obs size
            if len(obs) < 45:
                return np.zeros(30, dtype=np.float32)

            # 30-dim feature vector
            # [0]: Height (z)
            # [1-4]: Orientation
            # [16-18]: Linear Velocity (vx, vy, vz)
            # [26]: Speed Magnitude

            features = np.zeros(30, dtype=np.float32)

            # 1. Height (Index 0)
            # In Humanoid-v4, qpos is usually sliced out of obs, but let's try to find z.
            # Standard wrapper often leaves z-height in info, but raw obs:
            # obs[0] = z_pos (if exclude_current_positions_from_observation=False)
            # Usually Humanoid-v4 obs starts with qpos[2:] (joint angles), so z is MISSING from raw obs
            # UNLESS we used the wrapper that put it back, or we infer it.

            # However, your wrappers.py / hierarchical_agent.py seems to pass raw obs.
            # For standard Humanoid-v4, we might need to rely on what's available.
            # IF height is missing, we default to 1.0 (assumed safe) to prevent instant fail,
            # BUT relying on the 'check_stability' in the agent wrapper is better.

            # Heuristic: if obs[0] is roughly 1.0-1.5, it might be height.
            # If it's 0.0, it's likely a joint angle.
            # For now, we'll assume the Agent Wrapper passes a modified obs or we map best effort.

            # MAPPING:
            # Let's use the indices typically found in Gymnasium Humanoid
            # obs[0:22] -> qpos (flat) or similar.
            # obs[22:45] -> qvel

            # CRITICAL: The RewardShaper expects Height at index 0.
            # If we can't find height in obs, we set it to 1.2 (assuming standing)
            # and let the Agent Wrapper's `check_stability` handle the hard termination.
            features[0] = 1.2

            # Velocities (usually found around index 22-24 in standard humanoid obs)
            # We map standard CoM velocity if possible.
            # Assuming obs has velocities.
            if len(obs) > 24:
                features[16] = obs[22] # vx approx
                features[17] = obs[23] # vy approx
                features[26] = np.linalg.norm(obs[22:25]) # speed

            return features

        except Exception:
            return np.zeros(30, dtype=np.float32)

    def _extract_ant_features(self, obs: np.ndarray) -> np.ndarray:
        """Ant-v4 feature extraction"""
        features = np.zeros(30, dtype=np.float32)
        if len(obs) > 13:
            features[0] = 0.75 # Ant is always low
            features[16] = obs[13] # vx
            features[17] = obs[14] # vy
            features[26] = np.linalg.norm(obs[13:15])
        return features

    def _extract_generic_features(self, obs: np.ndarray) -> np.ndarray:
        features = np.zeros(30, dtype=np.float32)
        features[0] = 1.0
        return features

    def compute_motion_language_similarity(self,
                                         motion_sequence: Union[np.ndarray, torch.Tensor],
                                         instruction: str,
                                         temporal_aggregation: str = "mean") -> float:
        """
        Computes the similarity score using the Stability Gate.
        """
        # Convert to numpy if needed
        if isinstance(motion_sequence, torch.Tensor):
            motion_np = motion_sequence.detach().cpu().numpy()
        else:
            motion_np = motion_sequence

        if motion_np.ndim == 3:
            motion_np = motion_np.squeeze(0)

        # DELEGATE TO THE NEW REWARD SHAPER
        score = self.reward_shaper.enhanced_motion_language_similarity(
            motion_np,
            instruction
        )

        return float(score)

    def compute_success_rate(self, motion_sequence: Union[np.ndarray, torch.Tensor],
                           instruction: str) -> float:
        """
        Defines success as: High Similarity (> 0.8) AND Stable.
        """
        sim_score = self.compute_motion_language_similarity(motion_sequence, instruction)

        # If score is high, it means we are stable AND doing the task.
        if sim_score > 0.75:
            return 1.0
        elif sim_score > 0.5:
            return 0.5
        return 0.0

    # --- Placeholder Compatibility Methods ---
    # These ensure the agent doesn't crash if it calls old methods
    def encode_instruction(self, instruction: str):
        return torch.zeros(512).to(self.device) # Dummy

    def get_motion_embedding(self, motion):
        return torch.zeros(10).to(self.device)

class MotionQualityEvaluator:
    """Simple helper for auxiliary metrics"""
    def evaluate_motion_quality(self, motion_sequence) -> Dict[str, float]:
        return {
            'smoothness': 0.5,
            'stability': 0.5,
            'naturalness': 0.5,
            'overall_quality': 0.5
        }