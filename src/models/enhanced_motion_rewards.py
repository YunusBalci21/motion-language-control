#!/usr/bin/env python3
"""
Enhanced Motion Reward System
Addresses low similarity scores with better reward shaping
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, Tuple


class EnhancedMotionRewardShaper:
    """Enhanced reward shaping specifically for stability instructions"""

    def __init__(self):
        self.stability_keywords = [
            'stably', 'stable', 'balance', 'carefully', 'without falling',
            'maintain balance', 'steady', 'smoothly'
        ]

        self.movement_keywords = [
            'walk', 'move', 'step', 'forward', 'backward',
            'turn', 'rotate', 'run', 'march'
        ]

    def enhanced_stability_reward(self, motion_features: np.ndarray,
                                  instruction: str) -> float:
        """
        Enhanced reward specifically for stability-focused instructions
        """
        if len(motion_features.shape) == 1:
            motion_features = motion_features.reshape(1, -1)

        instruction_lower = instruction.lower()

        # Base stability metrics
        stability_reward = 0.0

        # 1. Height consistency (staying upright)
        if motion_features.shape[1] > 0:
            height_values = motion_features[:, 0]  # Z coordinate
            height_consistency = 1.0 - np.var(height_values)
            if height_consistency > 0:
                stability_reward += height_consistency * 0.3

        # 2. Orientation stability (not rolling/pitching too much)
        if motion_features.shape[1] > 7:
            # Roll and pitch from euler angles
            roll_values = motion_features[:, 3] if motion_features.shape[1] > 3 else np.zeros(motion_features.shape[0])
            roll_stability = np.exp(-np.var(roll_values))
            stability_reward += roll_stability * 0.2

        # 3. Movement consistency for walking
        if any(word in instruction_lower for word in ['walk', 'move', 'step']):
            if motion_features.shape[1] > 24:
                movement_values = motion_features[:, 24]  # Overall movement magnitude
                movement_consistency = 1.0 - np.var(movement_values)
                if movement_consistency > 0:
                    stability_reward += movement_consistency * 0.2

        # 4. Bonus for stability keywords
        stability_bonus = 0.0
        for keyword in self.stability_keywords:
            if keyword in instruction_lower:
                stability_bonus += 0.1

        stability_reward += min(stability_bonus, 0.3)  # Cap at 0.3

        # 5. Forward progress for walking instructions
        if 'forward' in instruction_lower and motion_features.shape[1] > 29:
            # Use forward displacement
            fwd_disp = motion_features[:, 29] if motion_features.shape[1] > 29 else 0
            forward_progress = np.mean(np.maximum(fwd_disp, 0)) * 0.2
            stability_reward += forward_progress

        return np.clip(stability_reward, 0.0, 1.0)

    def enhanced_motion_language_similarity(self, motion_sequence: np.ndarray,
                                            instruction: str) -> float:
        """
        Enhanced similarity computation with better stability focus
        """
        # Get base stability reward
        stability_reward = self.enhanced_stability_reward(motion_sequence, instruction)

        # Movement detection
        movement_detected = 0.0
        if motion_sequence.shape[1] > 24:
            movement_magnitude = np.mean(motion_sequence[:, 24])
            if movement_magnitude > 0.01:  # Lower threshold
                movement_detected = 0.3

        # Instruction-specific bonuses
        instruction_bonus = 0.0
        instruction_lower = instruction.lower()

        if 'stably' in instruction_lower:
            instruction_bonus = stability_reward * 0.5

        if 'walk' in instruction_lower:
            if movement_detected > 0:
                instruction_bonus += 0.2

        if 'forward' in instruction_lower:
            # Check for forward motion indicators
            if motion_sequence.shape[1] > 16:
                forward_indicators = np.mean(motion_sequence[:, 16:20])  # Joint velocities
                if forward_indicators > 0.01:
                    instruction_bonus += 0.2

        # Combine all components
        total_similarity = (
                0.4 * stability_reward +
                0.3 * movement_detected +
                0.3 * instruction_bonus
        )

        return np.clip(total_similarity, 0.0, 1.0)


def test_enhanced_rewards():
    """Test the enhanced reward system"""
    print("Testing Enhanced Motion Rewards")
    print("=" * 40)

    rewarder = EnhancedMotionRewardShaper()

    # Create test motion data
    stable_motion = np.random.randn(10, 30) * 0.05  # Small movements = stable
    stable_motion[:, 0] = 1.0 + np.random.randn(10) * 0.02  # Consistent height

    unstable_motion = np.random.randn(10, 30) * 0.3  # Large movements = unstable
    unstable_motion[:, 0] = 1.0 + np.random.randn(10) * 0.2  # Variable height

    # Test on stability instruction
    instruction = "walk stably"

    stable_sim = rewarder.enhanced_motion_language_similarity(stable_motion, instruction)
    unstable_sim = rewarder.enhanced_motion_language_similarity(unstable_motion, instruction)

    print(f"Instruction: '{instruction}'")
    print(f"Stable motion similarity: {stable_sim:.3f}")
    print(f"Unstable motion similarity: {unstable_sim:.3f}")

    if stable_sim > unstable_sim:
        print("✓ Enhanced rewards correctly prefer stable motion")
    else:
        print("✗ Enhanced rewards need more tuning")

    return rewarder


if __name__ == "__main__":
    test_enhanced_rewards()