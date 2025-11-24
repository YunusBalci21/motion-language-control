# src/models/enhanced_motion_rewards.py
"""
Enhanced Motion Reward System - WORKING VERSION
"""

import numpy as np

try:
    import torch
except ImportError:
    torch = None


class EnhancedMotionRewardShaper:
    def __init__(self):
        self.min_stable_height = 1.0
        self.critical_height = 0.8
        self.max_stable_speed = 3.0
        self.target_walk_speed = 0.8

    def _extract_state_features(self, motion_features: np.ndarray):
        """Extract state from motion features"""
        if len(motion_features.shape) == 1:
            motion_features = motion_features.reshape(1, -1)

        current = motion_features[-1]

        return {
            'height': float(current[0]) if current[0] > 0.1 else 1.0,
            'vx': float(current[16]) if len(current) > 16 else 0.0,
            'vy': float(current[17]) if len(current) > 17 else 0.0,
            'vz': float(current[18]) if len(current) > 18 else 0.0,
            'speed': float(current[26]) if len(current) > 26 else 0.0,
        }

    def _compute_stability_score(self, features):
        """Compute stability score"""
        height = features['height']
        speed = features['speed']

        if height < 0.2:
            return 0.0

        if height < 0.5:
            height_score = (height - 0.2) / 0.3
        else:
            height_score = 1.0

        if speed > self.max_stable_speed:
            speed_score = np.exp(-(speed - self.max_stable_speed))
        else:
            speed_score = 1.0

        stability = 0.5 * height_score + 0.5 * speed_score
        return np.clip(stability, 0.0, 1.0)

    def enhanced_stability_reward(self, motion_features: np.ndarray, instruction: str) -> float:
        features = self._extract_state_features(motion_features)
        return self._compute_stability_score(features)

    def enhanced_motion_language_similarity(self, motion_sequence: np.ndarray, instruction: str) -> float:
        """Compute motion-language similarity"""
        if torch is not None and isinstance(motion_sequence, torch.Tensor):
            motion_sequence = motion_sequence.detach().cpu().numpy()
        elif not isinstance(motion_sequence, np.ndarray):
            motion_sequence = np.array(motion_sequence)

        features = self._extract_state_features(motion_sequence)
        instruction_lower = instruction.lower()

        # Stability gate
        stability_score = self._compute_stability_score(features)
        if stability_score < 0.2:
            return 0.0

        task_reward = 0.0
        slow_mode = 'slowly' in instruction_lower or 'slow' in instruction_lower
        target_speed = 0.4 if slow_mode else 0.8

        # Forward movement
        if 'forward' in instruction_lower or 'walk' in instruction_lower:
            vx = features['vx']

            if vx > 0.3:
                # Great forward movement
                task_reward = 0.9
            elif vx > 0.1:
                # Good forward movement
                task_reward = 0.7
            elif vx > 0.05:
                # Some forward movement
                task_reward = 0.4
            else:
                # Not moving forward
                task_reward = 0.0

        # Backward movement
        elif 'backward' in instruction_lower or 'back' in instruction_lower:
            vx = features['vx']
            if vx < -0.3:
                task_reward = 0.9
            elif vx < -0.1:
                task_reward = 0.7
            elif vx < -0.05:
                task_reward = 0.4
            else:
                task_reward = 0.0

        # Stand still
        elif any(w in instruction_lower for w in ['stand', 'stop', 'still', 'balance']):
            speed = features['speed']
            if speed < 0.1:
                task_reward = 0.8
            elif speed < 0.3:
                task_reward = 0.5
            else:
                task_reward = 0.2

        # Generic movement
        else:
            if features['speed'] > 0.2:
                task_reward = 0.4

        # Apply stability gate
        if stability_score < 0.5:
            gate_factor = stability_score / 0.5
        else:
            gate_factor = 1.0

        final_score = gate_factor * task_reward
        return np.clip(final_score, 0.0, 1.0)

    def compute_multi_objective_reward(self, motion_sequence: np.ndarray,
                                      instruction: str,
                                      stability_weight: float = 0.4):
        features = self._extract_state_features(motion_sequence)

        stability_reward = self.enhanced_stability_reward(motion_sequence, instruction)
        task_reward = self.enhanced_motion_language_similarity(motion_sequence, instruction)

        total = stability_weight * stability_reward + (1 - stability_weight) * task_reward

        info = {
            'stability_reward': stability_reward,
            'task_reward': task_reward,
            'total_reward': total,
            'height': features['height'],
            'speed': features['speed'],
            'vx': features['vx'],
        }

        return total, info