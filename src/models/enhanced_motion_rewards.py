"""
Enhanced Motion Reward System - FIXED
"""

import numpy as np

try:
    import torch
except ImportError:
    torch = None


class EnhancedMotionRewardShaper:
    def __init__(self):
        self.min_stable_height = 1.2
        self.critical_height = 0.9
        self.max_stable_speed = 2.0
        self.target_walk_speed = 0.5

    def _extract_state_features(self, motion_features: np.ndarray):
        if len(motion_features.shape) == 1:
            motion_features = motion_features.reshape(1, -1)

        current = motion_features[-1]

        return {
            'height': current[0],
            'vx': current[16] if len(current) > 16 else 0.0,
            'vy': current[17] if len(current) > 17 else 0.0,
            'vz': current[18] if len(current) > 18 else 0.0,
            'speed': np.linalg.norm(current[16:19]) if len(current) > 18 else 0.0,
        }

    def _compute_stability_score(self, features):
        height = features['height']
        speed = features['speed']

        # CRITICAL: If fallen, return 0 immediately
        if height < self.critical_height:
            return 0.0

        # Height score
        if height < self.min_stable_height:
            height_score = (height - self.critical_height) / (self.min_stable_height - self.critical_height)
        else:
            height_score = 1.0

        # Speed control
        if speed > self.max_stable_speed:
            speed_score = np.exp(-(speed - self.max_stable_speed))
        else:
            speed_score = 1.0

        # Combine (70% height, 30% speed)
        stability = 0.7 * height_score + 0.3 * speed_score

        return np.clip(stability, 0.0, 1.0)

    def enhanced_stability_reward(self, motion_features: np.ndarray, instruction: str) -> float:
        features = self._extract_state_features(motion_features)
        return self._compute_stability_score(features)

    def enhanced_motion_language_similarity(self, motion_sequence: np.ndarray, instruction: str) -> float:
        if torch is not None and isinstance(motion_sequence, torch.Tensor):
            motion_sequence = motion_sequence.detach().cpu().numpy()
        elif not isinstance(motion_sequence, np.ndarray):
            motion_sequence = np.array(motion_sequence)

        features = self._extract_state_features(motion_sequence)
        instruction_lower = instruction.lower()

        # STEP 1: STABILITY GATE
        stability_score = self._compute_stability_score(features)

        # If fallen or too unstable, return 0
        if stability_score < 0.3:
            return 0.0

        # STEP 2: TASK REWARD
        task_reward = 0.0

        # Detect slow mode
        slow_mode = 'slowly' in instruction_lower or 'slow' in instruction_lower
        target_speed = 0.3 if slow_mode else self.target_walk_speed

        # Standing/Static
        if any(w in instruction_lower for w in ['stand', 'stop', 'still', 'balance']):
            speed = features['speed']
            if speed < 0.1:
                task_reward = 1.0
            elif speed < 0.3:
                task_reward = 1.0 - (speed / 0.3)
            else:
                task_reward = 0.0

        # Forward movement
        elif 'forward' in instruction_lower or 'walk' in instruction_lower:
            vx = features['vx']

            if vx < 0.1:
                task_reward = 0.0
            else:
                # Gaussian around target speed
                speed_error = abs(vx - target_speed)
                task_reward = np.exp(-4.0 * (speed_error ** 2))

                # Penalty for too fast
                if vx > target_speed * 1.5:
                    task_reward *= 0.3

        # Backward movement
        elif 'backward' in instruction_lower or 'back' in instruction_lower:
            vx = features['vx']
            if vx > -0.1:
                task_reward = 0.0
            else:
                speed_error = abs(abs(vx) - target_speed)
                task_reward = np.exp(-4.0 * (speed_error ** 2))

        # Generic movement
        else:
            if features['speed'] > 0.2:
                task_reward = 0.3

        # STEP 3: APPLY STABILITY GATE
        # Stability must be >0.7 for full task reward
        if stability_score < 0.7:
            gate_factor = stability_score / 0.7
        else:
            gate_factor = 1.0

        final_score = gate_factor * task_reward

        return np.clip(final_score, 0.0, 1.0)

    def compute_multi_objective_reward(self, motion_sequence: np.ndarray,
                                      instruction: str,
                                      stability_weight: float = 0.6):
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


def test_enhanced_rewards():
    print("Testing FIXED Rewards")
    print("=" * 50)

    rewarder = EnhancedMotionRewardShaper()

    # Test 1: Fallen
    print("\n1. FALLEN (height=0.7m)")
    falling = np.zeros((1, 30))
    falling[0, 0] = 0.7
    falling[0, 16] = 1.0
    score = rewarder.enhanced_motion_language_similarity(falling, "walk forward")
    print(f"   Score: {score:.3f} (Should be 0.0)")
    assert score < 0.1, f"Failed: Score {score:.3f} should be 0.0"

    # Test 2: Too fast
    print("\n2. TOO FAST (height=1.0m, vx=2.5m/s)")
    running = np.zeros((1, 30))
    running[0, 0] = 1.0
    running[0, 16] = 2.5
    score = rewarder.enhanced_motion_language_similarity(running, "walk forward")
    print(f"   Score: {score:.3f} (Should be low)")
    assert score < 0.3, f"Failed: Score {score:.3f} should be <0.3"

    # Test 3: Good standing
    print("\n3. GOOD STANDING (height=1.3m, speed=0.05m/s)")
    standing = np.zeros((1, 30))
    standing[0, 0] = 1.3
    standing[0, 16] = 0.05
    score = rewarder.enhanced_motion_language_similarity(standing, "stand still")
    print(f"   Score: {score:.3f} (Should be high)")
    assert score > 0.7, f"Failed: Score {score:.3f} should be >0.7"

    # Test 4: Good walking
    print("\n4. GOOD WALKING (height=1.3m, vx=0.5m/s)")
    walking = np.zeros((1, 30))
    walking[0, 0] = 1.3
    walking[0, 16] = 0.5
    score = rewarder.enhanced_motion_language_similarity(walking, "walk forward")
    print(f"   Score: {score:.3f} (Should be high)")
    assert score > 0.7, f"Failed: Score {score:.3f} should be >0.7"

    print("\n" + "=" * 50)
    print("✓ ALL TESTS PASSED")


if __name__ == "__main__":
    test_enhanced_rewards()