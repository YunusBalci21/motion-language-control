# src/models/enhanced_motion_rewards.py
"""
Enhanced Motion Reward System for Humanoid-v4
Fixed observation space parsing and improved reward shaping
"""

import numpy as np
from typing import Dict, Tuple, Optional

try:
    import torch
except ImportError:
    torch = None


class EnhancedMotionRewardShaper:
    """
    Reward shaper for motion-language control of Humanoid-v4.

    Humanoid-v4 Observation Space (376 dims total):
    - [0]: z position (torso height from ground)
    - [1:5]: root quaternion (qw, qx, qy, qz)
    - [5:22]: joint positions (17 joints)
    - [22:25]: root linear velocity (vx, vy, vz) in world frame
    - [25:28]: root angular velocity (wx, wy, wz)
    - [28:45]: joint velocities (17 joints)
    - [45:185]: cinert (10*14 = 140 values)
    - [185:269]: cvel (14*6 = 84 values)
    - [269:286]: qfrc_actuator (17 values)
    - [286:370]: cfrc_ext (14*6 = 84 values)
    """

    # Observation indices for Humanoid-v4
    IDX_HEIGHT = 0
    IDX_QUAT_START = 1
    IDX_QUAT_END = 5
    IDX_JOINT_POS_START = 5
    IDX_JOINT_POS_END = 22
    IDX_ROOT_VEL_START = 22  # vx, vy, vz
    IDX_ROOT_VEL_END = 25
    IDX_ROOT_ANGVEL_START = 25  # wx, wy, wz
    IDX_ROOT_ANGVEL_END = 28
    IDX_JOINT_VEL_START = 28
    IDX_JOINT_VEL_END = 45

    def __init__(self, config: Optional[Dict] = None):
        config = config or {}

        # Height thresholds
        self.standing_height = config.get('standing_height', 1.25)  # Good standing
        self.min_stable_height = config.get('min_stable_height', 1.0)  # Still okay
        self.critical_height = config.get('critical_height', 0.7)  # Falling

        # Speed thresholds
        self.target_walk_speed = config.get('target_walk_speed', 0.8)
        self.target_slow_speed = config.get('target_slow_speed', 0.4)
        self.max_stable_speed = config.get('max_stable_speed', 2.5)

        # Reward weights (for multi-objective)
        self.stability_weight = config.get('stability_weight', 0.4)
        self.task_weight = config.get('task_weight', 0.6)

        # Smoothing for temporal consistency
        self.prev_reward = 0.0
        self.reward_smoothing = config.get('reward_smoothing', 0.1)

    def _to_numpy(self, x) -> np.ndarray:
        """Convert input to numpy array."""
        if torch is not None and isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        elif isinstance(x, np.ndarray):
            return x
        else:
            return np.array(x)

    def _extract_state_features(self, obs: np.ndarray) -> Dict[str, float]:
        """
        Extract interpretable features from Humanoid-v4 observation.

        Args:
            obs: Observation array, can be (obs_dim,) or (seq_len, obs_dim)

        Returns:
            Dictionary with extracted features
        """
        obs = self._to_numpy(obs)

        # Handle sequence input - take last observation
        if len(obs.shape) == 2:
            obs = obs[-1]

        # Validate observation size
        if len(obs) < 45:
            # Fallback for non-standard observations
            return self._extract_minimal_features(obs)

        # Extract all features
        height = float(obs[self.IDX_HEIGHT])

        # Root quaternion (w, x, y, z)
        quat = obs[self.IDX_QUAT_START:self.IDX_QUAT_END]
        qw, qx, qy, qz = quat

        # Compute roll, pitch, yaw from quaternion
        # Roll (x-axis rotation)
        sinr_cosp = 2 * (qw * qx + qy * qz)
        cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        # Pitch (y-axis rotation)
        sinp = 2 * (qw * qy - qz * qx)
        sinp = np.clip(sinp, -1.0, 1.0)
        pitch = np.arcsin(sinp)

        # Yaw (z-axis rotation)
        siny_cosp = 2 * (qw * qz + qx * qy)
        cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        # Root velocities
        vx = float(obs[self.IDX_ROOT_VEL_START])  # Forward/backward
        vy = float(obs[self.IDX_ROOT_VEL_START + 1])  # Left/right
        vz = float(obs[self.IDX_ROOT_VEL_START + 2])  # Up/down

        # Angular velocities
        wx = float(obs[self.IDX_ROOT_ANGVEL_START])
        wy = float(obs[self.IDX_ROOT_ANGVEL_START + 1])
        wz = float(obs[self.IDX_ROOT_ANGVEL_START + 2])

        # Joint velocities for energy computation
        joint_vels = obs[self.IDX_JOINT_VEL_START:self.IDX_JOINT_VEL_END]

        # Derived quantities
        horizontal_speed = np.sqrt(vx ** 2 + vy ** 2)
        total_speed = np.sqrt(vx ** 2 + vy ** 2 + vz ** 2)
        angular_speed = np.sqrt(wx ** 2 + wy ** 2 + wz ** 2)
        joint_energy = np.sum(np.square(joint_vels))

        # Uprightness (1.0 = perfectly upright)
        # Using the z-component of the up vector after rotation
        uprightness = 1 - 2 * (qx ** 2 + qy ** 2)  # Simplified from rotation matrix
        uprightness = float(np.clip(uprightness, -1.0, 1.0))

        return {
            'height': height,
            'roll': float(roll),
            'pitch': float(pitch),
            'yaw': float(yaw),
            'vx': vx,
            'vy': vy,
            'vz': vz,
            'wx': wx,
            'wy': wy,
            'wz': wz,
            'horizontal_speed': horizontal_speed,
            'total_speed': total_speed,
            'angular_speed': angular_speed,
            'joint_energy': float(joint_energy),
            'uprightness': uprightness,
        }

    def _extract_minimal_features(self, obs: np.ndarray) -> Dict[str, float]:
        """Fallback feature extraction for non-standard observations."""
        return {
            'height': float(obs[0]) if len(obs) > 0 else 1.0,
            'roll': 0.0,
            'pitch': 0.0,
            'yaw': 0.0,
            'vx': 0.0,
            'vy': 0.0,
            'vz': 0.0,
            'wx': 0.0,
            'wy': 0.0,
            'wz': 0.0,
            'horizontal_speed': 0.0,
            'total_speed': 0.0,
            'angular_speed': 0.0,
            'joint_energy': 0.0,
            'uprightness': 1.0,
        }

    def compute_stability_reward(self, obs: np.ndarray) -> Tuple[float, Dict]:
        """
        Compute stability reward based on balance and posture.

        Returns:
            Tuple of (reward, info_dict)
        """
        features = self._extract_state_features(obs)

        height = features['height']
        pitch = features['pitch']
        roll = features['roll']
        uprightness = features['uprightness']
        angular_speed = features['angular_speed']
        vz = features['vz']

        # 1. Height reward (Gaussian around target height)
        height_error = abs(height - self.standing_height)
        height_reward = np.exp(-2.0 * height_error ** 2)

        # Severe penalty for low height
        if height < self.critical_height:
            height_reward = 0.0
        elif height < self.min_stable_height:
            height_reward *= 0.5

        # 2. Uprightness reward
        upright_reward = np.exp(-3.0 * (pitch ** 2 + roll ** 2))

        # 3. Angular stability (penalize spinning/wobbling)
        angular_penalty = np.exp(-0.5 * angular_speed ** 2)

        # 4. Vertical velocity penalty (penalize falling)
        falling_penalty = 1.0
        if vz < -0.5:  # Falling fast
            falling_penalty = np.exp(vz)  # Exponential penalty

        # Combine stability components
        stability = (
                0.35 * height_reward +
                0.35 * upright_reward +
                0.15 * angular_penalty +
                0.15 * falling_penalty
        )

        info = {
            'height_reward': height_reward,
            'upright_reward': upright_reward,
            'angular_penalty': angular_penalty,
            'falling_penalty': falling_penalty,
            'stability_total': stability,
        }

        return float(np.clip(stability, 0.0, 1.0)), info

    def compute_task_reward(self, obs: np.ndarray, instruction: str) -> Tuple[float, Dict]:
        """
        Compute task-specific reward based on instruction.

        Args:
            obs: Observation array
            instruction: Natural language instruction

        Returns:
            Tuple of (reward, info_dict)
        """
        features = self._extract_state_features(obs)
        instruction_lower = instruction.lower().strip()

        vx = features['vx']
        vy = features['vy']
        horizontal_speed = features['horizontal_speed']
        height = features['height']
        yaw = features['yaw']

        task_reward = 0.0
        task_info = {'matched_task': 'none'}

        # Check for speed modifiers
        is_slow = any(w in instruction_lower for w in ['slow', 'carefully', 'gently'])
        is_fast = any(w in instruction_lower for w in ['fast', 'quick', 'run'])

        if is_slow:
            target_speed = self.target_slow_speed
        elif is_fast:
            target_speed = self.target_walk_speed * 1.5
        else:
            target_speed = self.target_walk_speed

        # === FORWARD MOVEMENT ===
        if any(w in instruction_lower for w in ['forward', 'walk', 'move ahead', 'go straight']):
            task_info['matched_task'] = 'walk_forward'

            if vx > 0.2:
                # Prefer matching target_speed (penalize going too fast OR too slow)
                speed_match = np.exp(-2.0 * (vx - target_speed) ** 2)

                # Still require forward movement (prevents gaming by matching via noise)
                movement_bonus = np.clip(vx / max(target_speed, 1e-6), 0.0, 1.0)

                drift_penalty = np.exp(-2.0 * vy ** 2)

                task_reward = (0.6 * speed_match + 0.4 * movement_bonus) * drift_penalty
            elif vx > 0.05:  # Some forward movement
                task_reward = 0.3 + 0.4 * (vx / 0.2)
            elif vx > 0:  # Tiny forward movement - still reward it!
                task_reward = 0.2 * (vx / 0.05)
            else:
                # Moving backward - no reward
                task_reward = 0.0

        # === BACKWARD MOVEMENT ===
        elif any(w in instruction_lower for w in ['backward', 'back', 'reverse', 'retreat']):
            task_info['matched_task'] = 'walk_backward'

            if vx < -0.05:  # Moving backward
                speed_match = np.exp(-2.0 * (-vx - target_speed) ** 2)
                movement_bonus = min(-vx / target_speed, 1.0)
                drift_penalty = np.exp(-2.0 * vy ** 2)

                task_reward = 0.5 * speed_match + 0.3 * movement_bonus + 0.2 * drift_penalty
            else:
                task_reward = max(0, -vx) * 0.5

        # === TURN LEFT ===
        elif any(w in instruction_lower for w in ['turn left', 'rotate left', 'left turn']):
            task_info['matched_task'] = 'turn_left'
            wz = features['wz']

            if wz > 0.1:  # Turning left (positive angular velocity around z)
                task_reward = min(wz / 0.5, 1.0)
            else:
                task_reward = max(0, wz) * 0.5

        # === TURN RIGHT ===
        elif any(w in instruction_lower for w in ['turn right', 'rotate right', 'right turn']):
            task_info['matched_task'] = 'turn_right'
            wz = features['wz']

            if wz < -0.1:  # Turning right (negative angular velocity)
                task_reward = min(-wz / 0.5, 1.0)
            else:
                task_reward = max(0, -wz) * 0.5

        # === STAND STILL ===
        elif any(w in instruction_lower for w in ['stand', 'stop', 'still', 'stay', 'balance', 'halt']):
            task_info['matched_task'] = 'stand_still'

            # Strong reward for minimal movement
            if horizontal_speed < 0.05:
                task_reward = 1.0  # Perfect stillness
            elif horizontal_speed < 0.1:
                task_reward = 0.8
            elif horizontal_speed < 0.2:
                task_reward = 0.5
            else:
                # Penalize movement when told to stand still
                task_reward = max(0, 0.3 - horizontal_speed * 0.5)

        # === SIDESTEP LEFT ===
        elif 'left' in instruction_lower and 'step' in instruction_lower:
            task_info['matched_task'] = 'sidestep_left'

            if vy > 0.05:
                task_reward = min(vy / 0.3, 1.0)
            else:
                task_reward = max(0, vy) * 0.5

        # === SIDESTEP RIGHT ===
        elif 'right' in instruction_lower and 'step' in instruction_lower:
            task_info['matched_task'] = 'sidestep_right'

            if vy < -0.05:
                task_reward = min(-vy / 0.3, 1.0)
            else:
                task_reward = max(0, -vy) * 0.5

        # === DEFAULT: Generic movement ===
        else:
            task_info['matched_task'] = 'generic'
            # Reward any controlled movement
            if horizontal_speed > 0.1 and height > self.min_stable_height:
                task_reward = 0.3

        task_info['task_reward'] = task_reward
        task_info['target_speed'] = target_speed

        return float(np.clip(task_reward, 0.0, 1.0)), task_info

    def compute_energy_penalty(self, obs: np.ndarray) -> float:
        """Compute energy efficiency penalty."""
        features = self._extract_state_features(obs)
        joint_energy = features['joint_energy']

        # Normalize and convert to penalty (higher energy = lower reward)
        energy_penalty = np.exp(-0.001 * joint_energy)
        return float(energy_penalty)

    def enhanced_motion_language_similarity(
            self,
            motion_sequence: np.ndarray,
            instruction: str
    ) -> float:
        """
        Main interface for computing motion-language similarity reward.

        This is the primary method called by the training loop.
        """
        obs = self._to_numpy(motion_sequence)

        # Compute stability (gate for task reward)
        stability_reward, stability_info = self.compute_stability_reward(obs)

        # If very unstable, don't reward task completion
        if stability_reward < 0.2:
            return stability_reward * 0.5  # Small reward for not falling completely

        # Compute task reward
        task_reward, task_info = self.compute_task_reward(obs, instruction)

        # Stability-gated task reward
        # Higher stability = task reward matters more
        gate = np.clip(stability_reward, 0.3, 1.0)

        combined = gate * task_reward

        return float(np.clip(combined, 0.0, 1.0))

    def enhanced_stability_reward(
            self,
            motion_features: np.ndarray,
            instruction: str
    ) -> float:
        """Legacy interface for stability reward."""
        reward, _ = self.compute_stability_reward(motion_features)
        return reward

    def compute_multi_objective_reward(
            self,
            motion_sequence: np.ndarray,
            instruction: str,
            stability_weight: Optional[float] = None
    ) -> Tuple[float, Dict]:
        """
        Compute comprehensive multi-objective reward.

        Args:
            motion_sequence: Observation or sequence of observations
            instruction: Natural language instruction
            stability_weight: Weight for stability vs task (default: self.stability_weight)

        Returns:
            Tuple of (total_reward, info_dict)
        """
        obs = self._to_numpy(motion_sequence)

        if stability_weight is None:
            stability_weight = self.stability_weight

        # Extract features once
        features = self._extract_state_features(obs)

        # Compute components
        stability_reward, stability_info = self.compute_stability_reward(obs)
        task_reward, task_info = self.compute_task_reward(obs, instruction)
        energy_penalty = self.compute_energy_penalty(obs)

        # Weighted combination
        total = (
                stability_weight * stability_reward +
                (1 - stability_weight) * task_reward * 0.9 +
                0.1 * energy_penalty  # Small energy efficiency bonus
        )

        # Apply smoothing for temporal consistency
        smoothed_total = (
                self.reward_smoothing * self.prev_reward +
                (1 - self.reward_smoothing) * total
        )
        self.prev_reward = smoothed_total

        info = {
            'stability_reward': stability_reward,
            'task_reward': task_reward,
            'energy_penalty': energy_penalty,
            'total_reward': smoothed_total,
            'raw_total': total,
            'height': features['height'],
            'vx': features['vx'],
            'vy': features['vy'],
            'horizontal_speed': features['horizontal_speed'],
            'pitch': features['pitch'],
            'roll': features['roll'],
            'uprightness': features['uprightness'],
            **task_info,
            **{f'stab_{k}': v for k, v in stability_info.items()},
        }

        return float(np.clip(smoothed_total, 0.0, 1.0)), info


# Convenience function for creating reward shaper
def create_reward_shaper(config: Optional[Dict] = None) -> EnhancedMotionRewardShaper:
    """Factory function for creating reward shaper."""
    return EnhancedMotionRewardShaper(config)