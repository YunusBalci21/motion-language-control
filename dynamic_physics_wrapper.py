import sys
import gymnasium as gym
import numpy as np
import time
from collections import deque

# Ensure we can import from src
sys.path.append("src")
from agents.hierarchical_agent import DirectMotionLanguageWrapper


class DynamicPhysicsWrapper(DirectMotionLanguageWrapper):
    """
    An upgraded environment wrapper that allows the LLM to adjust
    reward weights and parameters dynamically during training.
    """

    def __init__(self, env, tokenizer, **kwargs):
        super().__init__(env, tokenizer, **kwargs)

        # Dynamic Hyperparameters (controlled by LLM)
        self.params = {
            "forward_reward_weight": 2.0,  # How much to reward moving forward
            "stability_weight": 1.0,  # How much to penalize falling
            "energy_penalty": 1e-3,  # Penalty for aggressive movement
            "velocity_threshold": 0.05,  # Minimum speed to get "moving" bonus
            "target_speed": 1.0,  # Desired speed
            "action_magnitude": 1.0,  # Action clipping range
            "drift_penalty": 0.1  # Penalty for moving sideways
        }

    def update_parameters(self, new_params: dict):
        """Called by the Coach to update physics rules"""
        for k, v in new_params.items():
            if k in self.params:
                self.params[k] = float(v)
        # print(f"  ⚡ Physics params updated: {self.params}")

    def compute_instruction_progress_bonus(self, vx: float, yaw_rate: float) -> float:
        """Override to use dynamic weights"""
        il = self.current_instruction.lower()
        bonus = 0.0

        # Use dynamic target speed
        v_tgt = self.params["target_speed"]
        if "slow" in il: v_tgt *= 0.5
        if "fast" in il: v_tgt *= 1.5

        if "forward" in il:
            # 1. Alignment Reward (Gaussian bell curve around target)
            sigma = 0.35
            align = np.exp(-((vx - v_tgt) ** 2) / (2.0 * sigma ** 2))

            # 2. Ramp Reward (Linear reward for any forward movement)
            # Critical for getting "stuck" agents moving
            ramp = max(0.0, min(vx - self.params["velocity_threshold"], 1.0))

            # Combine using dynamic weights
            bonus += (self.params["forward_reward_weight"] * align) + (
                        self.params["forward_reward_weight"] * 0.5 * ramp)

            # Drift penalty (if moving sideways)
            if hasattr(self, 'forward_speed_from_obs'):
                # Assuming index 1 (y-vel) is drift for non-holonomic,
                # but for now we stick to simple logic
                pass

        elif "stop" in il or "stand" in il:
            if abs(vx) < self.params["velocity_threshold"]:
                bonus += self.params["stability_weight"] * 2.0

        return float(bonus)

    def step(self, action):
        # 1. Scale action based on aggressiveness param
        scaled_action = action * self.params["action_magnitude"]

        # 2. Run Step
        obs, reward, terminated, truncated, info = super().step(scaled_action)

        # 3. Recalculate Energy Penalty dynamically
        # Super class already applied a fixed one, we adjust it
        original_energy = 1e-3 * np.sum(np.square(action))
        new_energy = self.params["energy_penalty"] * np.sum(np.square(scaled_action))

        # Adjust reward (remove old penalty, add new)
        reward = reward + original_energy - new_energy

        return obs, reward, terminated, truncated, info