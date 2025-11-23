import numpy as np
import gymnasium as gym


def patch_motion_tokenizer(tokenizer_cls):
    """
    Monkey-patches the MotionTokenizer to correctly handle HalfCheetah and Ant
    velocities, which were previously returning 0.0 (causing the agent to stand still).
    """

    # Save the original method to avoid infinite recursion if patched twice
    if hasattr(tokenizer_cls, '_original_extract'):
        original_extract = tokenizer_cls._original_extract
    else:
        original_extract = tokenizer_cls.extract_motion_from_obs
        tokenizer_cls._original_extract = original_extract

    def _extract_halfcheetah_features(self, obs: np.ndarray) -> np.ndarray:
        """
        HalfCheetah-v4 Observation Space (17 dims):
        [0-8]: Position-related (angles)
        [8-17]: Velocities
        Index 8 is X-velocity (Forward)
        """
        features = np.zeros(30, dtype=np.float32)

        # Standardize observation
        if obs.ndim > 1:
            obs = obs.flatten()

        # HalfCheetah doesn't have a Z-height in standard obs
        features[0] = 1.0  # Dummy stable height

        if len(obs) >= 17:
            # Linear Velocity (approximate mapping)
            features[16] = obs[8]  # vx (forward)
            features[17] = obs[9]  # vz (vertical velocity)
            features[26] = np.abs(obs[8])  # Speed magnitude

        return features

    def _extract_ant_features_robust(self, obs: np.ndarray) -> np.ndarray:
        """
        Ant-v4 Observation Space (27 dims):
        [0-12]: Positions (qpos)
        [13-26]: Velocities (qvel)
        Index 13 is X-velocity
        """
        features = np.zeros(30, dtype=np.float32)

        if obs.ndim > 1:
            obs = obs.flatten()

        features[0] = 0.75  # Standard Ant height

        if len(obs) >= 27:
            features[16] = obs[13]  # vx
            features[17] = obs[14]  # vy
            features[26] = np.linalg.norm(obs[13:15])  # speed

        return features

    def new_extract(self, obs: np.ndarray, env_name: str) -> np.ndarray:
        if "Cheetah" in env_name:
            return _extract_halfcheetah_features(self, obs)
        elif "Ant" in env_name:
            return _extract_ant_features_robust(self, obs)
        else:
            return original_extract(self, obs, env_name)

    # Apply the patch
    tokenizer_cls.extract_motion_from_obs = new_extract
    print("✓ MotionTokenizer patched for HalfCheetah and Ant velocity tracking")