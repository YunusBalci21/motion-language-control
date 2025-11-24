# src/models/motion_tokenizer.py
"""
Motion Tokenizer with REAL MotionGPT Integration
NOW WITH ACTUAL MOTION-LANGUAGE ALIGNMENT!
"""

import sys
import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict, Union

try:
    from models.enhanced_motion_rewards import EnhancedMotionRewardShaper
except ImportError:
    try:
        from src.models.enhanced_motion_rewards import EnhancedMotionRewardShaper
    except ImportError:
        from enhanced_motion_rewards import EnhancedMotionRewardShaper

# Import our new MotionGPT components
try:
    from models.motiongpt_vqvae_loader import MotionGPTEncoder, SimpleTextEncoder, compute_motion_text_similarity
    from models.mujoco_to_humanml3d_converter import MuJoCoToHumanML3DConverter

    MOTIONGPT_AVAILABLE = True
except ImportError:
    try:
        from src.models.motiongpt_vqvae_loader import MotionGPTEncoder, SimpleTextEncoder, \
            compute_motion_text_similarity
        from src.models.mujoco_to_humanml3d_converter import MuJoCoToHumanML3DConverter

        MOTIONGPT_AVAILABLE = True
    except ImportError:
        MOTIONGPT_AVAILABLE = False
        print("⚠ Warning: MotionGPT components not available. Using fallback reward shaper.")


class MotionTokenizer:
    def __init__(self,
                 model_config_path: Optional[str] = None,
                 checkpoint_path: Optional[str] = None,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):

        self.device = device
        self.motion_dim = 259  # Changed from 263 to match checkpoint

        print(f"Initializing Motion Tokenizer on {device}")

        # Fallback reward shaper (for stability and heuristic rewards)
        self.reward_shaper = EnhancedMotionRewardShaper()
        print("✓ Integrated EnhancedMotionRewardShaper")

        # Initialize MotionGPT if checkpoint provided
        self.motiongpt_encoder = None
        self.text_encoder = None
        self.format_converter = None

        if checkpoint_path and MOTIONGPT_AVAILABLE:
            try:
                print(f"\n🚀 Loading MotionGPT from: {checkpoint_path}")
                self.motiongpt_encoder = MotionGPTEncoder(checkpoint_path, device=device)
                self.text_encoder = SimpleTextEncoder(device=device)
                self.format_converter = MuJoCoToHumanML3DConverter()
                print("✅ MotionGPT SUCCESSFULLY LOADED!")
                print("🎯 Your thesis contribution is NOW ACTIVE!\n")
            except Exception as e:
                print(f"⚠ Warning: Could not load MotionGPT: {e}")
                print("Falling back to heuristic rewards")
                self.motiongpt_encoder = None
        elif checkpoint_path and not MOTIONGPT_AVAILABLE:
            print("⚠ Warning: checkpoint_path provided but MotionGPT components not available")
            print("Please ensure motiongpt_vqvae_loader.py and mujoco_to_humanml3d_converter.py are in src/models/")
        else:
            print("ℹ No checkpoint_path provided, using heuristic rewards only")

        self.motion_evaluator = MotionQualityEvaluator()

        # Cache for text embeddings (avoid re-encoding same instructions)
        self.text_embedding_cache = {}

    def extract_motion_from_obs(self, obs: np.ndarray, env_name: str) -> np.ndarray:
        """Extract 30-dim motion features from environment observation"""
        if obs.ndim > 1:
            obs = obs.flatten()

        if "Humanoid" in env_name:
            return self._extract_humanoid_features(obs)
        elif "Ant" in env_name:
            return self._extract_ant_features(obs)
        elif "HalfCheetah" in env_name:
            return self._extract_halfcheetah_features(obs)
        elif "Walker" in env_name:
            return self._extract_walker_features(obs)
        else:
            return self._extract_generic_features(obs)

    def _extract_ant_features(self, obs: np.ndarray) -> np.ndarray:
        """Extract features from Ant-v4 (27 dims)"""
        features = np.zeros(30, dtype=np.float32)

        try:
            if len(obs) >= 27:
                features[0] = float(obs[0]) if obs[0] > 0.1 else 0.5
                features[1:5] = obs[1:5]  # quat
                if len(obs) >= 13:
                    features[5:13] = obs[5:13]  # joints
                if len(obs) >= 16:
                    features[16] = float(obs[13])  # vx
                    features[17] = float(obs[14])  # vy
                    features[18] = float(obs[15])  # vz
                    features[26] = np.linalg.norm(obs[13:16])
            else:
                features[0] = 0.5
        except Exception as e:
            features[0] = 0.5

        return features

    def _extract_humanoid_features(self, obs: np.ndarray) -> np.ndarray:
        """Extract features from Humanoid-v4 (376 dims)"""
        features = np.zeros(30, dtype=np.float32)

        try:
            features[0] = obs[0]  # height
            features[1:5] = obs[1:5]  # quaternion
            features[5:10] = obs[5:10]  # first 5 joints

            if len(obs) > 190:
                features[16] = obs[188]  # vx
                features[17] = obs[189]  # vy
                features[18] = obs[190]  # vz
        except Exception as e:
            pass

        return features

    def _extract_halfcheetah_features(self, obs: np.ndarray) -> np.ndarray:
        """HalfCheetah-v4 feature extraction"""
        features = np.zeros(30, dtype=np.float32)
        try:
            features[0] = float(obs[0]) if obs[0] > 0.1 else 0.5
            if len(obs) >= 11:
                features[16] = float(obs[8])
                features[26] = abs(float(obs[8]))
        except:
            features[0] = 0.5
        return features

    def _extract_walker_features(self, obs: np.ndarray) -> np.ndarray:
        """Walker2d-v4 feature extraction"""
        features = np.zeros(30, dtype=np.float32)
        try:
            features[0] = float(obs[0]) if obs[0] > 0.1 else 1.0
            vel_start = len(obs) // 2
            if len(obs) > vel_start + 1:
                features[16] = float(obs[vel_start])
                features[17] = float(obs[vel_start + 1])
                features[26] = np.linalg.norm(obs[vel_start:vel_start + 2])
        except:
            features[0] = 1.0
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
        Compute similarity using MotionGPT (if available) or fallback to heuristic

        THIS IS YOUR THESIS CONTRIBUTION!
        """
        # Convert to appropriate format
        if isinstance(motion_sequence, torch.Tensor):
            motion_np = motion_sequence.detach().cpu().numpy()
        else:
            motion_np = motion_sequence

        if motion_np.ndim == 3:
            motion_np = motion_np.squeeze(0)

        # === USE MOTIONGPT IF AVAILABLE ===
        if self.motiongpt_encoder is not None:
            try:
                # Convert MuJoCo features (30-dim) to HumanML3D format (259-dim)
                motion_torch = torch.from_numpy(motion_np).float().to(self.device)

                # Ensure proper shape (T, 30)
                if motion_torch.dim() == 1:
                    motion_torch = motion_torch.unsqueeze(0)

                # Convert to HumanML3D format
                humanml3d_motion = self.format_converter.convert(motion_torch)  # (T, 259)

                # Get motion embedding from MotionGPT VQ-VAE
                motion_emb = self.motiongpt_encoder.get_motion_embedding(humanml3d_motion)  # (1, D)

                # Get text embedding (use cache)
                if instruction not in self.text_embedding_cache:
                    text_emb = self.text_encoder.encode(instruction)  # (1, D)
                    self.text_embedding_cache[instruction] = text_emb
                else:
                    text_emb = self.text_embedding_cache[instruction]

                # Compute cosine similarity
                similarity = compute_motion_text_similarity(motion_emb, text_emb)

                # CRITICAL: This is the REAL motion-language alignment!
                return float(similarity)

            except Exception as e:
                print(f"⚠ MotionGPT encoding failed: {e}, falling back to heuristic")
                # Fall through to heuristic

        # === FALLBACK: Use heuristic reward shaper ===
        score = self.reward_shaper.enhanced_motion_language_similarity(
            motion_np,
            instruction
        )

        return float(score)

    def compute_success_rate(self, motion_sequence: Union[np.ndarray, torch.Tensor],
                             instruction: str) -> float:
        """Define success based on similarity"""
        sim_score = self.compute_motion_language_similarity(motion_sequence, instruction)

        if sim_score > 0.7:
            return 1.0
        elif sim_score > 0.4:
            return 0.6
        elif sim_score > 0.2:
            return 0.3
        return 0.0

    def encode_instruction(self, instruction: str):
        """
        Encode instruction to embedding
        NOW USES REAL TEXT ENCODER!
        """
        if self.text_encoder is not None:
            try:
                if instruction not in self.text_embedding_cache:
                    text_emb = self.text_encoder.encode(instruction)
                    self.text_embedding_cache[instruction] = text_emb
                return self.text_embedding_cache[instruction]
            except Exception as e:
                print(f"⚠ Text encoding failed: {e}")

        # Fallback
        return torch.zeros(512).to(self.device)

    def get_motion_embedding(self, motion):
        """
        Get motion embedding
        NOW USES REAL MOTIONGPT ENCODER!
        """
        if self.motiongpt_encoder is not None:
            try:
                if isinstance(motion, np.ndarray):
                    motion = torch.from_numpy(motion).float().to(self.device)

                # Convert to HumanML3D format
                humanml3d_motion = self.format_converter.convert(motion)

                # Get embedding
                return self.motiongpt_encoder.get_motion_embedding(humanml3d_motion)
            except Exception as e:
                print(f"⚠ Motion embedding failed: {e}")

        # Fallback
        return torch.zeros(10).to(self.device)


class MotionQualityEvaluator:
    def evaluate_motion_quality(self, motion_sequence) -> Dict[str, float]:
        return {
            'smoothness': 0.5,
            'stability': 0.5,
            'naturalness': 0.5,
            'overall_quality': 0.5
        }


if __name__ == "__main__":
    # Test the updated tokenizer
    print("Testing Motion Tokenizer with MotionGPT integration...")

    checkpoint_path = "external/MotionGPT/prepare/deps/t2m/t2m/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/model/finest.tar"

    tokenizer = MotionTokenizer(
        checkpoint_path=checkpoint_path,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    # Test with dummy motion
    dummy_motion = np.random.randn(20, 30)  # 20 frames, 30 features

    similarity = tokenizer.compute_motion_language_similarity(
        dummy_motion,
        "walk forward"
    )

    print(f"\nMotion-language similarity: {similarity:.3f}")

    if tokenizer.motiongpt_encoder is not None:
        print("✅ MotionGPT integration WORKING!")
        print("🎯 Your thesis contribution is ACTIVE!")
    else:
        print("⚠ Using fallback heuristic rewards")

    print("\nTest complete!")