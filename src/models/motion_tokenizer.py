"""
Physics-Based Motion Tokenizer
Fixed version that actually works with real movement detection
"""

import sys
import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import yaml
from typing import Optional, Tuple, Dict, Union
import warnings

# Add MotionGPT to path
project_root = Path(__file__).parent.parent.parent
motiongpt_path = project_root / "external" / "motiongpt"
sys.path.append(str(motiongpt_path))

try:
    from transformers import T5Tokenizer, T5EncoderModel, T5Config
    TRANSFORMERS_AVAILABLE = True
    print("Transformers available for language processing")
except ImportError as e:
    print(f"Transformers import failed: {e}")
    TRANSFORMERS_AVAILABLE = False


class MotionTokenizer:
    """
    Physics-based motion understanding system
    Uses actual movement data instead of broken MotionGPT fallback
    """

    def __init__(self,
                 model_config_path: Optional[str] = None,
                 checkpoint_path: Optional[str] = None,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):

        self.device = device

        # Motion tracking parameters
        self.motion_dim = 263
        self.max_sequence_length = 196

        # Language model parameters
        self.language_dim = 512

        print(f"Initializing Physics-Based Motion Tokenizer on {device}")

        # Load language model for instruction processing
        self.language_tokenizer, self.language_encoder = self._load_language_model()

        # Create physics-based motion analyzer
        self.motion_analyzer = PhysicsMotionAnalyzer()

        # Create simple but effective motion-language alignment
        self.alignment_network = self._create_simple_alignment_network()

        # Motion tracking
        self.motion_tracker = MotionTracker()

        print("Physics-Based Motion Tokenizer initialized successfully")

    def _load_language_model(self):
        """Load T5 language model for instruction encoding"""
        if not TRANSFORMERS_AVAILABLE:
            print("Transformers not available, using hash-based instruction encoding")
            return None, None

        try:
            tokenizer = T5Tokenizer.from_pretrained('t5-small')
            encoder = T5EncoderModel.from_pretrained('t5-small')
            encoder.to(self.device)
            encoder.eval()
            print("Loaded T5-small for instruction encoding")
            return tokenizer, encoder
        except Exception as e:
            print(f"Failed to load T5 model: {e}")
            return None, None

    def _create_simple_alignment_network(self):
        """Create simple motion-language alignment network"""
        # Simple network that maps motion features to language space
        alignment_net = nn.Sequential(
            nn.Linear(10, 128),  # 10 motion features -> 128
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, self.language_dim),
            nn.Tanh()
        ).to(self.device)

        print("Created simple motion-language alignment network")
        return alignment_net

    def extract_motion_from_obs(self, obs: np.ndarray, env_name: str = "Humanoid-v4") -> np.ndarray:
        """Extract motion features using physics-based analysis"""
        if env_name.startswith("Humanoid"):
            return self._extract_humanoid_physics_features(obs)
        elif env_name.startswith("HalfCheetah"):
            return self._extract_cheetah_physics_features(obs)
        elif env_name.startswith("Ant"):
            return self._extract_ant_physics_features(obs)
        else:
            return self._extract_generic_physics_features(obs)

    def _extract_humanoid_physics_features(self, obs: np.ndarray) -> np.ndarray:
        """Extract meaningful physics features from Humanoid-v4"""
        try:
            # Humanoid-v4 observation space (376 dims):
            # 0: z-coordinate of center of mass
            # 1-4: quaternion orientation of torso
            # 5-22: joint angles (17 joints)
            # 23-39: joint velocities (17 joints)
            # 40-56: next joint positions
            # etc.

            if len(obs) >= 40:
                # Extract meaningful motion features
                com_z = obs[0] if len(obs) > 0 else 0.0  # Height
                orientation = obs[1:5] if len(obs) > 4 else np.zeros(4)  # Quaternion
                joint_angles = obs[5:22] if len(obs) > 22 else np.zeros(17)
                joint_velocities = obs[23:40] if len(obs) > 40 else np.zeros(17)

                # Create physics-based motion descriptor
                motion_features = np.concatenate([
                    [com_z],  # Height (for jumping detection)
                    orientation,  # 4 dims - orientation for turning
                    joint_angles[:10],  # First 10 joint angles (most important)
                    joint_velocities[:10],  # First 10 joint velocities
                    [np.mean(np.abs(joint_velocities))],  # Overall movement magnitude
                    [np.std(joint_velocities)],  # Movement variability
                    [np.mean(joint_angles)],  # Average joint position
                    [np.std(joint_angles)]   # Joint configuration spread
                ])

                # Pad to expected size (30 features)
                if len(motion_features) < 30:
                    motion_features = np.pad(motion_features, (0, 30 - len(motion_features)))
                else:
                    motion_features = motion_features[:30]

            else:
                # Fallback for insufficient data
                motion_features = np.zeros(30)

            return motion_features.astype(np.float32)

        except Exception as e:
            print(f"Humanoid physics extraction failed: {e}")
            return np.zeros(30, dtype=np.float32)

    def _extract_cheetah_physics_features(self, obs: np.ndarray) -> np.ndarray:
        """Extract physics features from HalfCheetah"""
        try:
            if len(obs) >= 17:
                # HalfCheetah structure
                root_angle = obs[0] if len(obs) > 0 else 0.0
                joint_angles = obs[1:7] if len(obs) > 7 else np.zeros(6)
                root_vel = obs[8] if len(obs) > 8 else 0.0
                joint_vels = obs[9:15] if len(obs) > 15 else np.zeros(6)

                motion_features = np.concatenate([
                    [root_angle, root_vel],
                    joint_angles,
                    joint_vels,
                    [np.mean(np.abs(joint_vels))],  # Movement magnitude
                    [np.std(joint_vels)],  # Movement variability
                    np.zeros(12)  # Padding to 30
                ])[:30]
            else:
                motion_features = np.zeros(30)

            return motion_features.astype(np.float32)

        except Exception as e:
            print(f"Cheetah physics extraction failed: {e}")
            return np.zeros(30, dtype=np.float32)

    def _extract_ant_physics_features(self, obs: np.ndarray) -> np.ndarray:
        """Extract physics features from Ant"""
        try:
            if len(obs) >= 27:
                com_pos = obs[0:2] if len(obs) > 2 else np.zeros(2)
                orientation = obs[2:6] if len(obs) > 6 else np.zeros(4)
                joint_angles = obs[6:14] if len(obs) > 14 else np.zeros(8)
                velocities = obs[14:25] if len(obs) > 25 else np.zeros(11)

                motion_features = np.concatenate([
                    com_pos,
                    orientation,
                    joint_angles,
                    velocities[:8],  # First 8 velocities
                    [np.mean(np.abs(velocities))],
                    [np.std(velocities)],
                    np.zeros(5)  # Padding to 30
                ])[:30]
            else:
                motion_features = np.zeros(30)

            return motion_features.astype(np.float32)

        except Exception as e:
            print(f"Ant physics extraction failed: {e}")
            return np.zeros(30, dtype=np.float32)

    def _extract_generic_physics_features(self, obs: np.ndarray) -> np.ndarray:
        """Generic physics feature extraction"""
        try:
            # Take first 30 observations or pad
            if len(obs) >= 30:
                return obs[:30].astype(np.float32)
            else:
                return np.pad(obs, (0, 30 - len(obs))).astype(np.float32)
        except Exception:
            return np.zeros(30, dtype=np.float32)

    def encode_instruction(self, instruction: str) -> torch.Tensor:
        """Encode instruction using T5 or hash-based fallback"""
        if self.language_tokenizer is None or self.language_encoder is None:
            # Hash-based consistent encoding
            import hashlib
            hash_val = int(hashlib.md5(instruction.encode()).hexdigest()[:8], 16)
            np.random.seed(hash_val)
            embedding = torch.from_numpy(np.random.randn(self.language_dim)).float().to(self.device)
            np.random.seed()  # Reset
            return embedding

        try:
            instruction = instruction.lower().strip()
            inputs = self.language_tokenizer(
                instruction,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=77
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.language_encoder(**inputs)
                instruction_embedding = outputs.last_hidden_state.mean(dim=1).squeeze()

            return instruction_embedding

        except Exception as e:
            print(f"Instruction encoding failed: {e}")
            # Fallback
            import hashlib
            hash_val = int(hashlib.md5(instruction.encode()).hexdigest()[:8], 16)
            np.random.seed(hash_val)
            embedding = torch.from_numpy(np.random.randn(self.language_dim)).float().to(self.device)
            np.random.seed()
            return embedding

    def compute_motion_language_similarity(self,
                                         motion_sequence: Union[np.ndarray, torch.Tensor],
                                         instruction: str,
                                         temporal_aggregation: str = "mean") -> float:
        """
        Compute physics-based motion-language similarity
        This actually works with real movement detection!
        """
        try:
            # Convert to numpy for physics analysis
            if isinstance(motion_sequence, torch.Tensor):
                motion_np = motion_sequence.detach().cpu().numpy()
            else:
                motion_np = motion_sequence

            # Ensure correct shape
            if motion_np.ndim == 3:
                motion_np = motion_np.squeeze(0)  # Remove batch dim
            if motion_np.ndim == 1:
                motion_np = motion_np.reshape(1, -1)  # Add time dim

            # Extract physics-based motion features
            motion_descriptor = self.motion_analyzer.analyze_motion_sequence(motion_np, instruction)

            # Convert to tensor for network processing
            motion_features = torch.from_numpy(motion_descriptor).float().to(self.device)

            # Project motion to language space
            aligned_motion = self.alignment_network(motion_features)

            # Get instruction embedding
            instruction_embedding = self.encode_instruction(instruction)

            # Compute similarity
            similarity = F.cosine_similarity(
                aligned_motion.unsqueeze(0),
                instruction_embedding.unsqueeze(0),
                dim=1
            ).item()

            # Normalize to [0, 1]
            similarity = (similarity + 1) / 2

            # Add physics-based bonus for actual movement
            physics_bonus = motion_descriptor[9]  # Movement detection score
            final_similarity = 0.7 * similarity + 0.3 * physics_bonus

            return max(0.0, min(1.0, final_similarity))

        except Exception as e:
            print(f"Motion-language similarity computation failed: {e}")
            return 0.0

    def compute_success_rate(self, motion_sequence: Union[np.ndarray, torch.Tensor],
                           instruction: str) -> float:
        """Compute success rate using physics-based motion analysis"""
        try:
            # Convert to numpy
            if isinstance(motion_sequence, torch.Tensor):
                motion_np = motion_sequence.detach().cpu().numpy()
            else:
                motion_np = motion_sequence

            if motion_np.ndim == 3:
                motion_np = motion_np.squeeze(0)
            if motion_np.ndim == 1:
                motion_np = motion_np.reshape(1, -1)

            # Use physics analyzer to check task completion
            return self.motion_analyzer.check_task_completion(motion_np, instruction)

        except Exception as e:
            print(f"Success rate computation failed: {e}")
            return 0.0

    # Placeholder methods for compatibility
    def get_motion_embedding(self, motion_sequence):
        """Placeholder for compatibility"""
        if isinstance(motion_sequence, np.ndarray):
            motion_tensor = torch.from_numpy(motion_sequence).float()
        else:
            motion_tensor = motion_sequence.float()
        return motion_tensor

    def encode_motion(self, motion_sequence):
        """Placeholder for compatibility"""
        return torch.zeros(10, device=self.device)

    def decode_motion(self, token_indices):
        """Placeholder for compatibility"""
        return torch.zeros(10, 30, device=self.device)


class PhysicsMotionAnalyzer:
    """Analyzes motion using physics-based features"""

    def __init__(self):
        self.movement_threshold = 0.01
        self.orientation_threshold = 0.1

    def analyze_motion_sequence(self, motion_sequence: np.ndarray, instruction: str) -> np.ndarray:
        """
        Analyze motion sequence and create physics-based descriptor
        Returns 10-dimensional feature vector
        """
        try:
            if motion_sequence.shape[0] < 2:
                return np.zeros(10)

            # Extract key motion features (assuming 30-dim motion features)
            if motion_sequence.shape[1] >= 30:
                height_changes = np.diff(motion_sequence[:, 0])  # Z-coordinate changes
                orientation_changes = np.diff(motion_sequence[:, 1:5], axis=0)  # Quaternion changes
                joint_velocity_mag = np.mean(np.abs(motion_sequence[:, 15:25]), axis=1)  # Joint velocity magnitude
                overall_movement = np.mean(motion_sequence[:, 26])  # Overall movement magnitude
            else:
                # Fallback for shorter sequences
                height_changes = np.diff(motion_sequence[:, 0]) if motion_sequence.shape[1] > 0 else np.array([0])
                orientation_changes = np.diff(motion_sequence[:, 1:min(5, motion_sequence.shape[1])], axis=0)
                joint_velocity_mag = np.ones(motion_sequence.shape[0]) * 0.1
                overall_movement = 0.1

            # Compute physics-based features
            features = np.array([
                np.mean(height_changes),  # 0: Vertical movement
                np.std(height_changes),   # 1: Vertical movement variability
                np.mean(np.abs(height_changes)),  # 2: Vertical movement magnitude
                np.mean(np.linalg.norm(orientation_changes, axis=1)) if orientation_changes.size > 0 else 0,  # 3: Orientation change
                np.std(np.linalg.norm(orientation_changes, axis=1)) if orientation_changes.size > 0 else 0,   # 4: Orientation variability
                np.mean(joint_velocity_mag),  # 5: Average joint velocity
                np.std(joint_velocity_mag),   # 6: Joint velocity variability
                np.max(joint_velocity_mag),   # 7: Maximum joint velocity
                overall_movement,             # 8: Overall movement score
                1.0 if np.mean(joint_velocity_mag) > self.movement_threshold else 0.0  # 9: Movement detection
            ])

            return features.astype(np.float32)

        except Exception as e:
            print(f"Motion analysis failed: {e}")
            return np.zeros(10, dtype=np.float32)

    def check_task_completion(self, motion_sequence: np.ndarray, instruction: str) -> float:
        """Check if task was completed based on physics"""
        try:
            instruction_lower = instruction.lower()

            # Analyze motion
            motion_features = self.analyze_motion_sequence(motion_sequence, instruction)

            vertical_movement = motion_features[2]      # Vertical movement magnitude
            orientation_change = motion_features[3]     # Orientation change magnitude
            overall_movement = motion_features[8]       # Overall movement score
            movement_detected = motion_features[9]      # Movement detection

            # Task-specific success detection
            if 'forward' in instruction_lower or 'backward' in instruction_lower:
                # For walking tasks, check for sustained movement
                return 1.0 if (overall_movement > 0.05 and movement_detected > 0.5) else 0.0

            elif 'turn' in instruction_lower:
                # For turning tasks, check for orientation change
                return 1.0 if orientation_change > self.orientation_threshold else 0.0

            elif 'jump' in instruction_lower:
                # For jumping, check for vertical movement
                return 1.0 if vertical_movement > 0.05 else 0.0

            elif 'stop' in instruction_lower:
                # For stopping, check for low movement
                return 1.0 if overall_movement < 0.02 else 0.0

            else:
                # Generic movement task
                return 1.0 if movement_detected > 0.5 else 0.0

        except Exception as e:
            print(f"Task completion check failed: {e}")
            return 0.0


class MotionTracker:
    """Tracks motion state over time"""

    def __init__(self, history_length: int = 20):
        self.history_length = history_length
        self.reset()

    def reset(self):
        """Reset motion tracking"""
        self.motion_history = []
        self.position_history = []

    def update(self, motion_features: np.ndarray):
        """Update motion tracking"""
        self.motion_history.append(motion_features.copy())
        if len(self.motion_history) > self.history_length:
            self.motion_history.pop(0)

    def get_recent_motion(self, window: int = 10) -> np.ndarray:
        """Get recent motion window"""
        if len(self.motion_history) == 0:
            return np.zeros((window, 30))

        recent = self.motion_history[-window:] if len(self.motion_history) >= window else self.motion_history

        # Pad if necessary
        while len(recent) < window:
            recent = [recent[0]] + recent

        return np.array(recent)


class MotionQualityEvaluator:
    """Evaluate motion quality using physics metrics"""

    def __init__(self):
        pass

    def evaluate_motion_quality(self, motion_sequence: torch.Tensor) -> Dict[str, float]:
        """Evaluate motion quality using physics-based metrics"""
        try:
            if isinstance(motion_sequence, torch.Tensor):
                motion_np = motion_sequence.detach().cpu().numpy()
            else:
                motion_np = motion_sequence

            if motion_np.ndim == 3:
                motion_np = motion_np.squeeze(0)

            if motion_np.shape[0] < 3:
                return {'smoothness': 0.0, 'stability': 0.0, 'naturalness': 0.0, 'overall_quality': 0.0}

            # Physics-based quality metrics

            # Smoothness: consistency of movement
            if motion_np.shape[1] >= 10:
                velocities = np.diff(motion_np[:, :10], axis=0)
                accelerations = np.diff(velocities, axis=0)
                smoothness = 1.0 / (1.0 + np.mean(np.var(accelerations, axis=0)))
            else:
                smoothness = 0.5

            # Stability: balance and control
            if motion_np.shape[1] >= 5:
                com_variations = np.var(motion_np[:, 0])  # Height variations
                orientation_vars = np.var(motion_np[:, 1:5], axis=0)  # Orientation variations
                stability = 1.0 / (1.0 + com_variations + np.mean(orientation_vars))
            else:
                stability = 0.5

            # Naturalness: human-like movement patterns
            if motion_np.shape[1] >= 20:
                joint_coordination = np.corrcoef(motion_np[:, 5:15].T)
                naturalness = np.mean(np.abs(joint_coordination[~np.isnan(joint_coordination)]))
                naturalness = min(1.0, max(0.0, naturalness))
            else:
                naturalness = 0.5

            # Overall quality
            overall_quality = 0.4 * smoothness + 0.3 * stability + 0.3 * naturalness

            return {
                'smoothness': float(smoothness),
                'stability': float(stability),
                'naturalness': float(naturalness),
                'overall_quality': float(overall_quality)
            }

        except Exception as e:
            print(f"Motion quality evaluation failed: {e}")
            return {'smoothness': 0.0, 'stability': 0.0, 'naturalness': 0.0, 'overall_quality': 0.0}


def test_motion_tokenizer():
    """Test the physics-based motion tokenizer"""
    print("Testing Physics-Based Motion Tokenizer")
    print("=" * 40)

    # Create tokenizer
    tokenizer = MotionTokenizer()

    # Test motion analysis
    print("Testing motion analysis...")

    # Create test motion sequences
    stationary_motion = np.random.randn(20, 30) * 0.01  # Very small movements
    walking_motion = np.random.randn(20, 30) * 0.1      # Larger movements
    walking_motion[:, 8] = np.linspace(0, 1, 20)        # Add progressive movement

    instructions = [
        "walk forward",
        "stop moving",
        "turn left",
        "jump up"
    ]

    print("\nTesting motion-language alignment...")
    for instruction in instructions:
        # Test with walking motion
        similarity_walking = tokenizer.compute_motion_language_similarity(
            walking_motion, instruction
        )
        success_walking = tokenizer.compute_success_rate(walking_motion, instruction)

        # Test with stationary motion
        similarity_stationary = tokenizer.compute_motion_language_similarity(
            stationary_motion, instruction
        )
        success_stationary = tokenizer.compute_success_rate(stationary_motion, instruction)

        print(f"  '{instruction}':")
        print(f"    Walking: similarity={similarity_walking:.3f}, success={success_walking:.3f}")
        print(f"    Stationary: similarity={similarity_stationary:.3f}, success={success_stationary:.3f}")

    # Test motion quality evaluation
    print("\nTesting motion quality evaluation...")
    evaluator = MotionQualityEvaluator()
    quality_metrics = evaluator.evaluate_motion_quality(walking_motion)

    for metric, value in quality_metrics.items():
        print(f"  {metric}: {value:.3f}")

    print("\nPhysics-based motion tokenizer test completed!")


if __name__ == "__main__":
    test_motion_tokenizer()