"""
Motion-Language Agent with Fixed MotionGPT Integration and Stability Enhancements
Direct motion-language learning with proper evaluation metrics, video recording, and stability fixes
"""

import sys
import os
from pathlib import Path
from typing import Optional, Union, Callable, Tuple, List
import time
import json
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecNormalize,
    VecEnv,
)
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise
import imageio

# --------------------------------------------------------------------------
# MuJoCo GL (Windows friendly)
# --------------------------------------------------------------------------
if os.name == "nt":
    os.environ["MUJOCO_GL"] = os.environ.get("MUJOCO_GL", "glfw")
elif "MUJOCO_GL" not in os.environ:
    os.environ["MUJOCO_GL"] = "glfw"

# --------------------------------------------------------------------------
# Project imports
# --------------------------------------------------------------------------
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root / "src"))

from models.motion_tokenizer import MotionTokenizer  # noqa: E402


# ==========================================================================#
#                    Enhanced DirectMotionLanguageWrapper                    #
# ==========================================================================#
class DirectMotionLanguageWrapper(gym.Wrapper):
    """
    Enhanced environment wrapper with instruction tracking, motion features,
    robust turning signal (yaw-rate from obs with MuJoCo indexing),
    forward speed/displacement for walk/backward tasks,
    success metrics, optional video recording, and STABILITY FIXES.
    """

    def __init__(
        self,
        env,
        motion_tokenizer: MotionTokenizer,
        instruction: str = "walk forward",
        reward_scale: float = 1.0,
        motion_history_length: int = 20,
        reward_aggregation: str = "weighted_recent",
        record_video: bool = False,
        video_path: Optional[str] = None,
        stability_focus: bool = False,  # NEW: Enable stability features
    ):
        super().__init__(env)
        self.motion_tokenizer = motion_tokenizer
        self.current_instruction = instruction
        self.reward_scale = reward_scale
        self.motion_history_length = motion_history_length
        self.reward_aggregation = reward_aggregation
        self.record_video = record_video
        self.video_path = video_path
        self.stability_focus = stability_focus  # NEW

        # Motion tracking
        self.motion_history: deque = deque(maxlen=motion_history_length)
        self.observation_history: deque = deque(maxlen=motion_history_length)
        self.yaw_rate_history: deque = deque(maxlen=motion_history_length)
        self.forward_speed_history: deque = deque(maxlen=motion_history_length)

        # Video recording
        self.video_frames: Optional[List[np.ndarray]] = [] if record_video else None
        self.episode_count = 0

        # Environment info
        self.env_name = env.spec.id if hasattr(env, "spec") and env.spec else "Unknown"

        # Reward computation
        self.language_reward_weight = 0.5
        self.success_threshold = 0.6
        self.reward_smoothing = 0.9
        self.prev_language_reward = 0.0

        # Performance tracking
        self.step_count = 0
        self.total_steps = 0  # Add missing attribute
        self.episode_step_count = 0
        self.episode_language_rewards: List[float] = []
        self.episode_similarities: List[float] = []
        self.episode_success_rates: List[float] = []
        self.computation_times: List[float] = []

        # Success tracking
        self.current_episode_success = False
        self.success_history: deque = deque(maxlen=100)

        # STABILITY FIXES: Enhanced tracking
        self.fall_count = 0
        self.stability_bonus = 0.0
        self.consecutive_stable_steps = 0
        self.max_action_magnitude = 0.5 if stability_focus else 1.0

        # --- MuJoCo indexing for robust yaw/heading and forward speed signals ---
        self.mj_model = None
        self.mj_data = None
        self.dt = 0.02
        self.nq = 0
        self.nv = 0
        self.qpos_removed = 0
        self.qvel_lo = 0
        self.qvel_hi = 0
        self._prev_xy = None
        self._prev_heading = None
        self.init_mujoco_handles()

        print("DirectMotionLanguageWrapper initialized:")
        print(f"  Environment: {self.env_name}")
        print(f"  Instruction: '{instruction}'")
        print(f"  Motion history length: {motion_history_length}")
        print(f"  Reward aggregation: {reward_aggregation}")
        print(f"  Video recording: {record_video}")
        print(f"  Stability focus: {stability_focus}")  # NEW

    # -------------------- MuJoCo helpers (robust yaw/fwd) - PUBLIC METHODS -------------------- #
    def init_mujoco_handles(self) -> None:
        """Initialize MuJoCo model/data and compute observation index ranges."""
        try:
            unwrapped = self.unwrapped
            self.mj_model = getattr(unwrapped, "model", None)
            self.mj_data = getattr(unwrapped, "data", None)
            if self.mj_model is not None and self.mj_data is not None:
                self.dt = float(self.mj_model.opt.timestep * getattr(unwrapped, "frame_skip", 1))
                self.nq = int(self.mj_model.nq)
                self.nv = int(self.mj_model.nv)
                # Humanoid obs layout: [qpos[2:], qvel, cinert, cvel, qfrc_actuator, cfrc_ext]
                self.qpos_removed = self.nq - 2  # qpos[2:]
                self.qvel_lo = self.qpos_removed
                self.qvel_hi = self.qvel_lo + self.nv
                self._prev_xy = np.array(self.mj_data.qpos[0:2], dtype=np.float64)
                self._prev_heading = None
        except Exception:
            # leave defaults; we'll still have tokenizer-based features
            pass

    def yaw_rate_from_obs(self, obs: np.ndarray) -> float:
        """Extract local z angular velocity (wz) from qvel using obs slices."""
        try:
            if self.mj_model is None:
                return float("nan")
            obs1 = obs if obs.ndim == 1 else obs[0]  # handle VecEnv [1, D]
            qvel = obs1[self.qvel_lo : self.qvel_hi]
            if len(qvel) >= 6:
                return float(qvel[5])  # wz
        except Exception:
            pass
        return float("nan")

    def fallback_yaw_rate_fd(self) -> float:
        """Finite-difference heading rate from xy velocity (robust fallback)."""
        try:
            if self.mj_data is None:
                return 0.0
            xy = self.mj_data.qpos[0:2].copy()
            v = (xy - self._prev_xy) / self.dt if self._prev_xy is not None else np.array([0.0, 0.0])
            self._prev_xy = xy
            heading = float(np.arctan2(v[1], v[0])) if np.linalg.norm(v) > 1e-8 else (self._prev_heading or 0.0)
            if self._prev_heading is None:
                self._prev_heading = heading
                return 0.0
            yaw_rate = (heading - self._prev_heading) / self.dt
            self._prev_heading = heading
            return float(yaw_rate)
        except Exception:
            return 0.0

    def forward_speed_from_obs(self, obs: np.ndarray) -> float:
        """Extract forward (x) speed vx from qvel using obs slices."""
        try:
            if self.mj_model is None:
                return float("nan")
            obs1 = obs if obs.ndim == 1 else obs[0]
            qvel = obs1[self.qvel_lo : self.qvel_hi]
            return float(qvel[0]) if len(qvel) > 0 else float("nan")
        except Exception:
            return float("nan")

    def forward_speed_fd(self) -> float:
        """Fallback forward speed via finite-diff of qpos[0] (x)."""
        try:
            if self.mj_data is None or self._prev_xy is None:
                return 0.0
            x = float(self.mj_data.qpos[0])
            vx = (x - float(self._prev_xy[0])) / self.dt
            return float(vx)
        except Exception:
            return 0.0

    # STABILITY FIXES: Enhanced stability detection - PUBLIC METHOD
    def check_stability(self, obs: np.ndarray) -> Tuple[bool, float]:
        """Check if humanoid is stable and compute stability bonus."""
        try:
            # Check height (z-coordinate)
            height = self.unwrapped.data.qpos[2] if hasattr(self.unwrapped, 'data') else 1.0

            # Check if fallen
            if height < 0.8:
                return False, -10.0  # Fall penalty

            # Stability bonus based on height and orientation
            stability_bonus = 0.0

            # Height bonus
            if height > 1.0:
                stability_bonus += min(2.0, height * 0.5)

            # Orientation stability (check quaternion from obs)
            if len(obs) > 7:
                quat = obs[3:7]  # quaternion from obs
                # Check if upright (w component should be close to 1)
                if len(quat) >= 4:
                    uprightness = abs(quat[0])  # w component
                    if uprightness > 0.8:
                        stability_bonus += 0.5

            # Velocity stability (penalize high angular velocities)
            yaw_rate = self.yaw_rate_from_obs(obs)
            if not np.isnan(yaw_rate) and abs(yaw_rate) < 2.0:
                stability_bonus += 0.2

            return True, stability_bonus

        except Exception:
            return True, 0.0

    # -------------------------- Public interface -------------------------- #
    def set_instruction(self, instruction: str) -> None:
        """Change instruction and reset tracking."""
        self.current_instruction = instruction
        self.motion_history.clear()
        self.observation_history.clear()
        self.yaw_rate_history.clear()
        self.forward_speed_history.clear()
        self.prev_language_reward = 0.0
        self.current_episode_success = False
        self.fall_count = 0  # NEW
        self.consecutive_stable_steps = 0  # NEW
        print(f"Instruction changed to: '{instruction}'")

    # ------------------------ Feature extraction - PUBLIC METHOD -------------------------- #
    def extract_motion_features(self, obs: np.ndarray) -> np.ndarray:
        """Extract motion features using tokenizer."""
        return self.motion_tokenizer.extract_motion_from_obs(obs, self.env_name)

    # -------------------- Language reward & metrics - PUBLIC METHODS ----------------------- #
    def compute_enhanced_motion_language_reward(self) -> Tuple[float, float, float, dict]:
        """
        FIXED: Compute enhanced motion-language reward with proper temporal handling
        Returns: (language_reward, similarity, success_rate, quality_metrics)
        """
        if len(self.motion_history) < 5:
            return 0.0, 0.0, 0.0, {}

        start_time = time.time()
        try:
            # FIXED: Use proper temporal sequences instead of single-frame averaging
            if self.reward_aggregation == "recent_window":
                motion_sequence = np.array(list(self.motion_history)[-10:])
            elif self.reward_aggregation == "full_sequence":
                motion_sequence = np.array(list(self.motion_history))
            elif self.reward_aggregation == "weighted_recent":
                # FIXED: Keep temporal dimension instead of averaging to single frame
                motion_sequence = np.array(list(self.motion_history)[-10:])  # Keep last 10 frames
            else:
                motion_sequence = np.array(list(self.motion_history)[-10:])

            # Convert to tensor - KEEP temporal dimension
            motion_tensor = torch.from_numpy(motion_sequence).float()
            if motion_tensor.dim() == 2:
                motion_tensor = motion_tensor.unsqueeze(0)  # Add batch dim if needed

            # FIXED: Use direct similarity computation (same as debug script)
            similarity = self.motion_tokenizer.compute_motion_language_similarity(
                motion_tensor,
                self.current_instruction,
                temporal_aggregation="mean"  # Let the tokenizer handle aggregation
            )

            success_rate = self.motion_tokenizer.compute_success_rate(
                motion_tensor, self.current_instruction
            )

            quality_metrics = self.motion_tokenizer.motion_evaluator.evaluate_motion_quality(
                motion_tensor
            )

            # Base reward from similarity
            base_reward = float(similarity) * self.reward_scale

            # Success bonus
            if success_rate > 0.5:
                success_bonus = float(success_rate) * 0.5
                self.current_episode_success = True
            else:
                success_bonus = 0.0

            # Quality bonus
            quality_bonus = float(quality_metrics.get("overall_quality", 0.0)) * 0.2

            total_language_reward = base_reward + success_bonus + quality_bonus

            # Temporal smoothing (reduced to preserve signal)
            smoothed = 0.7 * self.prev_language_reward + 0.3 * total_language_reward  # Less smoothing
            self.prev_language_reward = smoothed

            self.computation_times.append(time.time() - start_time)

            return smoothed, float(similarity), float(success_rate), quality_metrics

        except Exception as e:
            print(f"Enhanced reward computation failed: {e}")
            return 0.0, 0.0, 0.0, {}

    # ----------------------- Progress bonuses (turn/walk) - PUBLIC METHOD ---------------------- #
    def compute_instruction_progress_bonus(self) -> float:
        """Instruction-specific progress bonus using yaw and forward displacement."""
        if len(self.motion_history) < 10:
            return 0.0

        try:
            recent_motion = np.array(list(self.motion_history)[-10:])
            instruction_lower = self.current_instruction.lower()

            # --- Integrated yaw change over last 10 steps ---
            yaw_change = 0.0
            if len(self.yaw_rate_history) >= 10:
                yaw_change = float(np.sum(list(self.yaw_rate_history)[-10:]) * self.dt)

            # --- Integrated forward displacement over last 10 steps ---
            fwd_disp = 0.0
            if len(self.forward_speed_history) >= 10:
                fwd_disp = float(np.sum(list(self.forward_speed_history)[-10:]) * self.dt)

            # Forward progress
            if "forward" in instruction_lower:
                return max(0.0, fwd_disp * 1.0)

            # Backward progress
            if "backward" in instruction_lower:
                return max(0.0, -fwd_disp * 1.0)

            # Turn left: positive yaw change is left
            if "turn left" in instruction_lower:
                movement_bonus = 0.0
                if recent_motion.shape[1] > 24:
                    movement_mag = float(np.mean(recent_motion[:, 24]))
                    if movement_mag > 0.02:
                        movement_bonus = 0.1
                turn_bonus = max(0.0, yaw_change * 3.0)  # slightly stronger scale
                orientation_bonus = 0.0
                if recent_motion.shape[1] >= 8:
                    quat_changes = np.diff(recent_motion[:, 4:8], axis=0)
                    orientation_activity = float(np.mean(np.abs(quat_changes)))
                    orientation_bonus = min(0.2, orientation_activity * 5.0)
                return min(0.6, turn_bonus + movement_bonus + orientation_bonus)

            # Turn right: negative yaw change
            if "turn right" in instruction_lower:
                movement_bonus = 0.0
                if recent_motion.shape[1] > 24:
                    movement_mag = float(np.mean(recent_motion[:, 24]))
                    if movement_mag > 0.02:
                        movement_bonus = 0.1
                turn_bonus = max(0.0, (-yaw_change) * 3.0)
                orientation_bonus = 0.0
                if recent_motion.shape[1] >= 8:
                    quat_changes = np.diff(recent_motion[:, 4:8], axis=0)
                    orientation_activity = float(np.mean(np.abs(quat_changes)))
                    orientation_bonus = min(0.2, orientation_activity * 5.0)
                return min(0.6, turn_bonus + movement_bonus + orientation_bonus)

            # Generic "turn"
            if "turn" in instruction_lower:
                turn_activity = float(np.mean(np.abs(list(self.yaw_rate_history)[-10:]))) if len(self.yaw_rate_history) >= 10 else 0.0
                movement_bonus = 0.0
                if recent_motion.shape[1] > 24:
                    movement_mag = float(np.mean(recent_motion[:, 24]))
                    if movement_mag > 0.02:
                        movement_bonus = 0.1
                orientation_bonus = 0.0
                if recent_motion.shape[1] >= 8:
                    quat_changes = np.diff(recent_motion[:, 4:8], axis=0)
                    orientation_activity = float(np.mean(np.abs(quat_changes)))
                    orientation_bonus = min(0.2, orientation_activity * 5.0)
                return min(0.5, turn_activity * 3.0 + movement_bonus + orientation_bonus)

            # Jump (very naive fallback)
            if "jump" in instruction_lower and recent_motion.shape[1] > 1:
                z_span = float(np.max(recent_motion[-5:, 0]) - np.min(recent_motion[-5:, 0]))
                return min(0.3, z_span * 0.2)

            # Stop / stay still
            if "stop" in instruction_lower or "still" in instruction_lower:
                movement_variance = float(np.var(recent_motion[-5:], axis=0).mean())
                return max(0.0, 0.2 - movement_variance * 10)

            return 0.0

        except Exception as e:
            print(f"Progress bonus computation failed: {e}")
            return 0.0

    def create_enhanced_observation(self, obs: np.ndarray) -> np.ndarray:
        """Create enhanced observation with CONSISTENT motion history size."""
        try:
            # FIXED: Always return same observation size for VecNormalize compatibility
            MOTION_FEATURES_SIZE = 60  # Fixed size for consistency

            # Get recent motion features
            if len(self.motion_history) > 0:
                recent_motion = np.array(list(self.motion_history)[-5:])  # Last 5 frames
                motion_features = recent_motion.flatten()

                # FIXED: Always use exactly MOTION_FEATURES_SIZE dimensions
                if len(motion_features) >= MOTION_FEATURES_SIZE:
                    motion_features = motion_features[:MOTION_FEATURES_SIZE]
                else:
                    # Pad with zeros if not enough features
                    padded = np.zeros(MOTION_FEATURES_SIZE)
                    padded[:len(motion_features)] = motion_features
                    motion_features = padded
            else:
                # FIXED: Always provide motion features, even if empty (zeros)
                motion_features = np.zeros(MOTION_FEATURES_SIZE)

            # Always combine with original observation
            enhanced_obs = np.concatenate([obs, motion_features])
            return enhanced_obs

        except Exception as e:
            print(f"Enhanced observation creation failed: {e}")
            # FIXED: Return consistent size even on error
            return np.concatenate([obs, np.zeros(60)])

    # ------------------------------ Step/Reset - FIXED ---------------------------- #
    def step(self, action):
        """Execute action and return enhanced observation with language rewards"""

        # Execute the action in base environment
        obs, env_reward, terminated, truncated, info = self.env.step(action)

        # Extract and update motion state - FIXED: Use correct method name
        motion_state = self.extract_motion_features(obs)
        self.motion_history.append(motion_state.copy())
        self.total_steps += 1
        self.episode_step_count += 1

        # Update tracking histories
        yaw_rate = self.yaw_rate_from_obs(obs)
        if not np.isnan(yaw_rate):
            self.yaw_rate_history.append(yaw_rate)

        forward_speed = self.forward_speed_from_obs(obs)
        if not np.isnan(forward_speed):
            self.forward_speed_history.append(forward_speed)

        self.observation_history.append(obs.copy())

        # Compute language reward using motion similarity
        language_reward = 0.0
        progress_bonus = 0.0

        if len(self.motion_history) >= 2:
            # Calculate language-motion alignment reward - FIXED: Use public method
            try:
                lang_reward, similarity, success_rate, quality_metrics = self.compute_enhanced_motion_language_reward()
                language_reward = lang_reward

                # Store metrics
                self.episode_similarities.append(similarity)
                self.episode_success_rates.append(success_rate)

                # Update info with metrics
                info.update({
                    'motion_language_similarity': similarity,
                    'success_rate': success_rate,
                    'motion_smoothness': quality_metrics.get('smoothness', 0.0),
                    'motion_stability': quality_metrics.get('stability', 0.0),
                    'motion_naturalness': quality_metrics.get('naturalness', 0.0),
                    'motion_overall_quality': quality_metrics.get('overall_quality', 0.0),
                })
            except Exception as e:
                print(f"Language reward computation failed: {e}")
                language_reward = 0.0

            # Add progress bonus for instruction following - FIXED: Use public method
            progress_bonus = self.compute_instruction_progress_bonus()

        # Check stability - FIXED: Use public method
        is_stable, stability_bonus = self.check_stability(obs)

        if not is_stable:
            self.fall_count += 1
            self.consecutive_stable_steps = 0
        else:
            self.consecutive_stable_steps += 1

        # Total language reward
        total_language_reward = language_reward + progress_bonus + stability_bonus

        # Combine with environment reward
        if hasattr(self, 'language_reward_weight') and self.language_reward_weight > 0:
            total_reward = ((1 - self.language_reward_weight) * env_reward +
                           self.language_reward_weight * total_language_reward)
        else:
            total_reward = env_reward + total_language_reward

        # Enhanced observation with motion history - FIXED: Use public method
        enhanced_obs = self.create_enhanced_observation(obs)

        # Store episode rewards
        self.episode_language_rewards.append(language_reward)

        # Update info with detailed metrics
        info.update({
            'language_reward': language_reward,
            'progress_bonus': progress_bonus,
            'stability_bonus': stability_bonus,
            'total_language_reward': total_language_reward,
            'env_reward': env_reward,
            'original_reward': env_reward,  # For compatibility
            'total_reward': total_reward,
            'instruction': self.current_instruction,
            'step': self.total_steps,
            'fall_count': self.fall_count,
            'consecutive_stable_steps': self.consecutive_stable_steps,
            'current_episode_success': self.current_episode_success,
        })

        # Record video frame
        if self.record_video and self.video_frames is not None:
            try:
                if hasattr(self.env, 'render'):
                    frame = self.env.render()
                    if frame is not None:
                        self.video_frames.append(frame)
            except Exception as e:
                # Silently handle video recording errors
                pass

        return enhanced_obs, float(total_reward), terminated, truncated, info

    def reset(self, **kwargs):
        """Reset with episode stats, video save, and trackers reset."""
        # Save last episode video
        if self.record_video and self.video_frames and len(self.video_frames) > 10:
            try:
                if self.video_path:
                    video_filename = f"{self.video_path}/episode_{self.episode_count}_{self.current_instruction.replace(' ', '_')}.mp4"
                    Path(video_filename).parent.mkdir(parents=True, exist_ok=True)
                    imageio.mimsave(video_filename, self.video_frames, fps=30)
                    print(f"Saved video: {video_filename}")
            except Exception as e:
                print(f"Video saving failed: {e}")

        # Episode stats
        episode_stats = {}
        if self.episode_language_rewards:
            episode_stats = {
                "episode_avg_language_reward": float(np.mean(self.episode_language_rewards)),
                "episode_avg_similarity": float(np.mean(self.episode_similarities)) if self.episode_similarities else 0.0,
                "episode_avg_success_rate": float(np.mean(self.episode_success_rates)) if self.episode_success_rates else 0.0,
                "episode_final_similarity": float(self.episode_similarities[-1]) if self.episode_similarities else 0.0,
                "episode_success": bool(self.current_episode_success),
                "episode_length": int(self.episode_step_count),
                "episode_fall_count": int(self.fall_count),  # NEW
                "episode_max_stable_steps": int(self.consecutive_stable_steps),  # NEW
            }

        self.success_history.append(bool(self.current_episode_success))

        # Reset env
        obs, info = self.env.reset(**kwargs)

        # Reset trackers
        self.motion_history.clear()
        self.observation_history.clear()
        self.yaw_rate_history.clear()
        self.forward_speed_history.clear()
        self.prev_language_reward = 0.0
        self.episode_step_count = 0
        self.current_episode_success = False

        # Reset stability tracking
        self.fall_count = 0
        self.consecutive_stable_steps = 0
        self.stability_bonus = 0.0

        # Reset video buffer
        if self.record_video:
            self.video_frames = []
        self.episode_count += 1

        # Seed yaw finite-diff tracker
        if self.mj_model is not None and self.mj_data is not None:
            self._prev_xy = np.array(self.mj_data.qpos[0:2], dtype=np.float64)
            self._prev_heading = None

        # Add initial motion
        motion_features = self.extract_motion_features(obs)
        self.motion_history.append(motion_features)
        self.observation_history.append(obs.copy())

        # Info enrich
        info.update(
            {
                "instruction": self.current_instruction,
                "episode_start": True,
                "episode_count": self.episode_count,
                "recent_success_rate": float(np.mean(list(self.success_history))) if self.success_history else 0.0,
                **episode_stats,
            }
        )

        # Reset episode arrays
        self.episode_language_rewards = []
        self.episode_similarities = []
        self.episode_success_rates = []

        return obs, info


# ==========================================================================#
#                   Enhanced Motion-Language Agent with Stability          #
# ==========================================================================#
class EnhancedMotionLanguageAgent:
    """
    Enhanced Motion-Language Agent with MotionGPT tokenizer integration,
    optional VecNormalize for training, comprehensive evaluation, and STABILITY FIXES.
    """

    def __init__(
        self,
        env_name: str = "Humanoid-v4",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        motion_model_config: Optional[str] = None,
        motion_checkpoint: Optional[str] = None,
        use_stability_focus: bool = True,  # NEW: Enable stability by default
    ):
        self.env_name = env_name
        self.device = device
        self.use_stability_focus = use_stability_focus  # NEW

        print("Initializing Motion-Language Agent with Stability Enhancements")
        print(f"Environment: {env_name}")
        print(f"Device: {device}")
        print(f"Stability focus: {use_stability_focus}")

        # Motion tokenizer (language+motion encoders)
        print("Loading MotionGPT tokenizer...")
        self.motion_tokenizer = MotionTokenizer(
            model_config_path=motion_model_config,
            checkpoint_path=motion_checkpoint,
            device=device,
        )

        # Create a quick base env to ensure MuJoCo works
        self.test_environment_creation()

        # RL agent
        self.rl_agent: Optional[Union[PPO, SAC]] = None

        # STABILITY FIXES: Enhanced PPO configuration for stability
        self.training_config = {
            "learning_rate": 1e-4,      # Slower learning for stability
            "n_steps": 4096,           # Longer rollouts
            "batch_size": 128,         # Larger batches
            "n_epochs": 10,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.1,         # Smaller clipping for stability
            "ent_coef": 0.005,         # Less exploration for stability
            "vf_coef": 0.5,
            "max_grad_norm": 0.3,      # Smaller gradients for stability
        }

        # VecNormalize settings (enhanced for stability)
        self.use_vecnormalize =  False
        self.vecnormalize_config = {
            "training": True,
            "norm_obs": True,
            "norm_reward": False,      # Don't normalize rewards for stability
            "clip_obs": 5.0,           # Reduce clipping for stability
            "clip_reward": np.inf,
            "gamma": 0.99,
        }

        print("Motion-Language Agent initialized successfully")

    def run_evaluation(self, instruction, language_reward_weight, num_episodes,
                       render, deterministic, record_video, video_path, model_path=None):
        """Common evaluation logic for both PPO and SAC"""
        # Build eval environment
        render_mode = "human" if render else "rgb_array" if record_video else None
        venv: VecEnv = DummyVecEnv(
            [
                lambda: self.make_single_env(
                    instruction=instruction,
                    language_reward_weight=language_reward_weight,
                    record_video=record_video,
                    video_path=video_path,
                    render_mode=render_mode,
                )
            ]
        )

        # Try to load normalization stats
        if model_path:
            stats_path = self.find_vecnormalize_stats(model_path)
            if stats_path and stats_path.exists():
                try:
                    venv = VecNormalize.load(str(stats_path), venv)
                    venv.training = False
                    venv.norm_reward = False
                    print(f"Loaded VecNormalize stats from: {stats_path}")
                except Exception as e:
                    print(f"Failed to load VecNormalize: {e}")

        # Make sure policy is in eval mode
        self.rl_agent.policy.eval()

        # Run evaluation episodes
        episode_results = []
        total_success_count = 0

        for ep in range(num_episodes):
            obs = venv.reset()
            if isinstance(obs, tuple):
                obs = obs[0]

            ep_data = {
                "total_reward": 0.0,
                "language_reward": 0.0,
                "env_reward": 0.0,
                "similarities": [],
                "success_rates": [],
                "motion_quality": {"smoothness": [], "stability": [], "naturalness": [], "overall_quality": []},
                "stability_metrics": {"fall_count": 0, "max_stable_steps": 0, "avg_stability_bonus": []},  # NEW
            }

            episode_success = False
            for step in range(1000):
                action, _ = self.rl_agent.predict(obs, deterministic=deterministic)
                obs, rewards, dones, infos = venv.step(action)

                r = float(rewards[0])
                info0 = infos[0] if isinstance(infos, (list, tuple)) and infos else {}

                ep_data["total_reward"] += r
                ep_data["language_reward"] += float(info0.get("language_reward", 0.0))
                ep_data["env_reward"] += float(info0.get("original_reward", 0.0))
                ep_data["similarities"].append(float(info0.get("motion_language_similarity", 0.0)))
                ep_data["success_rates"].append(float(info0.get("success_rate", 0.0)))
                ep_data["motion_quality"]["smoothness"].append(float(info0.get("motion_smoothness", 0.0)))
                ep_data["motion_quality"]["stability"].append(float(info0.get("motion_stability", 0.0)))
                ep_data["motion_quality"]["naturalness"].append(float(info0.get("motion_naturalness", 0.0)))
                ep_data["motion_quality"]["overall_quality"].append(float(info0.get("motion_overall_quality", 0.0)))

                # NEW: Track stability metrics
                ep_data["stability_metrics"]["fall_count"] = int(info0.get("fall_count", 0))
                ep_data["stability_metrics"]["max_stable_steps"] = max(
                    ep_data["stability_metrics"]["max_stable_steps"],
                    int(info0.get("consecutive_stable_steps", 0))
                )
                ep_data["stability_metrics"]["avg_stability_bonus"].append(float(info0.get("stability_bonus", 0.0)))

                if bool(info0.get("current_episode_success", False)):
                    episode_success = True

                if dones[0]:
                    break

            if episode_success:
                total_success_count += 1
            episode_results.append(ep_data)

            print(
                f"Episode {ep+1}: Total={ep_data['total_reward']:.2f}, "
                f"Language={ep_data['language_reward']:.2f}, "
                f"Similarity={np.mean(ep_data['similarities']) if ep_data['similarities'] else 0.0:.3f}, "
                f"Success={episode_success}, "
                f"Falls={ep_data['stability_metrics']['fall_count']}, "
                f"MaxStable={ep_data['stability_metrics']['max_stable_steps']}"
            )

        try:
            venv.close()
        except Exception:
            pass

        # Compute results (enhanced with stability metrics)
        results = {
            "instruction": instruction,
            "num_episodes": num_episodes,
            "mean_total_reward": float(np.mean([ep["total_reward"] for ep in episode_results])),
            "std_total_reward": float(np.std([ep["total_reward"] for ep in episode_results])),
            "mean_language_reward": float(np.mean([ep["language_reward"] for ep in episode_results])),
            "mean_env_reward": float(np.mean([ep["env_reward"] for ep in episode_results])),
            "mean_similarity": float(
                np.mean([np.mean(ep["similarities"]) for ep in episode_results if ep["similarities"]])
            ) if episode_results else 0.0,
            "mean_success_rate": float(
                np.mean([np.mean(ep["success_rates"]) for ep in episode_results if ep["success_rates"]])
            ) if episode_results else 0.0,
            "episode_success_rate": float(total_success_count / max(1, num_episodes)),
            "total_successful_episodes": int(total_success_count),
            "mean_motion_overall_quality": float(
                np.mean([
                    np.mean(ep["motion_quality"]["overall_quality"])
                    for ep in episode_results
                    if ep["motion_quality"]["overall_quality"]
                ])
            ) if episode_results else 0.0,
            # NEW: Stability metrics
            "mean_fall_count": float(np.mean([ep["stability_metrics"]["fall_count"] for ep in episode_results])),
            "mean_max_stable_steps": float(np.mean([ep["stability_metrics"]["max_stable_steps"] for ep in episode_results])),
            "mean_stability_bonus": float(
                np.mean([
                    np.mean(ep["stability_metrics"]["avg_stability_bonus"])
                    for ep in episode_results
                    if ep["stability_metrics"]["avg_stability_bonus"]
                ])
            ) if episode_results else 0.0,
            "episode_data": episode_results,
        }

        print(f"\nEvaluation Results for '{instruction}':")
        print(f"  Mean Total Reward: {results['mean_total_reward']:.2f} ± {results['std_total_reward']:.2f}")
        print(f"  Mean Language Reward: {results['mean_language_reward']:.2f}")
        print(f"  Mean Motion-Language Similarity: {results['mean_similarity']:.3f}")
        print(f"  Episode Success Rate: {results['episode_success_rate']:.1%}")
        print(f"  Mean Motion Quality: {results['mean_motion_overall_quality']:.3f}")
        print(f"  Mean Fall Count: {results['mean_fall_count']:.2f}")
        print(f"  Mean Max Stable Steps: {results['mean_max_stable_steps']:.0f}")
        print(f"  Mean Stability Bonus: {results['mean_stability_bonus']:.3f}")

        return results

    # ---------------------------- Env helpers ----------------------------- #
    def test_environment_creation(self):
        """Try main env; if it fails, fall back to simpler MuJoCo envs."""
        try:
            e = gym.make(self.env_name)
            e.close()
            print(f"✓ Environment {self.env_name} is working correctly")
        except Exception as e:
            print(f"⚠️  Warning: {self.env_name} failed to load: {e}")
            alternatives = ["HalfCheetah-v4", "Ant-v4", "Walker2d-v4", "CartPole-v1"]
            for alt in alternatives:
                try:
                    e2 = gym.make(alt)
                    e2.close()
                    print(f"✓ Alternative environment {alt} is working")
                    self.env_name = alt
                    break
                except Exception:
                    continue
            else:
                raise RuntimeError("No compatible environments available")

    def make_single_env(
        self,
        instruction: str,
        language_reward_weight: float,
        record_video: bool = False,
        video_path: Optional[str] = None,
        render_mode: Optional[str] = None,
    ) -> gym.Env:
        """Factory for a single wrapped env (with stability focus)."""
        # STABILITY FIX: Use more stable Humanoid configuration
        env_kwargs = {}
        if render_mode is not None:
            env_kwargs["render_mode"] = render_mode

        # Add stability-focused environment parameters
        if self.use_stability_focus and "Humanoid" in self.env_name:
            env_kwargs.update({
                'reset_noise_scale': 0.01,  # Reduce reset noise for stability
                'ctrl_cost_weight': 0.1,    # Penalize large control inputs
            })

        try:
            env = gym.make(self.env_name, **env_kwargs)
        except:
            # Fallback if kwargs not supported
            env = gym.make(self.env_name)

        env = Monitor(env)
        env = DirectMotionLanguageWrapper(
            env,
            self.motion_tokenizer,
            instruction=instruction,
            reward_scale=1.0,
            motion_history_length=20,
            reward_aggregation="weighted_recent",
            record_video=record_video,
            video_path=video_path,
            stability_focus=self.use_stability_focus,  # NEW: Pass stability focus
        )
        env.language_reward_weight = language_reward_weight
        return env

    def create_training_environment(
        self,
        instruction: str = "walk forward",
        language_reward_weight: float = 0.5,
        n_envs: int = 1,
        use_multiprocessing: bool = False,
        record_video: bool = False,
        video_path: Optional[str] = None,
    ) -> VecEnv:
        """Create vectorized environment for training with stability improvements."""

        def make_env() -> Callable[[], gym.Env]:
            return lambda: self.make_single_env(
                instruction=instruction,
                language_reward_weight=language_reward_weight,
                record_video=record_video,
                video_path=video_path,
                render_mode=None,
            )

        # Create vectorized environment
        if n_envs == 1:
            venv = DummyVecEnv([make_env()])
        elif use_multiprocessing:
            venv = SubprocVecEnv([make_env() for _ in range(n_envs)])
        else:
            venv = DummyVecEnv([make_env() for _ in range(n_envs)])

        # STABILITY FIX: Enhanced VecNormalize with stability
        if self.use_vecnormalize:
            venv = VecNormalize(
                venv,
                **self.vecnormalize_config
            )

        return venv

    # ------------------------------ Training ------------------------------ #
    def train_on_instruction(
        self,
        instruction: str = "walk forward",
        total_timesteps: int = 100000,
        language_reward_weight: float = 0.5,
        save_path: str = "./checkpoints/",
        eval_freq: int = 10000,
        n_envs: int = 4,
        verbose: int = 1,
        record_training_videos: bool = False,
        use_vecnormalize: bool = False,
    ) -> str:
        """
        Train PPO on an instruction with stability enhancements.
        Saves model zip and vecnormalize.pkl (if used) into save_path.
        """
        print(f"Training on instruction: '{instruction}' with stability enhancements")
        print(f"Language reward weight: {language_reward_weight}")
        print(f"Parallel environments: {n_envs}")
        print(f"Stability focus: {self.use_stability_focus}")

        Path(save_path).mkdir(parents=True, exist_ok=True)
        video_path = f"{save_path}/training_videos" if record_training_videos else None

        # Override vecnormalize setting
        self.use_vecnormalize = use_vecnormalize

        # Build env(s)
        raw_env = self.create_training_environment(
            instruction=instruction,
            language_reward_weight=language_reward_weight,
            n_envs=n_envs,
            use_multiprocessing=(n_envs > 1),
            record_video=record_training_videos,
            video_path=video_path,
        )

        env: VecEnv = raw_env

        # PPO agent with stability-focused configuration
        self.rl_agent = PPO(
            "MlpPolicy",
            env,
            learning_rate=self.training_config["learning_rate"],
            n_steps=self.training_config["n_steps"],
            batch_size=self.training_config["batch_size"],
            n_epochs=self.training_config["n_epochs"],
            gamma=self.training_config["gamma"],
            gae_lambda=self.training_config["gae_lambda"],
            clip_range=self.training_config["clip_range"],
            ent_coef=self.training_config["ent_coef"],
            vf_coef=self.training_config["vf_coef"],
            max_grad_norm=self.training_config["max_grad_norm"],
            verbose=verbose,
            device=self.device,
            tensorboard_log="./logs/",
        )

        # Callbacks
        checkpoint_callback = CheckpointCallback(
            save_freq=max(1, eval_freq // max(1, n_envs)),
            save_path=save_path,
            name_prefix=f"motion_lang_{instruction.replace(' ', '_')}",
        )

        eval_env = self.create_training_environment(
            instruction=instruction, language_reward_weight=language_reward_weight, n_envs=1
        )

        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=save_path,
            log_path=save_path,
            eval_freq=max(1, eval_freq // max(1, n_envs)),
            deterministic=True,
            render=False,
            n_eval_episodes=5,
        )

        # Train
        print(f"Starting training for {total_timesteps} timesteps...")
        print("Using DIRECT motion-language learning with stability enhancements")

        start_time = time.time()
        self.rl_agent.learn(
            total_timesteps=total_timesteps, callback=[checkpoint_callback, eval_callback], progress_bar=True
        )
        training_time = time.time() - start_time

        # Save final model
        final_path = f"{save_path}/final_model_{instruction.replace(' ', '_')}.zip"
        self.rl_agent.save(final_path)

        # Save VecNormalize stats if used
        if use_vecnormalize and isinstance(env, VecNormalize):
            env.save(os.path.join(save_path, "vecnormalize.pkl"))
            print(f"Saved VecNormalize stats to: {os.path.join(save_path, 'vecnormalize.pkl')}")

        print(f"Training completed in {training_time:.2f} seconds!")
        print(f"Final model saved to: {final_path}")

        # Close envs
        try:
            env.close()
        except Exception:
            pass
        try:
            eval_env.close()
        except Exception:
            pass

        return final_path

    # STABILITY FIX: Training with stability focus method
    def train_on_instruction_stable(
        self,
        instruction: str = "walk forward",
        total_timesteps: int = 100000,
        **kwargs
    ) -> str:
        """Train with enhanced stability (convenience method)."""

        # Force stability features
        original_stability = self.use_stability_focus
        self.use_stability_focus = True

        try:
            result = self.train_on_instruction(
                instruction=instruction,
                total_timesteps=total_timesteps,
                **kwargs
            )
            return result
        finally:
            # Restore original setting
            self.use_stability_focus = original_stability

    # ------------------------------ Evaluation --------------------------- #
    @staticmethod
    def find_vecnormalize_stats(model_path: Union[str, Path]) -> Optional[Path]:
        """Search for vecnormalize.pkl near the model zip."""
        mp = Path(model_path)
        candidates = [
            mp.with_name("vecnormalize.pkl"),
            mp.parent / "vecnormalize.pkl",
            mp.parents[1] / "vecnormalize.pkl" if len(mp.parents) > 1 else None,
        ]
        for p in candidates:
            if p and p.exists():
                return p
        return None

    @staticmethod
    def underlying_gym_env(venv: Union[VecEnv, VecNormalize]) -> gym.Env:
        """Return underlying base env for rendering (Dummy/Subproc -> Monitor -> Wrapper)."""
        try:
            if isinstance(venv, VecNormalize):
                base = venv.venv  # VecEnv
                return base.envs[0]
            return venv.envs[0]
        except Exception:
            return venv  # best effort

    def evaluate_instruction(
        self,
        instruction: str = "walk forward",
        model_path: Optional[str] = None,
        num_episodes: int = 10,
        language_reward_weight: float = 0.5,
        render: bool = False,
        deterministic: bool = True,
        record_video: bool = False,
        video_path: Optional[str] = None,
    ) -> dict:
        """
        Vectorized evaluation with stability metrics.
        Records videos from the underlying env if requested.
        """
        print(f"Evaluating on instruction: '{instruction}' with stability tracking")

        # Load PPO if provided
        if model_path and Path(model_path).exists():
            # Create a dummy VecEnv for loading; will be replaced by real eval venv
            dummy_env = DummyVecEnv(
                [
                    lambda: self.make_single_env(
                        instruction=instruction,
                        language_reward_weight=language_reward_weight,
                        record_video=False,
                        video_path=None,
                        render_mode=None,
                    )
                ]
            )
            self.rl_agent = PPO.load(model_path, env=dummy_env, device=self.device)
            dummy_env.close()
            print(f"Loaded model from: {model_path}")

        if self.rl_agent is None:
            print("No trained agent available! Train first or provide model_path.")
            return {}

        # Use common evaluation logic
        return self.run_evaluation(
            instruction, language_reward_weight, num_episodes,
            render, deterministic, record_video, video_path, model_path
        )


# Backward compatibility alias
MotionLanguageAgent = EnhancedMotionLanguageAgent


# ==========================================================================#
#                         Quick Smoke Test (Enhanced)                      #
# ==========================================================================#
def test_enhanced_agent_with_stability():
    """Quick end-to-end smoke test with stability features."""
    print("Testing Enhanced Motion-Language Agent with Stability")
    print("=" * 55)

    agent = EnhancedMotionLanguageAgent("Humanoid-v4", use_stability_focus=True)

    # 1) Quick training (very short) to ensure pipeline works
    print("\n1. Quick training test with stability...")
    model_path = agent.train_on_instruction_stable(  # Use stability method
        instruction="walk forward",
        total_timesteps=5000,
        save_path="./test_checkpoints_stable/",
        record_training_videos=False,
        use_vecnormalize=False,
        n_envs=1,
        verbose=0,
        eval_freq=2000,
    )

    # 2) Evaluation with stability metrics
    print("\n2. Enhanced evaluation test with stability metrics...")
    results = agent.evaluate_instruction(
        instruction="walk forward",
        num_episodes=2,
        model_path=model_path,
        record_video=False,
        video_path="./test_videos_stable/",
        render=False,
        deterministic=True,
    )

    print("\nStability metrics:")
    print(f"  Mean Fall Count: {results.get('mean_fall_count', 'N/A')}")
    print(f"  Mean Max Stable Steps: {results.get('mean_max_stable_steps', 'N/A')}")
    print(f"  Mean Stability Bonus: {results.get('mean_stability_bonus', 'N/A')}")

    print("\nEnhanced agent with stability test completed")
    return agent


if __name__ == "__main__":
    agent = test_enhanced_agent_with_stability()