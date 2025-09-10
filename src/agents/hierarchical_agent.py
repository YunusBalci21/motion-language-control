"""
Motion-Language Agent with Fixed MotionGPT Integration
Direct motion-language learning with proper evaluation metrics and video recording
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
#                         DirectMotionLanguageWrapper                        #
# ==========================================================================#
class DirectMotionLanguageWrapper(gym.Wrapper):
    """
    Enhanced environment wrapper with instruction tracking, motion features,
    robust turning signal (yaw-rate from obs with MuJoCo indexing),
    forward speed/displacement for walk/backward tasks,
    success metrics, and optional video recording.
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
    ):
        super().__init__(env)
        self.motion_tokenizer = motion_tokenizer
        self.current_instruction = instruction
        self.reward_scale = reward_scale
        self.motion_history_length = motion_history_length
        self.reward_aggregation = reward_aggregation
        self.record_video = record_video
        self.video_path = video_path

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
        self.episode_step_count = 0
        self.episode_language_rewards: List[float] = []
        self.episode_similarities: List[float] = []
        self.episode_success_rates: List[float] = []
        self.computation_times: List[float] = []

        # Success tracking
        self.current_episode_success = False
        self.success_history: deque = deque(maxlen=100)

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
        self._init_mujoco_handles()

        print("DirectMotionLanguageWrapper initialized:")
        print(f"  Environment: {self.env_name}")
        print(f"  Instruction: '{instruction}'")
        print(f"  Motion history length: {motion_history_length}")
        print(f"  Reward aggregation: {reward_aggregation}")
        print(f"  Video recording: {record_video}")

    # -------------------- MuJoCo helpers (robust yaw/fwd) -------------------- #
    def _init_mujoco_handles(self) -> None:
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

    def _yaw_rate_from_obs(self, obs: np.ndarray) -> float:
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

    def _fallback_yaw_rate_fd(self) -> float:
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

    def _forward_speed_from_obs(self, obs: np.ndarray) -> float:
        """Extract forward (x) speed vx from qvel using obs slices."""
        try:
            if self.mj_model is None:
                return float("nan")
            obs1 = obs if obs.ndim == 1 else obs[0]
            qvel = obs1[self.qvel_lo : self.qvel_hi]
            return float(qvel[0]) if len(qvel) > 0 else float("nan")
        except Exception:
            return float("nan")

    def _forward_speed_fd(self) -> float:
        """Fallback forward speed via finite-diff of qpos[0] (x)."""
        try:
            if self.mj_data is None or self._prev_xy is None:
                return 0.0
            x = float(self.mj_data.qpos[0])
            vx = (x - float(self._prev_xy[0])) / self.dt
            return float(vx)
        except Exception:
            return 0.0

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
        print(f"Instruction changed to: '{instruction}'")

    # ------------------------ Feature extraction -------------------------- #
    def _extract_motion_features(self, obs: np.ndarray) -> np.ndarray:
        """Extract motion features using tokenizer."""
        return self.motion_tokenizer.extract_motion_from_obs(obs, self.env_name)

    # -------------------- Language reward & metrics ----------------------- #
    def _compute_enhanced_motion_language_reward(self) -> Tuple[float, float, float, dict]:
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

    # ----------------------- Progress bonuses (turn/walk) ---------------------- #
    def _compute_instruction_progress_bonus(self) -> float:
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

    # ------------------------------ Step/Reset ---------------------------- #
    def step(self, action):
        """Step with feature extraction, yaw & forward signals, language reward and metrics."""
        obs, env_reward, terminated, truncated, info = self.env.step(action)

        # Track obs
        self.observation_history.append(obs.copy())

        # Yaw signals
        yaw_vel = self._yaw_rate_from_obs(obs)
        if not np.isfinite(yaw_vel):
            yaw_vel = self._fallback_yaw_rate_fd()
        self.yaw_rate_history.append(float(yaw_vel))

        # Forward speed signals
        fwd_v = self._forward_speed_from_obs(obs)
        if not np.isfinite(fwd_v):
            fwd_v = self._forward_speed_fd()
        self.forward_speed_history.append(float(fwd_v))

        # Extract motion features
        motion_features = self._extract_motion_features(obs)
        self.motion_history.append(motion_features)

        # Video
        if self.record_video and self.video_frames is not None:
            try:
                frame = self.env.render()
                if frame is not None:
                    self.video_frames.append(frame)
            except Exception as e:
                print(f"Video recording failed: {e}")

        # Rewards & metrics
        language_reward, similarity, success_rate, quality_metrics = self._compute_enhanced_motion_language_reward()
        progress_bonus = self._compute_instruction_progress_bonus()
        total_language_reward = language_reward + progress_bonus

        # Balance/upright bonus
        balance_bonus = 0.0
        if len(self.motion_history) > 0:
            recent = np.array(self.motion_history[-1])
            height = float(recent[0]) if recent.shape[0] > 0 else 0.0
            if height > 0.5:
                balance_bonus += 0.2
            roll = float(recent[3]) if recent.shape[0] > 3 else 0.0
            if abs(roll) < 0.5:
                balance_bonus += 0.1
        total_language_reward += balance_bonus

        # Combine with env reward
        if self.language_reward_weight > 0.0:
            total_reward = (1 - self.language_reward_weight) * env_reward + self.language_reward_weight * total_language_reward
        else:
            total_reward = env_reward

        # Tracking
        self.step_count += 1
        self.episode_step_count += 1
        self.episode_language_rewards.append(total_language_reward)
        self.episode_similarities.append(similarity)
        self.episode_success_rates.append(success_rate)

        # Info
        info.update(
            {
                "language_reward": float(total_language_reward),
                "motion_language_similarity": float(similarity),
                "success_rate": float(success_rate),
                "progress_bonus": float(progress_bonus),
                "balance_bonus": float(balance_bonus),
                "original_reward": float(env_reward),
                "instruction": self.current_instruction,
                "motion_history_length": len(self.motion_history),
                "step_count": self.step_count,
                "episode_step_count": self.episode_step_count,
                "avg_computation_time": float(np.mean(self.computation_times[-100:])) if self.computation_times else 0.0,
                "current_episode_success": bool(self.current_episode_success),
                # debug turn signals
                "debug_yaw_rate": float(yaw_vel),
                "debug_yaw_change_10": float(np.sum(list(self.yaw_rate_history)[-10:]) * self.dt)
                if len(self.yaw_rate_history) >= 10
                else 0.0,
                # debug forward signals
                "debug_forward_speed": float(fwd_v),
                "debug_forward_disp_10": float(np.sum(list(self.forward_speed_history)[-10:]) * self.dt)
                if len(self.forward_speed_history) >= 10
                else 0.0,
                # motion quality
                "motion_smoothness": float(quality_metrics.get("smoothness", 0.0)),
                "motion_stability": float(quality_metrics.get("stability", 0.0)),
                "motion_naturalness": float(quality_metrics.get("naturalness", 0.0)),
                "motion_overall_quality": float(quality_metrics.get("overall_quality", 0.0)),
            }
        )

        return obs, float(total_reward), terminated, truncated, info

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

        # Reset video buffer
        if self.record_video:
            self.video_frames = []
        self.episode_count += 1

        # Seed yaw finite-diff tracker
        if self.mj_model is not None:
            self._prev_xy = np.array(self.mj_data.qpos[0:2], dtype=np.float64)
            self._prev_heading = None

        # Add initial motion
        motion_features = self._extract_motion_features(obs)
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
#                       EnhancedMotionLanguageAgent                          #
# ==========================================================================#
class EnhancedMotionLanguageAgent:
    """
    Enhanced Motion-Language Agent with MotionGPT tokenizer integration,
    optional VecNormalize for training, and comprehensive evaluation.
    """

    def __init__(
        self,
        env_name: str = "Humanoid-v4",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        motion_model_config: Optional[str] = None,
        motion_checkpoint: Optional[str] = None,
    ):
        self.env_name = env_name
        self.device = device

        print("Initializing Motion-Language Agent")
        print(f"Environment: {env_name}")
        print(f"Device: {device}")

        # Motion tokenizer (language+motion encoders)
        print("Loading MotionGPT tokenizer...")
        self.motion_tokenizer = MotionTokenizer(
            model_config_path=motion_model_config,
            checkpoint_path=motion_checkpoint,
            device=device,
        )

        # Create a quick base env to ensure MuJoCo works
        self._test_environment_creation()

        # RL agent
        self.rl_agent: Optional[Union[PPO, SAC]] = None

        # PPO configuration
        self.training_config = {
            "learning_rate": 3e-4,
            "n_steps": 2048,
            "batch_size": 64,
            "n_epochs": 10,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "ent_coef": 0.01,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
        }

        print("Motion-Language Agent initialized successfully")

    def _run_evaluation(self, instruction, language_reward_weight, num_episodes,
                       render, deterministic, record_video, video_path, model_path=None):
        """Common evaluation logic for both PPO and SAC"""
        # Build eval environment
        render_mode = "human" if render else "rgb_array" if record_video else None
        venv: VecEnv = DummyVecEnv(
            [
                lambda: self._make_single_env(
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
            stats_path = self._find_vecnormalize_stats(model_path)
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
                f"Success={episode_success}"
            )

        try:
            venv.close()
        except Exception:
            pass

        # Compute results
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
            "episode_data": episode_results,
        }

        print(f"\nEvaluation Results for '{instruction}':")
        print(f"  Mean Total Reward: {results['mean_total_reward']:.2f} ± {results['std_total_reward']:.2f}")
        print(f"  Mean Language Reward: {results['mean_language_reward']:.2f}")
        print(f"  Mean Motion-Language Similarity: {results['mean_similarity']:.3f}")
        print(f"  Episode Success Rate: {results['episode_success_rate']:.1%}")
        print(f"  Mean Motion Quality: {results['mean_motion_overall_quality']:.3f}")

        return results

    # ---------------------------- Env helpers ----------------------------- #
    def _test_environment_creation(self):
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

    def _make_single_env(
        self,
        instruction: str,
        language_reward_weight: float,
        record_video: bool = False,
        video_path: Optional[str] = None,
        render_mode: Optional[str] = None,
    ) -> gym.Env:
        """Factory for a single wrapped env."""
        env_kwargs = {}
        if render_mode is not None:
            env_kwargs["render_mode"] = render_mode
        env = gym.make(self.env_name, **env_kwargs)
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
        """Create vectorized environment for training (no VecNormalize here)."""

        def make_env() -> Callable[[], gym.Env]:
            return lambda: self._make_single_env(
                instruction=instruction,
                language_reward_weight=language_reward_weight,
                record_video=record_video,
                video_path=video_path,
                render_mode=None,
            )

        if n_envs == 1:
            return DummyVecEnv([make_env()])
        if use_multiprocessing:
            return SubprocVecEnv([make_env() for _ in range(n_envs)])
        return DummyVecEnv([make_env() for _ in range(n_envs)])

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
        use_vecnormalize: bool = True,
    ) -> str:
        """
        Train PPO on an instruction. Optionally uses VecNormalize (recommended).
        Saves model zip and vecnormalize.pkl (if used) into save_path.
        """
        print(f"Training on instruction: '{instruction}'")
        print(f"Language reward weight: {language_reward_weight}")
        print(f"Parallel environments: {n_envs}")

        Path(save_path).mkdir(parents=True, exist_ok=True)
        video_path = f"{save_path}/training_videos" if record_training_videos else None

        # Build env(s)
        raw_env = self.create_training_environment(
            instruction=instruction,
            language_reward_weight=language_reward_weight,
            n_envs=n_envs,
            use_multiprocessing=(n_envs > 1),
            record_video=record_training_videos,
            video_path=video_path,
        )

        # Optional: normalize observations (highly recommended for Humanoid)
        env: VecEnv = raw_env
        if use_vecnormalize:
            env = VecNormalize(env, training=True, norm_obs=True, norm_reward=False, clip_obs=10.0)

        # PPO agent
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
        if use_vecnormalize:
            # Eval with same normalization stats (copy structure, but not weights)
            eval_env = VecNormalize(eval_env, training=False, norm_obs=True, norm_reward=False, clip_obs=10.0)
            # Sync running mean/var at the start (fresh)
            eval_env.obs_rms = env.obs_rms

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
        print("Using DIRECT motion-language learning (no pixel rendering)")

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

    # ------------------------------ Evaluation --------------------------- #
    @staticmethod
    def _find_vecnormalize_stats(model_path: Union[str, Path]) -> Optional[Path]:
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
    def _underlying_gym_env(venv: Union[VecEnv, VecNormalize]) -> gym.Env:
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
        Vectorized evaluation that will load VecNormalize stats if present.
        Records videos from the underlying env if requested.
        """
        print(f"Evaluating on instruction: '{instruction}'")

        # Load PPO if provided
        if model_path and Path(model_path).exists():
            # Create a dummy VecEnv for loading; will be replaced by real eval venv
            dummy_env = DummyVecEnv(
                [
                    lambda: self._make_single_env(
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
        return self._run_evaluation(
            instruction, language_reward_weight, num_episodes,
            render, deterministic, record_video, video_path, model_path
        )


# Backward compatibility alias
MotionLanguageAgent = EnhancedMotionLanguageAgent


# ==========================================================================#
#                              Quick Smoke Test                             #
# ==========================================================================#
def test_enhanced_agent():
    """Quick end-to-end smoke test (short run)."""
    print("Testing Enhanced Motion-Language Agent")
    print("=" * 45)

    agent = EnhancedMotionLanguageAgent("Humanoid-v4")

    # 1) Quick training (very short) to ensure pipeline works
    print("\n1. Quick training test...")
    model_path = agent.train_on_instruction(
        instruction="walk forward",
        total_timesteps=5000,
        save_path="./test_checkpoints/",
        record_training_videos=False,
        use_vecnormalize=True,  # save vecnormalize.pkl for later eval
        n_envs=1,
        verbose=0,
        eval_freq=2000,
    )

    # 2) Evaluation
    print("\n2. Enhanced evaluation test...")
    results = agent.evaluate_instruction(
        instruction="walk forward",
        num_episodes=2,
        model_path=model_path,
        record_video=False,
        video_path="./test_videos/",
        render=False,
        deterministic=True,
    )

    print("\nEnhanced agent test completed")
    return agent


if __name__ == "__main__":
    agent = test_enhanced_agent()