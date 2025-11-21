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
import math
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

try:
    from envs.humanoid_stable import (
        ensure_wrapped_humanoid_registered,
        upgrade_env_name,
    )
except Exception:
    def ensure_wrapped_humanoid_registered():
        return

    def upgrade_env_name(name: str) -> str:
        return name


# ==========================================================================#
#                    Enhanced DirectMotionLanguageWrapper                    #
# ==========================================================================#
class DirectMotionLanguageWrapper(gym.Wrapper):
    """
    Enhanced environment wrapper with instruction tracking, motion features,
    robust turning signal (yaw-rate), forward speed/displacement for walk/backward tasks,
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
        stability_focus: bool = False,
    ):
        super().__init__(env)
        self.motion_tokenizer = motion_tokenizer
        self.current_instruction = instruction
        self.reward_scale = reward_scale
        self.motion_history_length = motion_history_length
        self.reward_aggregation = reward_aggregation
        self.record_video = record_video
        self.video_path = video_path
        self.stability_focus = stability_focus

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
        self.total_steps = 0
        self.episode_step_count = 0
        self.episode_language_rewards: List[float] = []
        self.episode_similarities: List[float] = []
        self.episode_success_rates: List[float] = []
        self.computation_times: List[float] = []

        # Success tracking
        self.current_episode_success = False
        self.success_history: deque = deque(maxlen=100)

        # STABILITY FIXES
        self.fall_count = 0
        self.stability_bonus = 0.0
        self.consecutive_stable_steps = 0
        self.max_action_magnitude = 0.7 if stability_focus else 0.9

        # Shaping parameters
        self.progress_bonus_weight = 10
        self.speed_sigma = 0.35
        self.min_step_speed = 0.05

        self.action_filter_alpha = 0.9
        self.prev_action = None

        # Target speeds
        self._target_speeds = {
            'slow': 0.3,
            'normal': 0.5,
            'fast': 0.7
        }

        # MuJoCo handles
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
        self._prev_yaw = None
        self.init_mujoco_handles()

        self.observation_space = env.observation_space

        print("DirectMotionLanguageWrapper initialized:")
        print(f"  Environment: {self.env_name}")
        print(f"  Instruction: '{instruction}'")
        print(f"  Motion history length: {motion_history_length}")
        print(f"  Reward aggregation: {reward_aggregation}")
        print(f"  Video recording: {record_video}")
        print(f"  Stability focus: {stability_focus}")

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
                self.qpos_removed = self.nq - 2
                self.qvel_lo = self.qpos_removed
                self.qvel_hi = self.qvel_lo + self.nv
                self._prev_xy = np.array(self.mj_data.qpos[0:2], dtype=np.float64)
                self._prev_heading = None
                self._prev_yaw = None
        except Exception:
            pass

    @staticmethod
    def _quat_to_euler_xyz(qw, qx, qy, qz):
        sinr_cosp = 2 * (qw * qx + qy * qz)
        cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
        roll = math.atan2(sinr_cosp, cosr_cosp)

        sinp = 2 * (qw * qy - qz * qx)
        sinp = max(-1.0, min(1.0, sinp))
        pitch = math.asin(sinp)

        siny_cosp = 2 * (qw * qz + qx * qy)
        cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        return roll, pitch, yaw

    def get_yaw_and_rate(self):
        """Yaw from root quaternion; rate via finite-diff with wrap handling."""
        if self.mj_data is None:
            return 0.0, 0.0
        qw, qx, qy, qz = map(float, self.mj_data.qpos[3:7])
        roll, pitch, yaw = self._quat_to_euler_xyz(qw, qx, qy, qz)
        if self._prev_yaw is None:
            self._prev_yaw = yaw
            return yaw, 0.0
        dyaw = yaw - self._prev_yaw
        if dyaw > math.pi:
            dyaw -= 2 * math.pi
        if dyaw < -math.pi:
            dyaw += 2 * math.pi
        yaw_rate = dyaw / self.dt
        self._prev_yaw = yaw
        return yaw, yaw_rate

    def yaw_rate_from_obs(self, obs: np.ndarray) -> float:
        """Extract local z angular velocity (wz) from qvel using obs slices."""
        try:
            if self.mj_model is None:
                return float("nan")
            obs1 = obs if obs.ndim == 1 else obs[0]
            qvel = obs1[self.qvel_lo : self.qvel_hi]
            if len(qvel) >= 6:
                return float(qvel[5])
        except Exception:
            pass
        return float("nan")

    def fallback_yaw_rate_fd(self) -> float:
        """Finite-difference heading rate from xy velocity."""
        try:
            if self.mj_data is None:
                return 0.0
            xy = self.mj_data.qpos[0:2].copy()
            if self._prev_xy is None:
                self._prev_xy = xy
                return 0.0
            v = (xy - self._prev_xy) / self.dt
            self._prev_xy = xy
            speed = np.linalg.norm(v)
            if speed < 1e-8:
                return 0.0
            heading = float(np.arctan2(v[1], v[0]))
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
            if self.mj_data is None:
                return 0.0
            x = float(self.mj_data.qpos[0])
            if self._prev_xy is None:
                self._prev_xy = np.array([x, float(self.mj_data.qpos[1])], dtype=np.float64)
                return 0.0
            vx = (x - float(self._prev_xy[0])) / self.dt
            return float(vx)
        except Exception:
            return 0.0

    def check_stability(self, obs: np.ndarray) -> Tuple[bool, float]:
        """Strict stability check with speed control"""
        try:
            if self.mj_data is not None:
                height = float(self.mj_data.qpos[2])
                qw, qx, qy, qz = map(float, self.mj_data.qpos[3:7])
                uprightness = abs(qw)
            else:
                height = 1.0
                uprightness = 1.0

            # STRICT height check
            if height < 1.0:
                return False, -20.0

            # STRICT uprightness check
            if uprightness < 0.85:
                return False, -20.0

            # Speed check
            vx = self.forward_speed_from_obs(obs)
            if np.isnan(vx):
                vx = self.forward_speed_fd()

            if abs(vx) > 2.0:
                return False, -15.0

            # Compute stability bonus
            stability_bonus = 0.0

            if vx > 0.1:
                if height > 1.0:
                    stability_bonus += min(1.0, (height - 1.0) * 0.5)
                stability_bonus += max(0.0, (uprightness - 0.8) * 0.5)

            if abs(vx) < 0.05:
                stability_bonus -= 0.5

            yr = self.yaw_rate_from_obs(obs)
            if np.isnan(yr):
                yr = self.fallback_yaw_rate_fd()
            if abs(yr) < 2.0:
                stability_bonus += 0.1

            return True, stability_bonus
        except Exception:
            return True, 0.0

    def set_instruction(self, instruction: str) -> None:
        """Change instruction and reset tracking."""
        self.current_instruction = instruction
        self.motion_history.clear()
        self.observation_history.clear()
        self.yaw_rate_history.clear()
        self.forward_speed_history.clear()
        self.prev_language_reward = 0.0
        self.current_episode_success = False
        self.fall_count = 0
        self.consecutive_stable_steps = 0
        print(f"Instruction changed to: '{instruction}'")

    def extract_motion_features(self, obs: np.ndarray) -> np.ndarray:
        """Extract motion features using tokenizer."""
        return self.motion_tokenizer.extract_motion_from_obs(obs, self.env_name)

    def compute_enhanced_motion_language_reward(self) -> Tuple[float, float, float, dict]:
        """
        Compute enhanced motion-language reward with proper temporal handling
        Returns: (language_reward, similarity, success_rate, quality_metrics)
        """
        if len(self.motion_history) < 5:
            return 0.0, 0.0, 0.0, {}

        start_time = time.time()
        try:
            if self.reward_aggregation == "recent_window":
                motion_sequence = np.array(list(self.motion_history)[-10:])
            elif self.reward_aggregation == "full_sequence":
                motion_sequence = np.array(list(self.motion_history))
            else:
                motion_sequence = np.array(list(self.motion_history)[-10:])

            motion_tensor = torch.from_numpy(motion_sequence).float()
            if motion_tensor.dim() == 2:
                motion_tensor = motion_tensor.unsqueeze(0)

            similarity = self.motion_tokenizer.compute_motion_language_similarity(
                motion_tensor,
                self.current_instruction,
                temporal_aggregation="mean"
            )

            success_rate = self.motion_tokenizer.compute_success_rate(
                motion_tensor, self.current_instruction
            )

            quality_metrics = self.motion_tokenizer.motion_evaluator.evaluate_motion_quality(
                motion_tensor
            )

            base_reward = float(similarity) * self.reward_scale
            success_bonus = float(success_rate) * 0.5 if success_rate > 0.5 else 0.0
            quality_bonus = float(quality_metrics.get("overall_quality", 0.0)) * 0.2

            total = base_reward + success_bonus + quality_bonus
            smoothed = 0.7 * self.prev_language_reward + 0.3 * total
            self.prev_language_reward = smoothed

            self.computation_times.append(time.time() - start_time)
            return smoothed, float(similarity), float(success_rate), quality_metrics

        except Exception as e:
            print(f"Enhanced reward computation failed: {e}")
            return 0.0, 0.0, 0.0, {}

    def _target_speed_from_text(self) -> float:
        il = self.current_instruction.lower()
        if "slow" in il or "slowly" in il:
            return 0.3
        if "quick" in il or "run" in il or "fast" in il:
            return 0.7
        return 0.5

    def compute_instruction_progress_bonus(self, vx: float, yaw_rate: float) -> float:
        il = self.current_instruction.lower()
        bonus = 0.0
        sigma = self.speed_sigma

        # Check height first
        if self.mj_data is not None:
            height = float(self.mj_data.qpos[2])
            if height < 1.0:
                return -10.0

        # Stand still instruction
        if "stop" in il or "still" in il or "stand" in il:
            if self.mj_data is not None:
                qvel = self.mj_data.qvel
                total_vel = np.linalg.norm(qvel)

                if total_vel < 0.5:
                    bonus += 10.0
                elif total_vel < 1.0:
                    bonus += 5.0
                elif total_vel < 2.0:
                    bonus += 2.0
                else:
                    bonus -= 5.0

                if abs(vx) > 0.1:
                    bonus -= 10.0 * abs(vx)

                if len(self.forward_speed_history) >= 10:
                    recent_speeds = list(self.forward_speed_history)[-10:]
                    max_speed = max(abs(s) for s in recent_speeds)
                    if max_speed < 0.05:
                        bonus += 15.0
                    elif max_speed < 0.1:
                        bonus += 8.0

                if height > 1.2 and height < 1.4:
                    bonus += 3.0

                return bonus * 2.0

        # Forward movement
        elif "forward" in il:
            if vx > 2.5:
                return -5.0

            v_tgt = self._target_speed_from_text()
            align = math.exp(-((vx - v_tgt) ** 2) / (2.0 * sigma ** 2))
            ramp = max(0.0, min(vx - self.min_step_speed, 1.0))

            bonus += 2.0 * align + 1.0 * ramp

            if len(self.forward_speed_history) >= 10:
                recent = list(self.forward_speed_history)[-10:]
                speed_std = np.std(recent)
                mean_speed = np.mean(recent)

                if speed_std < 0.5 and mean_speed > 0.3:
                    bonus += 2.0 * (1.0 - speed_std)

        # Backward movement
        elif "backward" in il:
            v_tgt = -self._target_speed_from_text()
            align = math.exp(-((vx - v_tgt) ** 2) / (2.0 * sigma ** 2))
            ramp = max(0.0, -vx - self.min_step_speed)
            bonus += 1.2 * align + 0.4 * ramp

        # Turning
        elif "turn left" in il:
            bonus += 0.6 * max(0.0, yaw_rate) - 0.3 * max(0.0, -yaw_rate)
        elif "turn right" in il:
            bonus += 0.6 * max(0.0, -yaw_rate) - 0.3 * max(0.0, yaw_rate)

        return float(bonus)

    def step(self, action):
        """Execute action and return observation with language + shaping rewards."""

        # Smooth actions
        if self.prev_action is not None:
            action = self.action_filter_alpha * action + (1 - self.action_filter_alpha) * self.prev_action
        self.prev_action = action.copy()

        # Action limiting
        if self.max_action_magnitude < 1.0:
            action = np.clip(action, -self.max_action_magnitude, self.max_action_magnitude)

        # Execute
        obs, env_reward, terminated, truncated, info = self.env.step(action)

        # Update tracking
        try:
            if self.mj_data is not None:
                cur_xy = self.mj_data.qpos[0:2].copy()
                if self._prev_xy is None:
                    self._prev_xy = cur_xy
        except Exception:
            pass

        # Extract features
        motion_state = self.extract_motion_features(obs)
        self.motion_history.append(motion_state.copy())
        self.total_steps += 1
        self.episode_step_count += 1

        # Yaw rate
        _, yaw_rate = self.get_yaw_and_rate()
        self.yaw_rate_history.append(yaw_rate)

        # Forward speed
        vx = self.forward_speed_from_obs(obs)
        if np.isnan(vx):
            vx = self.forward_speed_fd()
        self.forward_speed_history.append(vx)

        self.observation_history.append(obs.copy())

        # Language reward
        language_reward = 0.0
        similarity = 0.0
        success_rate = 0.0
        quality_metrics = {}

        if len(self.motion_history) >= 2:
            try:
                lang_r, similarity, success_rate, quality_metrics = self.compute_enhanced_motion_language_reward()
                language_reward = lang_r
            except Exception as e:
                print(f"Language reward computation failed: {e}")
                language_reward = 0.0

        # Progress bonus
        progress_bonus = self.compute_instruction_progress_bonus(vx, yaw_rate)
        progress_bonus *= self.progress_bonus_weight

        # Stability check
        is_stable, stability_bonus = self.check_stability(obs)
        if not is_stable:
            self.fall_count += 1
            self.consecutive_stable_steps = 0
        else:
            self.consecutive_stable_steps += 1

        # Energy penalty
        energy_penalty = 1e-3 * float(np.sum(np.square(action)))

        # Uprightness bonus
        uprightness_bonus = 0.0
        if self.mj_data is not None:
            height = float(self.mj_data.qpos[2])
            qw = float(self.mj_data.qpos[3])

            if height > 1.2:
                uprightness_bonus += 1.0
            if abs(qw) > 0.9:
                uprightness_bonus += 1.0

        # Total reward
        total_language_reward = (language_reward + progress_bonus +
                                 stability_bonus + uprightness_bonus - energy_penalty)

        if self.language_reward_weight > 0:
            total_reward = ((1 - self.language_reward_weight) * env_reward +
                            self.language_reward_weight * total_language_reward)
        else:
            total_reward = env_reward + total_language_reward

        # Success tracking
        self.episode_similarities.append(float(similarity))
        self.episode_success_rates.append(float(success_rate))
        if success_rate > 0.6 or (("forward" in self.current_instruction.lower()) and vx > 0.8):
            self.current_episode_success = True

        self.episode_language_rewards.append(language_reward)

        # Info
        info.update({
            'language_reward': float(language_reward),
            'progress_bonus': float(progress_bonus),
            'stability_bonus': float(stability_bonus),
            'energy_penalty': float(energy_penalty),
            'total_language_reward': float(total_language_reward),
            'env_reward': float(env_reward),
            'original_reward': float(env_reward),
            'total_reward': float(total_reward),
            'instruction': self.current_instruction,
            'step': int(self.total_steps),
            'fall_count': int(self.fall_count),
            'consecutive_stable_steps': int(self.consecutive_stable_steps),
            'current_episode_success': bool(self.current_episode_success),
            'motion_language_similarity': float(similarity),
            'success_rate': float(success_rate),
            'motion_smoothness': float(quality_metrics.get('smoothness', 0.0)) if quality_metrics else 0.0,
            'motion_stability': float(quality_metrics.get('stability', 0.0)) if quality_metrics else 0.0,
            'motion_naturalness': float(quality_metrics.get('naturalness', 0.0)) if quality_metrics else 0.0,
            'motion_overall_quality': float(quality_metrics.get('overall_quality', 0.0)) if quality_metrics else 0.0,
            'vx': float(vx),
            'x_pos': float(self.mj_data.qpos[0]) if self.mj_data is not None else np.nan,
            'dt': float(self.dt),
        })

        if self.episode_step_count % 50 == 0:
            print(f"Step {self.episode_step_count}: vx={vx:.3f}, "
                  f"lang_r={language_reward:.3f}, prog={progress_bonus:.3f}, "
                  f"stab={stability_bonus:.3f}, total={total_reward:.3f}")

        # Record video
        if self.record_video and self.video_frames is not None:
            try:
                if hasattr(self.env, 'render'):
                    frame = self.env.render()
                    if frame is not None:
                        self.video_frames.append(frame)
            except Exception:
                pass

        return obs, float(total_reward), bool(terminated), bool(truncated), info

    def reset(self, **kwargs):
        """Reset with episode stats, video save, and trackers reset."""
        # Save video
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
                "episode_fall_count": int(self.fall_count),
                "episode_max_stable_steps": int(self.consecutive_stable_steps),
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

        self.fall_count = 0
        self.consecutive_stable_steps = 0
        self.stability_bonus = 0.0

        if self.record_video:
            self.video_frames = []
        self.episode_count += 1

        if self.mj_model is not None and self.mj_data is not None:
            self._prev_xy = np.array(self.mj_data.qpos[0:2], dtype=np.float64)
            self._prev_heading = None
            self._prev_yaw = None

        motion_features = self.extract_motion_features(obs)
        self.motion_history.append(motion_features)
        self.observation_history.append(obs.copy())

        info.update(
            {
                "instruction": self.current_instruction,
                "episode_start": True,
                "episode_count": self.episode_count,
                "recent_success_rate": float(np.mean(list(self.success_history))) if self.success_history else 0.0,
                **episode_stats,
            }
        )

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
        use_stability_focus: bool = True,
    ):
        if "humanoid" in env_name.lower():
            try:
                ensure_wrapped_humanoid_registered()
                env_name = upgrade_env_name(env_name)
            except Exception:
                pass

        self.env_name = env_name
        self.device = device
        self.use_stability_focus = use_stability_focus

        print("Initializing Motion-Language Agent with Stability Enhancements")
        print(f"Environment: {env_name}")
        print(f"Device: {device}")
        print(f"Stability focus: {use_stability_focus}")

        print("Loading MotionGPT tokenizer...")
        self.motion_tokenizer = MotionTokenizer(
            model_config_path=motion_model_config,
            checkpoint_path=motion_checkpoint,
            device=device,
        )

        self.test_environment_creation()

        self.rl_agent: Optional[Union[PPO, SAC]] = None

        self.training_config = {
            "learning_rate": 3e-5,
            "n_steps": 2048,
            "batch_size": 64,
            "n_epochs": 10,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.1,
            "ent_coef": 0.02,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
            "target_kl": 0.015,
            "log_std_init": -0.2,
        }

        self.use_vecnormalize = False
        self.vecnormalize_config = {
            "training": True,
            "norm_obs": True,
            "norm_reward": False,
            "clip_obs": 5.0,
            "clip_reward": np.inf,
            "gamma": 0.99,
        }

        print("Motion-Language Agent initialized successfully")

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
        """Factory for a single wrapped env."""
        if "humanoid" in self.env_name.lower():
            ensure_wrapped_humanoid_registered()

        env_kwargs = {}
        if render_mode is not None:
            env_kwargs["render_mode"] = render_mode

        if self.use_stability_focus and "Humanoid" in self.env_name:
            env_kwargs.update({'reset_noise_scale': 0.01})

        try:
            env = gym.make(self.env_name, **env_kwargs)
        except TypeError:
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
            stability_focus=self.use_stability_focus,
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
        """Create vectorized environment for training."""

        def make_env() -> Callable[[], gym.Env]:
            return lambda: self.make_single_env(
                instruction=instruction,
                language_reward_weight=language_reward_weight,
                record_video=record_video,
                video_path=video_path,
                render_mode=("rgb_array" if record_video else None),
            )

        if os.name == "nt":
            venv = DummyVecEnv([make_env() for _ in range(n_envs)])
        else:
            if n_envs == 1 or not use_multiprocessing:
                venv = DummyVecEnv([make_env()])
            else:
                venv = SubprocVecEnv([make_env() for _ in range(n_envs)])

        if self.use_vecnormalize:
            venv = VecNormalize(venv, **self.vecnormalize_config)

        return venv

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
        custom_callbacks: list = None,
    ) -> str:
        """Train PPO on an instruction with stability enhancements."""
        print(f"Training on instruction: '{instruction}' with stability enhancements")
        print(f"Language reward weight: {language_reward_weight}")
        print(f"Parallel environments: {n_envs}")
        print(f"Stability focus: {self.use_stability_focus}")

        Path(save_path).mkdir(parents=True, exist_ok=True)
        video_path = f"{save_path}/training_videos" if record_training_videos else None

        self.use_vecnormalize = use_vecnormalize

        raw_env = self.create_training_environment(
            instruction=instruction,
            language_reward_weight=language_reward_weight,
            n_envs=n_envs,
            use_multiprocessing=(n_envs > 1 and os.name != "nt"),
            record_video=record_training_videos,
            video_path=video_path,
        )

        env: VecEnv = raw_env

        policy_kwargs = dict(log_std_init=self.training_config["log_std_init"])
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
            target_kl=self.training_config["target_kl"],
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=self.device,
            tensorboard_log="./logs/",
        )

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

        print(f"Starting training for {total_timesteps} timesteps...")
        print("Using DIRECT motion-language learning with stability enhancements")

        callbacks = [checkpoint_callback, eval_callback]
        if custom_callbacks:
            callbacks.extend(custom_callbacks)

        start_time = time.time()
        self.rl_agent.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            progress_bar=True
        )
        training_time = time.time() - start_time

        final_path = f"{save_path}/final_model_{instruction.replace(' ', '_')}.zip"
        self.rl_agent.save(final_path)

        if use_vecnormalize and isinstance(env, VecNormalize):
            env.save(os.path.join(save_path, "vecnormalize.pkl"))
            print(f"Saved VecNormalize stats to: {os.path.join(save_path, 'vecnormalize.pkl')}")

        print(f"Training completed in {training_time:.2f} seconds!")
        print(f"Final model saved to: {final_path}")

        try:
            env.close()
        except Exception:
            pass
        try:
            eval_env.close()
        except Exception:
            pass

        return final_path

    def train_on_instruction_stable(
        self,
        instruction: str = "walk forward",
        total_timesteps: int = 100000,
        **kwargs
    ) -> str:
        original_stability = self.use_stability_focus
        self.use_stability_focus = True
        try:
            return self.train_on_instruction(
                instruction=instruction,
                total_timesteps=total_timesteps,
                **kwargs
            )
        finally:
            self.use_stability_focus = original_stability

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
        """Return underlying base env for rendering."""
        try:
            if isinstance(venv, VecNormalize):
                base = venv.venv
                return base.envs[0]
            return venv.envs[0]
        except Exception:
            return venv

    def run_evaluation(self, instruction, language_reward_weight, num_episodes,
                       render, deterministic, record_video, video_path, model_path=None):
        """Common evaluation logic"""
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

        self.rl_agent.policy.eval()

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
                "stability_metrics": {"fall_count": 0, "max_stable_steps": 0, "avg_stability_bonus": []},
                "vx_list": [], "x_list": [], "dt_list": []
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
                ep_data["vx_list"].append(float(info0.get("vx", np.nan)))
                ep_data["x_list"].append(float(info0.get("x_pos", np.nan)))
                ep_data["dt_list"].append(float(info0.get("dt", 0.02)))

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
            "mean_computation_time": 0.001,
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
        """Vectorized evaluation with stability metrics."""
        print(f"Evaluating on instruction: '{instruction}' with stability tracking")

        if model_path and Path(model_path).exists():
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

        return self.run_evaluation(
            instruction, language_reward_weight, num_episodes,
            render, deterministic, record_video, video_path, model_path
        )


MotionLanguageAgent = EnhancedMotionLanguageAgent


def test_enhanced_agent_with_stability():
    """Quick end-to-end smoke test with stability features."""
    print("Testing Enhanced Motion-Language Agent with Stability")
    print("=" * 55)

    agent = EnhancedMotionLanguageAgent("Humanoid-v4", use_stability_focus=True)

    print("\n1. Quick training test with stability...")
    model_path = agent.train_on_instruction_stable(
        instruction="walk forward",
        total_timesteps=5000,
        save_path="./test_checkpoints_stable/",
        record_training_videos=False,
        use_vecnormalize=False,
        n_envs=1,
        verbose=0,
        eval_freq=2000,
    )

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