"""
Motion-Language Agent - MERGED VERSION WITH COMPREHENSIVE LOGGING
Combines posture enforcement with detailed step-by-step tracking
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
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize, VecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
import imageio

if os.name == "nt":
    os.environ["MUJOCO_GL"] = os.environ.get("MUJOCO_GL", "glfw")
elif "MUJOCO_GL" not in os.environ:
    os.environ["MUJOCO_GL"] = "glfw"

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root / "src"))

from models.motion_tokenizer import MotionTokenizer

try:
    from envs.humanoid_stable import ensure_wrapped_humanoid_registered, upgrade_env_name
except Exception:
    def ensure_wrapped_humanoid_registered():
        return


    def upgrade_env_name(name: str) -> str:
        return name


class DirectMotionLanguageWrapper(gym.Wrapper):
    """Enhanced wrapper with detailed logging and stability checks"""

    def __init__(
            self,
            env,
            motion_tokenizer: MotionTokenizer,
            instruction: str = "walk forward",
            reward_scale: float = 1.0,
            motion_history_length: int = 32,
            reward_aggregation: str = "weighted_recent",
            record_video: bool = False,
            video_path: Optional[str] = None,
            stability_focus: bool = True,
    ):
        super().__init__(env)
        self.motion_tokenizer = motion_tokenizer
        self.current_instruction = instruction
        self.reward_scale = reward_scale
        self.motion_history_length = motion_history_length
        self.record_video = record_video
        self.video_path = video_path
        self.stability_focus = stability_focus

        self.motion_history: deque = deque(maxlen=motion_history_length)
        self.observation_history: deque = deque(maxlen=motion_history_length)
        self.forward_speed_history: deque = deque(maxlen=motion_history_length)
        self.video_frames: Optional[List[np.ndarray]] = [] if record_video else None

        self.mj_model = None
        self.mj_data = None
        self.dt = 0.02
        self._prev_x = None
        self.init_mujoco_handles()

        # Rewards
        self.language_reward_weight = 0.85
        self.episode_count = 0
        self.total_steps = 0
        self.episode_step_count = 0
        self.fall_count = 0
        self.consecutive_stable_steps = 0
        self.prev_language_reward = 0.0
        self.current_episode_success = False
        self.episode_language_rewards = []
        self.episode_similarities = []
        self.episode_success_rates = []
        self.success_history = deque(maxlen=100)

        # Get env name
        self.env_name = getattr(env, 'spec', None)
        if self.env_name:
            self.env_name = self.env_name.id
        else:
            self.env_name = "Humanoid-v4"

        print("DirectMotionLanguageWrapper initialized:")
        print(f"  Environment: {self.env_name}")
        print(f"  Instruction: '{instruction}'")
        print("  >> DETAILED LOGGING ENABLED (every 50 steps)")
        print("  >> NATURAL POSTURE ENFORCEMENT: ACTIVE")

    def init_mujoco_handles(self) -> None:
        try:
            unwrapped = self.unwrapped
            self.mj_model = getattr(unwrapped, "model", None)
            self.mj_data = getattr(unwrapped, "data", None)
            if self.mj_model is not None and self.mj_data is not None:
                self.dt = float(self.mj_model.opt.timestep * getattr(unwrapped, "frame_skip", 1))
                self._prev_x = float(self.mj_data.qpos[0])
        except Exception:
            pass

    def get_forward_speed(self) -> float:
        try:
            if self.mj_data is None: return 0.0
            if hasattr(self.mj_data, 'qvel') and len(self.mj_data.qvel) > 0:
                return float(self.mj_data.qvel[0])
            if self._prev_x is not None:
                current_x = float(self.mj_data.qpos[0])
                vx = (current_x - self._prev_x) / self.dt
                self._prev_x = current_x
                return float(vx)
            return 0.0
        except Exception:
            return 0.0

    def get_height(self) -> float:
        try:
            if self.mj_data is not None:
                return float(self.mj_data.qpos[2])
            return 1.0
        except Exception:
            return 1.0

    def get_torso_upright_score(self) -> float:
        """
        Returns 1.0 if torso is vertical, 0.0 if horizontal.
        """
        try:
            if self.mj_data is not None:
                quat = self.mj_data.qpos[3:7]
                uprightness = float(quat[0] ** 2)
                return uprightness
        except Exception:
            return 1.0

    def check_fallen(self) -> bool:
        height = self.get_height()
        if "Humanoid" in self.env_name and height < 1.0:
            return True
        return height < 0.3

    def check_stability(self, obs: np.ndarray) -> Tuple[bool, float]:
        """Comprehensive stability check"""
        try:
            height = self.get_height()
            upright_score = self.get_torso_upright_score()
            vx = self.get_forward_speed()

            # STRICT checks
            if height < 1.0:
                return False, -20.0

            if upright_score < 0.85:
                return False, -20.0

            if abs(vx) > 2.5:
                return False, -15.0

            # Compute stability bonus
            stability_bonus = 0.0

            # Height bonus
            if height > 1.2:
                stability_bonus += min(1.0, (height - 1.0) * 0.5)

            # Uprightness bonus
            if upright_score > 0.85:
                stability_bonus += max(0.0, (upright_score - 0.8) * 0.5)

            # Velocity bonus (moving but controlled)
            if 0.1 < abs(vx) < 2.0:
                stability_bonus += 0.2

            return True, stability_bonus
        except Exception:
            return True, 0.0

    def extract_motion_features(self, obs: np.ndarray) -> np.ndarray:
        return self.motion_tokenizer.extract_motion_from_obs(obs, self.env_name)

    def compute_enhanced_motion_language_reward(self) -> Tuple[float, float, float, dict]:
        if len(self.motion_history) < 20:
            return self.prev_language_reward, 0.0, 0.0, {}

        try:
            motion_sequence = np.array(list(self.motion_history))
            motion_tensor = torch.from_numpy(motion_sequence).float()
            if motion_tensor.dim() == 2: motion_tensor = motion_tensor.unsqueeze(0)

            similarity = self.motion_tokenizer.compute_motion_language_similarity(
                motion_tensor, self.current_instruction, temporal_aggregation="mean"
            )

            return float(similarity) * self.reward_scale, float(similarity), 0.0, {}
        except Exception as e:
            return self.prev_language_reward, 0.0, 0.0, {}

    def compute_instruction_progress_bonus(self, vx: float) -> float:
        """Compute progress bonus based on instruction"""
        il = self.current_instruction.lower()
        bonus = 0.0

        # Check height first
        height = self.get_height()
        if height < 1.0:
            return -10.0

        # Stand still
        if "stop" in il or "still" in il or "stand" in il:
            if self.mj_data is not None:
                qvel = self.mj_data.qvel
                total_vel = np.linalg.norm(qvel)

                if total_vel < 0.5:
                    bonus += 10.0
                elif total_vel < 1.0:
                    bonus += 5.0
                else:
                    bonus -= 5.0

                if abs(vx) > 0.1:
                    bonus -= 10.0 * abs(vx)

                if len(self.forward_speed_history) >= 10:
                    recent_speeds = list(self.forward_speed_history)[-10:]
                    max_speed = max(abs(s) for s in recent_speeds)
                    if max_speed < 0.05:
                        bonus += 15.0

                if 1.2 < height < 1.4:
                    bonus += 3.0

                return bonus * 2.0

        # Forward movement
        elif "forward" in il:
            if vx > 2.5:
                return -5.0

            # Progressive reward for forward velocity
            if vx > 0.8:
                bonus += 100.0
            elif vx > 0.5:
                bonus += 50.0
            elif vx > 0.1:
                bonus += 20.0
            else:
                bonus -= 5.0

            # Consistency bonus
            if len(self.forward_speed_history) >= 10:
                recent = list(self.forward_speed_history)[-10:]
                speed_std = np.std(recent)
                mean_speed = np.mean(recent)

                if speed_std < 0.5 and mean_speed > 0.3:
                    bonus += 10.0

        # Backward movement
        elif "backward" in il:
            if vx < -0.5:
                bonus += 50.0
            elif vx < -0.1:
                bonus += 20.0
            else:
                bonus -= 5.0

        return float(bonus)

    def step(self, action):
        obs, env_reward, terminated, truncated, info = self.env.step(action)

        motion_state = self.extract_motion_features(obs)
        self.motion_history.append(motion_state.copy())
        self.total_steps += 1
        self.episode_step_count += 1

        vx = self.get_forward_speed()
        self.forward_speed_history.append(vx)
        self.observation_history.append(obs.copy())

        # --- REWARD CALCULATION ---

        # 1. MotionGPT Alignment
        language_reward = 0.0
        similarity = 0.0
        if len(self.motion_history) >= 20:
            try:
                l_r, sim, _, _ = self.compute_enhanced_motion_language_reward()
                language_reward = l_r
                similarity = sim
            except:
                pass

        # 2. Progress bonus
        progress_bonus = self.compute_instruction_progress_bonus(vx)

        # 3. Stability check
        is_stable, stability_bonus = self.check_stability(obs)
        if not is_stable:
            self.fall_count += 1
            self.consecutive_stable_steps = 0
            terminated = True
        else:
            self.consecutive_stable_steps += 1

        # 4. Energy penalty
        energy_penalty = 0.01 * float(np.sum(np.square(action)))

        # 5. Uprightness bonus
        upright_score = self.get_torso_upright_score()
        height = self.get_height()

        uprightness_bonus = 0.0
        if height > 1.2:
            uprightness_bonus += 1.0
        if upright_score > 0.9:
            uprightness_bonus += 1.0

        # Total Reward
        total_language_reward = (language_reward + progress_bonus +
                                 stability_bonus + uprightness_bonus - energy_penalty)

        total_reward = (0.1 * env_reward) + (0.9 * total_language_reward)

        # Success tracking
        self.episode_similarities.append(float(similarity))
        self.episode_language_rewards.append(language_reward)

        # Info dict
        info.update({
            'language_reward': float(language_reward),
            'progress_bonus': float(progress_bonus),
            'stability_bonus': float(stability_bonus),
            'uprightness_bonus': float(uprightness_bonus),
            'energy_penalty': float(energy_penalty),
            'total_language_reward': float(total_language_reward),
            'total_reward': float(total_reward),
            'motion_language_similarity': float(similarity),
            'vx': float(vx),
            'upright': float(upright_score),
            'height': float(height),
            'fall_count': int(self.fall_count),
            'consecutive_stable_steps': int(self.consecutive_stable_steps),
        })

        # === DETAILED LOGGING (every 50 steps) ===
        if self.episode_step_count % 50 == 0:
            print(f"Step {self.episode_step_count}: "
                  f"vx={vx:.3f}, "
                  f"lang_r={language_reward:.3f}, "
                  f"prog={progress_bonus:.1f}, "
                  f"sim={similarity:.3f}, "
                  f"stab={stability_bonus:.2f}, "
                  f"total={total_reward:.1f}, "
                  f"upright={upright_score:.3f}, "
                  f"h={height:.2f}")

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
        # Episode stats
        if self.episode_language_rewards:
            avg_sim = np.mean(self.episode_similarities) if self.episode_similarities else 0.0
            print(f"\n=== Episode {self.episode_count} Complete ===")
            print(f"  Length: {self.episode_step_count}")
            print(f"  Avg Similarity: {avg_sim:.3f}")
            print(f"  Falls: {self.fall_count}")
            print(f"  Max Stable Steps: {self.consecutive_stable_steps}")
            print("=" * 40 + "\n")

        self.motion_history.clear()
        obs, info = self.env.reset(**kwargs)

        self.episode_step_count = 0
        self.fall_count = 0
        self.consecutive_stable_steps = 0
        self.episode_language_rewards = []
        self.episode_similarities = []
        self.episode_count += 1

        motion_features = self.extract_motion_features(obs)
        self.motion_history.append(motion_features)

        return obs, info


class EnhancedMotionLanguageAgent:
    def __init__(self, env_name="Humanoid-v4", device="cuda", motion_checkpoint=None, use_stability_focus=True,
                 motion_model_config=None):
        self.env_name = env_name
        self.device = device
        self.motion_tokenizer = MotionTokenizer(checkpoint_path=motion_checkpoint, device=device)
        self.training_config = {
            "learning_rate": 3e-4,
            "n_steps": 2048,
            "batch_size": 64,
            "n_epochs": 10,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "ent_coef": 0.0,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
        }
        self.policy_kwargs = {}
        self.use_vecnormalize = True
        self.vecnormalize_config = {"training": True, "norm_obs": True, "norm_reward": True, "clip_reward": 10.}

    def make_single_env(self, instruction, language_reward_weight, record_video=False, video_path=None,
                        render_mode=None):
        env = gym.make(self.env_name, render_mode=render_mode)
        env = DirectMotionLanguageWrapper(
            env, self.motion_tokenizer, instruction=instruction,
            record_video=record_video, video_path=video_path
        )
        return env

    def create_training_environment(self, instruction, language_reward_weight, n_envs=1, record_video=False,
                                    video_path=None, use_multiprocessing=False):
        return DummyVecEnv(
            [lambda: self.make_single_env(instruction, language_reward_weight, record_video, video_path)])

    def train_on_instruction(self, instruction, total_timesteps, language_reward_weight, save_path, **kwargs):
        print(f"Training on {instruction}...")

        n_envs = kwargs.get('n_envs', 1)
        env = self.create_training_environment(instruction, language_reward_weight, n_envs=n_envs)

        if self.use_vecnormalize:
            env = VecNormalize(env, **self.vecnormalize_config)

        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            policy_kwargs=self.policy_kwargs,
            **self.training_config
        )

        checkpoint_callback = CheckpointCallback(save_freq=20000, save_path=save_path, name_prefix="humanoid_posture")

        model.learn(total_timesteps=total_timesteps, callback=[checkpoint_callback], progress_bar=True)

        model.save(f"{save_path}/final_model_posture_corrected")
        if self.use_vecnormalize:
            env.save(f"{save_path}/vecnormalize.pkl")

        return f"{save_path}/final_model_posture_corrected.zip"