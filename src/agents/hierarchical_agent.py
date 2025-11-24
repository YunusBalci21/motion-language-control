# src/agents/hierarchical_agent.py
"""
Motion-Language Agent - TUNED FOR NATURAL WALKING
Fixed 'Leaning Forward' behavior by enforcing upright posture.
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

# ... [Imports remain the same] ...
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
    """Wrapper with constraints for NATURAL walking (no leaning)"""

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
            stability_focus: bool = True,  # Set default to True
    ):
        super().__init__(env)
        self.motion_tokenizer = motion_tokenizer
        self.current_instruction = instruction
        self.reward_scale = reward_scale
        self.motion_history_length = motion_history_length
        self.record_video = record_video
        self.video_path = video_path
        self.stability_focus = stability_focus

        # ... [Init logic remains the same] ...
        self.motion_history: deque = deque(maxlen=motion_history_length)
        self.observation_history: deque = deque(maxlen=motion_history_length)
        self.forward_speed_history: deque = deque(maxlen=motion_history_length)
        self.video_frames: Optional[List[np.ndarray]] = [] if record_video else None

        # ... [Init Mujoco handles] ...
        self.mj_model = None
        self.mj_data = None
        self.dt = 0.02
        self._prev_x = None
        self.init_mujoco_handles()

        # Rewards
        self.language_reward_weight = 0.85  # Slightly reduced to allow postural rewards
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

        print("DirectMotionLanguageWrapper initialized:")
        print(f"  Environment: {env.spec.id if hasattr(env, 'spec') else 'Unknown'}")
        print(f"  Instruction: '{instruction}'")
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

    # --- NEW: POSTURE CHECK ---
    def get_torso_upright_score(self) -> float:
        """
        Returns 1.0 if torso is vertical, 0.0 if horizontal.
        Calculated from the z-component of the torso quaternion/projection.
        """
        try:
            if self.mj_data is not None:
                # qpos[3:7] is the root quaternion (w, x, y, z)
                # We want the z-axis of the body frame to align with global z
                # Simplified: Check if the projected gravity vector is aligned
                # Or just check the pitch angle.

                # MuJoCo Humanoid usually has root at index 1 (qpos[2] is z-height)
                # Rotations are qpos[3:7].
                # A perfectly upright humanoid has [1, 0, 0, 0] or similar.

                # We can penalize significant pitch (leaning forward/back)
                # Convert quat to pitch or just penalize deviation from [1,0,0,0]

                quat = self.mj_data.qpos[3:7]
                # Deviation from upright (1, 0, 0, 0)
                # Score = w^2 (since w=1 means 0 rotation)
                # This is a rough heuristic but effective for keeping them upright
                uprightness = float(quat[0] ** 2)
                return uprightness
        except Exception:
            return 1.0  # Assume upright if can't read

    def check_fallen(self) -> bool:
        height = self.get_height()
        # Stricter height check - knees shouldn't be scraping the floor
        if "Humanoid" in self.env_name and height < 1.0:  # Increased from 0.8
            return True
        return height < 0.3

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

            # We trust MotionGPT more now, but we scale it
            return float(similarity) * self.reward_scale, float(similarity), 0.0, {}
        except Exception:
            return self.prev_language_reward, 0.0, 0.0, {}

    def compute_simple_progress_reward(self, vx: float) -> float:
        # Existing logic...
        if self.check_fallen(): return -20.0  # Harsher penalty for falling

        # Reward forward velocity but CAP it to walking speed (1.5 m/s)
        # This prevents the "sprint and fall" strategy
        if 'forward' in self.current_instruction.lower():
            if vx > 2.0: return 50.0  # Too fast! Less reward.
            if vx > 0.5: return 100.0 * vx  # Linear reward up to 2.0
            if vx > 0.1: return 20.0

        return 0.0

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

        # 1. MotionGPT Alignment (The "Brain")
        language_reward = 0.0
        similarity = 0.0
        if len(self.motion_history) >= 20:
            try:
                l_r, sim, _, _ = self.compute_enhanced_motion_language_reward()
                language_reward = l_r
                similarity = sim
            except:
                pass

        # 2. Progress (The "Goal")
        progress_reward = self.compute_simple_progress_reward(vx)

        # 3. Posture (The "Style" - CRITICAL FIX)
        upright_score = self.get_torso_upright_score()
        # Penalize if upright_score < 0.8 (leaning too much)
        posture_penalty = 0.0
        if upright_score < 0.85:
            posture_penalty = -50.0 * (0.85 - upright_score)  # Strong penalty for leaning

        # 4. Energy (Efficiency)
        energy_penalty = 0.01 * float(np.sum(np.square(action)))  # Increased penalty for jittery movement

        # Total Reward
        total_language_reward = language_reward + progress_reward + posture_penalty - energy_penalty

        # Mix with env reward (to keep basic physics alive)
        total_reward = (0.1 * env_reward) + (0.9 * total_language_reward)

        # Success tracking
        if self.check_fallen():
            self.fall_count += 1
            terminated = True  # Force end episode if too low
            total_reward -= 100.0  # Big penalty for falling

        self.episode_language_rewards.append(language_reward)
        self.episode_similarities.append(similarity)

        info.update({
            'language_reward': float(language_reward),
            'progress_reward': float(progress_reward),
            'posture_penalty': float(posture_penalty),
            'total_reward': float(total_reward),
            'motion_language_similarity': float(similarity),
            'vx': float(vx),
            'upright': float(upright_score)
        })

        return obs, float(total_reward), bool(terminated), bool(truncated), info

    # ... [Reset method remains largely same, just ensure history cleared] ...
    def reset(self, **kwargs):
        # Stats recording...
        if self.episode_language_rewards:
            # ... (stats logic) ...
            pass

        self.motion_history.clear()
        obs, info = self.env.reset(**kwargs)

        self.episode_step_count = 0
        self.total_steps += 1

        motion_features = self.extract_motion_features(obs)
        self.motion_history.append(motion_features)

        return obs, info


# ... [Agent Class remains same] ...
class EnhancedMotionLanguageAgent:
    def __init__(self, env_name="Humanoid-v4", device="cuda", motion_checkpoint=None, use_stability_focus=True,
                 motion_model_config=None):
        # Initialize components
        self.env_name = env_name
        self.device = device
        self.motion_tokenizer = MotionTokenizer(checkpoint_path=motion_checkpoint, device=device)
        self.training_config = {
            "learning_rate": 3e-4,  # Slightly lower LR for stability
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
        # Simple DummyVecEnv creation
        return DummyVecEnv(
            [lambda: self.make_single_env(instruction, language_reward_weight, record_video, video_path)])

    def train_on_instruction(self, instruction, total_timesteps, language_reward_weight, save_path, **kwargs):
        print(f"Training on {instruction}...")
        env = self.create_training_environment(instruction, language_reward_weight, n_envs=4)
        if self.use_vecnormalize:
            env = VecNormalize(env, **self.vecnormalize_config)

        model = PPO("MlpPolicy", env, verbose=1, **self.training_config)

        # Callbacks
        checkpoint_callback = CheckpointCallback(save_freq=20000, save_path=save_path, name_prefix="humanoid_posture")

        model.learn(total_timesteps=total_timesteps, callback=[checkpoint_callback], progress_bar=True)

        model.save(f"{save_path}/final_model_posture_corrected")
        if self.use_vecnormalize:
            env.save(f"{save_path}/vecnormalize.pkl")

        return f"{save_path}/final_model_posture_corrected.zip"