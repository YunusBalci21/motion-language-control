"""
Motion-Language Agent
"""

import sys
import os
from pathlib import Path
from typing import Optional, Tuple, List
import time
import json
from collections import deque

import numpy as np
import math
import torch
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback
import imageio

# Set MuJoCo rendering backend
if os.name == "nt":
    os.environ["MUJOCO_GL"] = os.environ.get("MUJOCO_GL", "glfw")
elif "MUJOCO_GL" not in os.environ:
    os.environ["MUJOCO_GL"] = "glfw"

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root / "src"))

from models.motion_tokenizer import MotionTokenizer

# Import stable humanoid environment
try:
    from envs.humanoid_stable import make_stable_humanoid
    STABLE_ENV_AVAILABLE = True
except ImportError:
    STABLE_ENV_AVAILABLE = False
    print("⚠ Warning: humanoid_stable not found, using raw Humanoid-v4")


# ============================================================================
# Utility Functions
# ============================================================================

def quat_to_uprightness(qw: float, qx: float, qy: float, qz: float) -> float:
    """
    Compute uprightness from quaternion (1.0 = perfectly upright, 0.0 = horizontal).
    This is the z-component of the rotated up vector.
    """
    return 1.0 - 2.0 * (qx * qx + qy * qy)


def quat_to_euler(qw: float, qx: float, qy: float, qz: float) -> Tuple[float, float, float]:
    """Convert quaternion to Euler angles (roll, pitch, yaw) in radians."""
    # Roll (x-axis rotation)
    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2.0 * (qw * qy - qz * qx)
    sinp = max(-1.0, min(1.0, sinp))
    pitch = math.asin(sinp)

    # Yaw (z-axis rotation)
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw


# ============================================================================
# Motion-Language Wrapper (SINGLE REWARD SOURCE)
# ============================================================================

class DirectMotionLanguageWrapper(gym.Wrapper):
    """
    Wrapper that computes motion-language rewards.
    
    THIS IS THE ONLY REWARD SOURCE - StabilityWrapper's reward shaping is disabled.
    
    Fixes applied:
    - Correct uprightness calculation
    - Sensible height thresholds (0.8 for termination)
    - Scaled progress bonuses (0-1 range, won't overwhelm similarity)
    - Partial reward during warmup (not 0)
    - Single unified reward signal
    """

    # Thresholds (must match StabilityWrapper for consistency)
    MIN_HEIGHT = 0.8
    TARGET_HEIGHT = 1.25
    STABLE_HEIGHT = 1.0
    MAX_PITCH = 1.0
    MAX_ROLL = 1.0
    
    # Warmup settings
    MIN_HISTORY_FOR_FULL_REWARD = 16
    WARMUP_REWARD = 0.3  # Partial reward during history buildup

    def __init__(
            self,
            env: gym.Env,
            motion_tokenizer: MotionTokenizer,
            instruction: str = "walk forward",
            reward_scale: float = 1.0,
            motion_history_length: int = 32,
            record_video: bool = False,
            video_path: Optional[str] = None,
    ):
        super().__init__(env)
        self.motion_tokenizer = motion_tokenizer
        self.current_instruction = instruction
        self.reward_scale = reward_scale
        self.motion_history_length = motion_history_length
        self.record_video = record_video
        self.video_path = video_path

        # History buffers
        self.motion_history: deque = deque(maxlen=motion_history_length)
        self.speed_history: deque = deque(maxlen=motion_history_length)
        self.video_frames: Optional[List[np.ndarray]] = [] if record_video else None

        # MuJoCo handles
        self.mj_model = None
        self.mj_data = None
        self.dt = 0.02
        self._init_mujoco_handles()

        # Episode tracking
        self.episode_count = 0
        self.total_steps = 0
        self.episode_step = 0
        self.episode_rewards = []
        self.episode_similarities = []

        # Get env name for feature extraction
        self.env_name = getattr(getattr(env, 'spec', None), 'id', None) or "Humanoid-v4"

        print(f"DirectMotionLanguageWrapper initialized:")
        print(f"  Environment: {self.env_name}")
        print(f"  Instruction: '{instruction}'")
        print(f"  Warmup reward: {self.WARMUP_REWARD} (until {self.MIN_HISTORY_FOR_FULL_REWARD} frames)")
        print(f"  THIS IS THE ONLY REWARD SOURCE (StabilityWrapper shaping disabled)")

    def _init_mujoco_handles(self) -> None:
        """Get MuJoCo model/data handles for direct state access."""
        try:
            unwrapped = self.unwrapped
            self.mj_model = getattr(unwrapped, "model", None)
            self.mj_data = getattr(unwrapped, "data", None)
            if self.mj_model is not None:
                self.dt = float(self.mj_model.opt.timestep * getattr(unwrapped, "frame_skip", 1))
        except Exception:
            pass

    def _get_state(self) -> dict:
        """Extract current state from MuJoCo data."""
        state = {
            'height': 1.25,
            'quat': (1.0, 0.0, 0.0, 0.0),
            'vx': 0.0, 'vy': 0.0, 'vz': 0.0,
            'uprightness': 1.0,
            'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
        }
        
        if self.mj_data is None:
            return state
            
        try:
            qpos = self.mj_data.qpos
            qvel = self.mj_data.qvel
            
            state['height'] = float(qpos[2])
            state['quat'] = tuple(float(x) for x in qpos[3:7])
            
            qw, qx, qy, qz = state['quat']
            state['uprightness'] = quat_to_uprightness(qw, qx, qy, qz)
            state['roll'], state['pitch'], state['yaw'] = quat_to_euler(qw, qx, qy, qz)
            
            # Root velocities
            if len(qvel) >= 3:
                state['vx'] = float(qvel[0])
                state['vy'] = float(qvel[1])
                state['vz'] = float(qvel[2])
                
        except Exception:
            pass
            
        return state

    def _compute_stability_reward(self, state: dict) -> float:
        """
        Compute reward for maintaining stable posture.
        Returns value in [0, 1] range.
        """
        height = state['height']
        pitch = state['pitch']
        roll = state['roll']
        
        # Height reward (Gaussian around target)
        height_error = abs(height - self.TARGET_HEIGHT)
        height_reward = math.exp(-2.0 * height_error ** 2)
        
        # Uprightness reward
        upright_reward = math.exp(-3.0 * (pitch ** 2 + roll ** 2))
        
        # Alive bonus for good posture
        alive_bonus = 0.0
        if height > self.STABLE_HEIGHT and state['uprightness'] > 0.8:
            alive_bonus = 0.2
            
        # Combine (weighted average, stays in ~[0, 1])
        stability = 0.4 * height_reward + 0.4 * upright_reward + 0.2 + alive_bonus
        
        return min(1.0, stability)

    def _compute_task_reward(self, state: dict) -> Tuple[float, dict]:
        """
        Compute task-specific reward based on instruction.
        Returns value in [0, 1] range.
        """
        vx = state['vx']
        vy = state['vy']
        
        instruction = self.current_instruction.lower()
        task_info = {'matched_task': 'unknown', 'target_speed': 0.0}
        reward = 0.0
        
        # === WALK FORWARD ===
        if any(w in instruction for w in ['forward', 'walk', 'move ahead']):
            task_info['matched_task'] = 'walk_forward'
            
            # Determine target speed
            if 'slow' in instruction:
                target_speed = 0.4
            elif 'fast' in instruction:
                target_speed = 1.2
            else:
                target_speed = 0.8
            task_info['target_speed'] = target_speed
                
            if vx > 0.1:
                # Speed matching (Gaussian around target)
                speed_error = abs(vx - target_speed)
                speed_reward = math.exp(-2.0 * speed_error ** 2)
                
                # Direction (penalize sideways drift)
                direction_reward = math.exp(-2.0 * vy ** 2)
                
                reward = 0.6 * speed_reward + 0.4 * direction_reward
            elif vx > 0:
                # Small forward movement - partial reward
                reward = 0.3 * (vx / 0.1)
            else:
                reward = 0.0
                
        # === WALK BACKWARD ===
        elif any(w in instruction for w in ['backward', 'back', 'reverse']):
            task_info['matched_task'] = 'walk_backward'
            task_info['target_speed'] = 0.6
            
            if vx < -0.1:
                speed_reward = min(-vx / 0.6, 1.0)
                direction_reward = math.exp(-2.0 * vy ** 2)
                reward = 0.6 * speed_reward + 0.4 * direction_reward
            elif vx < 0:
                reward = 0.3 * (-vx / 0.1)
                
        # === STAND STILL ===
        elif any(w in instruction for w in ['stand', 'still', 'stop', 'stay', 'balance']):
            task_info['matched_task'] = 'stand_still'
            task_info['target_speed'] = 0.0
            
            horizontal_speed = math.sqrt(vx ** 2 + vy ** 2)
            if horizontal_speed < 0.05:
                reward = 1.0
            elif horizontal_speed < 0.1:
                reward = 0.8
            elif horizontal_speed < 0.2:
                reward = 0.5
            else:
                reward = max(0, 0.3 - horizontal_speed * 0.3)
                
        # === TURN LEFT ===
        elif 'left' in instruction and 'turn' in instruction:
            task_info['matched_task'] = 'turn_left'
            reward = 0.3  # Placeholder - would need angular velocity
            
        # === TURN RIGHT ===
        elif 'right' in instruction and 'turn' in instruction:
            task_info['matched_task'] = 'turn_right'
            reward = 0.3
            
        else:
            task_info['matched_task'] = 'generic'
            if state['height'] > self.STABLE_HEIGHT:
                reward = 0.3
                
        return min(1.0, reward), task_info

    def _compute_motion_language_reward(self) -> Tuple[float, float, bool]:
        """
        Compute motion-language similarity using MotionGPT.
        
        Returns:
            (reward, raw_similarity, is_warmup)
        """
        history_len = len(self.motion_history)
        
        # FIXED: Partial reward during warmup instead of 0
        if history_len < self.MIN_HISTORY_FOR_FULL_REWARD:
            # Linearly increasing reward during warmup
            warmup_progress = history_len / self.MIN_HISTORY_FOR_FULL_REWARD
            warmup_reward = self.WARMUP_REWARD * warmup_progress
            return warmup_reward, 0.0, True
            
        try:
            motion_sequence = np.array(list(self.motion_history))
            
            similarity = self.motion_tokenizer.compute_motion_language_similarity(
                motion_sequence,
                self.current_instruction,
                temporal_aggregation="mean"
            )
            
            return float(similarity) * self.reward_scale, float(similarity), False
            
        except Exception as e:
            # On error, return warmup reward
            return self.WARMUP_REWARD, 0.0, True

    def _extract_motion_features(self, obs: np.ndarray) -> np.ndarray:
        """Extract motion features from observation."""
        return self.motion_tokenizer.extract_motion_from_obs(obs, self.env_name)

    def step(self, action):
        # Execute action (StabilityWrapper handles termination, but NOT reward shaping)
        obs, base_reward, terminated, truncated, info = self.env.step(action)
        
        self.total_steps += 1
        self.episode_step += 1
        
        # Extract state
        state = self._get_state()
        
        # Store motion features
        motion_features = self._extract_motion_features(obs)
        self.motion_history.append(motion_features)
        self.speed_history.append(state['vx'])
        
        # === COMPUTE ALL REWARD COMPONENTS ===
        
        # 1. Stability reward (posture)
        stability_reward = self._compute_stability_reward(state)
        
        # 2. Task reward (instruction heuristics)
        task_reward, task_info = self._compute_task_reward(state)
        
        # 3. Motion-language similarity (MotionGPT) - THE THESIS CONTRIBUTION
        language_reward, similarity, is_warmup = self._compute_motion_language_reward()
        
        # 4. Energy penalty
        energy_penalty = 0.001 * float(np.sum(np.square(action)))
        
        # 5. Consistency bonus
        consistency_bonus = 0.0
        if len(self.speed_history) >= 10:
            recent_speeds = list(self.speed_history)[-10:]
            speed_std = np.std(recent_speeds)
            mean_speed = np.mean(recent_speeds)
            
            if speed_std < 0.3 and mean_speed > 0.2:
                consistency_bonus = 0.2
        
        # === COMBINE INTO SINGLE REWARD ===
        # Stability gates the task/language rewards
        stability_gate = min(1.0, max(0.3, stability_reward))
        
        if is_warmup:
            # During warmup: emphasize stability and basic task
            total_reward = (
                0.5 * stability_reward +
                0.3 * task_reward * stability_gate +
                0.2 * language_reward  # Partial warmup reward
            )
        else:
            # After warmup: full reward structure with MotionGPT
            total_reward = (
                0.25 * stability_reward +              # Stay upright
                0.15 * task_reward * stability_gate +  # Follow instruction
                0.50 * language_reward * stability_gate +  # MotionGPT (main signal!)
                0.10 * consistency_bonus -             # Smooth movement
                energy_penalty                         # Efficiency
            )
        
        # Scale to reasonable range for PPO
        total_reward = total_reward * 2.0
        
        # Apply termination penalty (since StabilityWrapper's is disabled)
        if terminated and info.get('termination_reason'):
            total_reward = -5.0
        
        # Track for logging
        self.episode_rewards.append(total_reward)
        self.episode_similarities.append(similarity)
        
        # === INFO DICT ===
        info.update({
            'stability_reward': float(stability_reward),
            'task_reward': float(task_reward),
            'language_reward': float(language_reward),
            'motion_similarity': float(similarity),
            'consistency_bonus': float(consistency_bonus),
            'energy_penalty': float(energy_penalty),
            'total_reward': float(total_reward),
            'is_warmup': is_warmup,
            'height': state['height'],
            'vx': state['vx'],
            'vy': state['vy'],
            'uprightness': state['uprightness'],
            'pitch': state['pitch'],
            'roll': state['roll'],
            'matched_task': task_info['matched_task'],
            'target_speed': task_info['target_speed'],
            'episode_step': self.episode_step,
            'history_len': len(self.motion_history),
        })
        
        # === LOGGING ===
        if self.episode_step % 100 == 0:
            avg_sim = np.mean(self.episode_similarities[-100:]) if self.episode_similarities else 0
            warmup_str = " [WARMUP]" if is_warmup else ""
            print(f"  Step {self.episode_step}{warmup_str}: "
                  f"h={state['height']:.2f}, "
                  f"vx={state['vx']:.2f}, "
                  f"sim={similarity:.3f}, "
                  f"avg={avg_sim:.3f}, "
                  f"r={total_reward:.2f}")
        
        # === VIDEO ===
        if self.record_video and self.video_frames is not None:
            try:
                frame = self.env.render()
                if frame is not None:
                    self.video_frames.append(frame)
            except Exception:
                pass
        
        return obs, float(total_reward), bool(terminated), bool(truncated), info

    def reset(self, **kwargs):
        # Log episode summary
        if self.episode_rewards:
            avg_reward = np.mean(self.episode_rewards)
            avg_sim = np.mean(self.episode_similarities) if self.episode_similarities else 0
            max_sim = max(self.episode_similarities) if self.episode_similarities else 0
            print(f"\n{'='*50}")
            print(f"Episode {self.episode_count} Complete")
            print(f"  Steps: {self.episode_step}")
            print(f"  Avg Reward: {avg_reward:.2f}")
            print(f"  Avg Similarity: {avg_sim:.3f}")
            print(f"  Max Similarity: {max_sim:.3f}")
            print(f"{'='*50}\n")
        
        # Save video
        if self.record_video and self.video_frames and self.video_path:
            try:
                video_file = f"{self.video_path}/episode_{self.episode_count}.mp4"
                imageio.mimsave(video_file, self.video_frames, fps=30)
                print(f"Saved video: {video_file}")
            except Exception as e:
                print(f"Failed to save video: {e}")
            self.video_frames = []
        
        # Reset
        obs, info = self.env.reset(**kwargs)
        
        self.motion_history.clear()
        self.speed_history.clear()
        self.episode_rewards = []
        self.episode_similarities = []
        self.episode_step = 0
        self.episode_count += 1
        
        motion_features = self._extract_motion_features(obs)
        self.motion_history.append(motion_features)
        
        print(f"Starting Episode {self.episode_count}: '{self.current_instruction}'")
        
        return obs, info

    def set_instruction(self, instruction: str):
        """Change the current instruction."""
        self.current_instruction = instruction
        print(f"Instruction changed to: '{instruction}'")


# ============================================================================
# Main Agent Class
# ============================================================================

class EnhancedMotionLanguageAgent:
    """
    Main agent for motion-language control training.
    
    Uses stable humanoid with PD control but DISABLED reward shaping
    to avoid conflicting with DirectMotionLanguageWrapper.
    """
    
    def __init__(
            self,
            env_name: str = "Humanoid-v4",
            device: str = "cuda",
            motion_checkpoint: Optional[str] = None,
            use_stability_focus: bool = True,
    ):
        self.env_name = env_name
        self.device = device if torch.cuda.is_available() else "cpu"
        self.use_stability_focus = use_stability_focus
        
        # Initialize motion tokenizer
        self.motion_tokenizer = MotionTokenizer(
            checkpoint_path=motion_checkpoint,
            device=self.device
        )
        
        # PPO hyperparameters
        self.training_config = {
            "learning_rate": 1e-4,
            "n_steps": 2048,
            "batch_size": 64,
            "n_epochs": 5,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.1,
            "ent_coef": 0.005,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
            "target_kl": 0.03,
        }
        
        self.policy_kwargs = {
            "net_arch": dict(pi=[256, 256], vf=[256, 256]),
            "activation_fn": torch.nn.Tanh,
        }
        
        self.use_vecnormalize = True
        self.vecnormalize_config = {
            "training": True,
            "norm_obs": True,
            "norm_reward": True,
            "clip_reward": 10.0,
            "gamma": 0.99,
        }
        
        print(f"EnhancedMotionLanguageAgent initialized:")
        print(f"  Device: {self.device}")
        print(f"  Stable env available: {STABLE_ENV_AVAILABLE}")

    def _make_env(
            self,
            instruction: str,
            record_video: bool = False,
            video_path: Optional[str] = None,
            render_mode: Optional[str] = None,
    ) -> gym.Env:
        """Create environment with SINGLE reward source."""
        
        if STABLE_ENV_AVAILABLE and "Humanoid" in self.env_name:
            # Use stable humanoid with reward_shaping=False
            # This gives us PD control + physics tuning + termination
            # but NO conflicting reward signal
            env = make_stable_humanoid(
                max_episode_steps=1000,
                action_scale=0.4,
                action_smoothing=0.7,
                target_height=1.25,
                min_height=0.8,
                initial_noise=0.005,
                reward_shaping=False,  # CRITICAL: Disable to avoid conflict!
            )
            print("Using make_stable_humanoid(reward_shaping=False)")
        else:
            env = gym.make(self.env_name, render_mode=render_mode)
            print(f"Using raw {self.env_name}")
        
        # Add motion-language wrapper (ONLY reward source)
        env = DirectMotionLanguageWrapper(
            env,
            motion_tokenizer=self.motion_tokenizer,
            instruction=instruction,
            record_video=record_video,
            video_path=video_path,
        )
        
        return env

    def _make_vec_env(
            self,
            instruction: str,
            n_envs: int = 1,
            record_video: bool = False,
            video_path: Optional[str] = None,
    ):
        """Create vectorized environment."""
        
        def make_env_fn():
            return self._make_env(instruction, record_video, video_path)
        
        vec_env = DummyVecEnv([make_env_fn for _ in range(n_envs)])
        
        if self.use_vecnormalize:
            vec_env = VecNormalize(vec_env, **self.vecnormalize_config)
            
        return vec_env

    def train_on_instruction(
            self,
            instruction: str,
            total_timesteps: int = 500000,
            n_envs: int = 1,
            save_path: str = "./checkpoints",
            record_training_videos: bool = False,
    ) -> str:
        """Train on a specific instruction."""
        
        print(f"\n{'='*60}")
        print(f"TRAINING: '{instruction}'")
        print(f"Timesteps: {total_timesteps:,}")
        print(f"Parallel envs: {n_envs}")
        print(f"Reward source: DirectMotionLanguageWrapper ONLY")
        print(f"{'='*60}\n")
        
        os.makedirs(save_path, exist_ok=True)
        
        video_path = f"{save_path}/videos" if record_training_videos else None
        if video_path:
            os.makedirs(video_path, exist_ok=True)
            
        env = self._make_vec_env(
            instruction=instruction,
            n_envs=n_envs,
            record_video=record_training_videos,
            video_path=video_path,
        )
        
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log=f"{save_path}/tensorboard",
            policy_kwargs=self.policy_kwargs,
            **self.training_config,
        )
        
        checkpoint_callback = CheckpointCallback(
            save_freq=max(20000 // n_envs, 1000),
            save_path=save_path,
            name_prefix="motion_language",
        )
        
        try:
            model.learn(
                total_timesteps=total_timesteps,
                callback=[checkpoint_callback],
                progress_bar=True,
            )
        except KeyboardInterrupt:
            print("\nTraining interrupted")
        
        final_path = f"{save_path}/final_model"
        model.save(final_path)
        
        if self.use_vecnormalize:
            env.save(f"{save_path}/vecnormalize.pkl")
        
        env.close()
        
        print(f"\nModel saved: {final_path}.zip")
        return f"{final_path}.zip"

    def train_on_instruction_stable(
            self,
            instruction: str,
            total_timesteps: int = 300000,
            n_envs: int = 4,
            save_path: str = "./checkpoints",
            record_training_videos: bool = False,
    ) -> str:
        """Convenience method matching original API."""
        return self.train_on_instruction(
            instruction=instruction,
            total_timesteps=total_timesteps,
            n_envs=n_envs,
            save_path=save_path,
            record_training_videos=record_training_videos,
        )

    def evaluate(
            self,
            model_path: str,
            instruction: str,
            n_episodes: int = 5,
            record_video: bool = True,
            video_path: str = "./eval_videos",
    ) -> dict:
        """Evaluate a trained model."""
        
        print(f"\nEvaluating: {model_path}")
        
        if record_video:
            os.makedirs(video_path, exist_ok=True)
        
        model = PPO.load(model_path)
        
        env = self._make_env(
            instruction=instruction,
            record_video=record_video,
            video_path=video_path,
            render_mode="rgb_array" if record_video else None,
        )
        
        episode_rewards = []
        episode_lengths = []
        episode_similarities = []
        
        for ep in range(n_episodes):
            obs, info = env.reset()
            done = False
            ep_reward = 0
            ep_length = 0
            ep_sims = []
            
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                
                ep_reward += reward
                ep_length += 1
                if 'motion_similarity' in info:
                    ep_sims.append(info['motion_similarity'])
                    
                done = terminated or truncated
                
            episode_rewards.append(ep_reward)
            episode_lengths.append(ep_length)
            if ep_sims:
                episode_similarities.append(np.mean(ep_sims))
                
            print(f"  Episode {ep+1}: r={ep_reward:.1f}, len={ep_length}, sim={np.mean(ep_sims) if ep_sims else 0:.3f}")
        
        env.close()
        
        results = {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'mean_length': np.mean(episode_lengths),
            'mean_similarity': np.mean(episode_similarities) if episode_similarities else 0,
        }
        
        print(f"\nResults: reward={results['mean_reward']:.1f}±{results['std_reward']:.1f}, "
              f"len={results['mean_length']:.0f}, sim={results['mean_similarity']:.3f}")
        
        return results


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("Testing EnhancedMotionLanguageAgent...\n")
    
    agent = EnhancedMotionLanguageAgent(
        env_name="Humanoid-v4",
        use_stability_focus=True,
    )
    
    env = agent._make_env("walk forward")
    obs, info = env.reset()
    
    print(f"Obs shape: {obs.shape}")
    print(f"Action space: {env.action_space}")
    
    print("\nRunning 20 test steps...")
    for i in range(20):
        action = env.action_space.sample() * 0.1
        obs, reward, term, trunc, info = env.step(action)
        
        if i < 5 or i >= 15:  # Show first 5 and last 5
            print(f"  Step {i+1}: r={reward:.2f}, h={info['height']:.2f}, "
                  f"warmup={info['is_warmup']}, sim={info['motion_similarity']:.3f}")
        elif i == 5:
            print("  ...")
        
        if term or trunc:
            print(f"  Terminated: {info.get('termination_reason', 'truncated')}")
            break
    
    env.close()
    print("\nTest complete!")
