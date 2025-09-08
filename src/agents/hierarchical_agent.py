"""
Motion-Language Agent with Fixed MotionGPT Integration
Direct motion-language learning with proper evaluation metrics and video recording
"""

import sys
import os
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from collections import deque
import time
import imageio
import json

# Fix MuJoCo OpenGL context
if os.name == 'nt':
    os.environ['MUJOCO_GL'] = 'glfw'
elif 'MUJOCO_GL' not in os.environ:
    os.environ['MUJOCO_GL'] = 'glfw'

# Add project src to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root / "src"))

from models.motion_tokenizer import MotionTokenizer


class DirectMotionLanguageWrapper(gym.Wrapper):
    """
    Enhanced environment wrapper with proper success rate computation and video recording
    """

    def __init__(self,
                 env,
                 motion_tokenizer: MotionTokenizer,
                 instruction: str = "walk forward",
                 reward_scale: float = 1.0,
                 motion_history_length: int = 20,
                 reward_aggregation: str = "weighted_recent",
                 record_video: bool = False,
                 video_path: str = None):

        super().__init__(env)
        self.motion_tokenizer = motion_tokenizer
        self.current_instruction = instruction
        self.reward_scale = reward_scale
        self.motion_history_length = motion_history_length
        self.reward_aggregation = reward_aggregation
        self.record_video = record_video
        self.video_path = video_path

        # Motion tracking
        self.motion_history = deque(maxlen=motion_history_length)
        self.observation_history = deque(maxlen=motion_history_length)

        # Video recording
        self.video_frames = [] if record_video else None
        self.episode_count = 0

        # Environment info
        self.env_name = env.spec.id if hasattr(env, 'spec') else "Unknown"

        # Reward computation
        self.language_reward_weight = 0.5
        self.success_threshold = 0.6
        self.reward_smoothing = 0.9
        self.prev_language_reward = 0.0

        # Performance tracking
        self.step_count = 0
        self.episode_step_count = 0
        self.episode_language_rewards = []
        self.episode_similarities = []
        self.episode_success_rates = []
        self.computation_times = []

        # Success tracking
        self.current_episode_success = False
        self.success_history = deque(maxlen=100)

        print(f"DirectMotionLanguageWrapper initialized:")
        print(f"  Environment: {self.env_name}")
        print(f"  Instruction: '{instruction}'")
        print(f"  Motion history length: {motion_history_length}")
        print(f"  Reward aggregation: {reward_aggregation}")
        print(f"  Video recording: {record_video}")

    def set_instruction(self, instruction: str):
        """Change instruction and reset tracking"""
        self.current_instruction = instruction
        self.motion_history.clear()
        self.observation_history.clear()
        self.prev_language_reward = 0.0
        self.current_episode_success = False
        print(f"Instruction changed to: '{instruction}'")

    def _extract_motion_features(self, obs: np.ndarray) -> np.ndarray:
        """Extract motion features using improved tokenizer"""
        motion_features = self.motion_tokenizer.extract_motion_from_obs(obs, self.env_name)
        return motion_features

    def _compute_enhanced_motion_language_reward(self) -> tuple:
        """
        Compute enhanced motion-language reward with success rate
        Returns: (language_reward, similarity, success_rate, quality_metrics)
        """
        if len(self.motion_history) < 5:
            return 0.0, 0.0, 0.0, {}

        start_time = time.time()

        try:
            # Prepare motion sequence
            if self.reward_aggregation == "recent_window":
                motion_sequence = np.array(list(self.motion_history)[-10:])
            elif self.reward_aggregation == "full_sequence":
                motion_sequence = np.array(list(self.motion_history))
            elif self.reward_aggregation == "weighted_recent":
                motion_list = list(self.motion_history)
                weights = np.exp(np.linspace(-1, 0, len(motion_list)))
                weighted_motion = np.average(motion_list, axis=0, weights=weights)
                motion_sequence = np.expand_dims(weighted_motion, axis=0)
            else:
                motion_sequence = np.array(list(self.motion_history)[-10:])

            # Convert to tensor
            motion_tensor = torch.from_numpy(motion_sequence).float()
            if motion_tensor.dim() == 2:
                motion_tensor = motion_tensor.unsqueeze(0)

            # Compute motion-language similarity
            similarity = self.motion_tokenizer.compute_motion_language_similarity(
                motion_tensor,
                self.current_instruction,
                temporal_aggregation="weighted_recent"
            )

            # Compute success rate
            success_rate = self.motion_tokenizer.compute_success_rate(
                motion_tensor,
                self.current_instruction
            )

            # Compute motion quality metrics
            quality_metrics = self.motion_tokenizer.motion_evaluator.evaluate_motion_quality(
                motion_tensor
            )

            # Enhanced reward computation
            base_reward = similarity * self.reward_scale

            # Success bonus
            if success_rate > 0.5:
                success_bonus = success_rate * 0.5  # Additional reward for success
                self.current_episode_success = True
            else:
                success_bonus = 0.0

            # Quality bonus
            quality_bonus = quality_metrics.get('overall_quality', 0.0) * 0.2

            # Combined reward
            total_language_reward = base_reward + success_bonus + quality_bonus

            # Temporal smoothing
            smoothed_reward = (self.reward_smoothing * self.prev_language_reward +
                             (1 - self.reward_smoothing) * total_language_reward)
            self.prev_language_reward = smoothed_reward

            # Track computation time
            computation_time = time.time() - start_time
            self.computation_times.append(computation_time)

            return smoothed_reward, similarity, success_rate, quality_metrics

        except Exception as e:
            print(f"Enhanced reward computation failed: {e}")
            return 0.0, 0.0, 0.0, {}

    def _compute_instruction_progress_bonus(self) -> float:
        """Compute instruction-specific progress bonus"""
        if len(self.motion_history) < 10:
            return 0.0

        try:
            recent_motion = np.array(list(self.motion_history)[-10:])
            instruction_lower = self.current_instruction.lower()

            if "forward" in instruction_lower:
                if recent_motion.shape[1] > 0:
                    x_movement = recent_motion[-1, 0] - recent_motion[0, 0]
                    return max(0.0, x_movement * 0.1)

            elif "backward" in instruction_lower:
                if recent_motion.shape[1] > 0:
                    x_movement = recent_motion[0, 0] - recent_motion[-1, 0]  # Negative movement
                    return max(0.0, x_movement * 0.1)

            elif "turn" in instruction_lower:
                if recent_motion.shape[1] > 2:
                    orientation_change = np.var(recent_motion[:, 2:5])
                    return min(0.2, orientation_change * 0.5)

            elif "jump" in instruction_lower:
                if recent_motion.shape[1] > 1:
                    z_movement = np.max(recent_motion[-5:, 1]) - np.min(recent_motion[-5:, 1])
                    return min(0.3, z_movement * 0.2)

            elif "stop" in instruction_lower or "still" in instruction_lower:
                movement_variance = np.var(recent_motion[-5:], axis=0).mean()
                return max(0.0, 0.2 - movement_variance * 10)

            return 0.0

        except Exception as e:
            print(f"Progress bonus computation failed: {e}")
            return 0.0

    def step(self, action):
        """Enhanced step function with comprehensive metrics tracking"""
        # Execute action
        obs, env_reward, terminated, truncated, info = self.env.step(action)

        # Extract and store motion features
        motion_features = self._extract_motion_features(obs)
        self.motion_history.append(motion_features)
        self.observation_history.append(obs.copy())

        # Record video frame if enabled
        if self.record_video and self.video_frames is not None:
            try:
                frame = self.env.render()
                if frame is not None:
                    self.video_frames.append(frame)
            except Exception as e:
                print(f"Video recording failed: {e}")

        # Compute enhanced rewards and metrics
        language_reward, similarity, success_rate, quality_metrics = self._compute_enhanced_motion_language_reward()

        # Compute progress bonus
        progress_bonus = self._compute_instruction_progress_bonus()

        # Total language reward
        total_language_reward = language_reward + progress_bonus

        # Combine with environment reward
        if self.language_reward_weight > 0:
            total_reward = ((1 - self.language_reward_weight) * env_reward +
                           self.language_reward_weight * total_language_reward)
        else:
            total_reward = env_reward

        # Track performance
        self.step_count += 1
        self.episode_step_count += 1
        self.episode_language_rewards.append(total_language_reward)
        self.episode_similarities.append(similarity)
        self.episode_success_rates.append(success_rate)

        # Enhanced info logging
        info.update({
            'language_reward': total_language_reward,
            'motion_language_similarity': similarity,
            'success_rate': success_rate,
            'progress_bonus': progress_bonus,
            'original_reward': env_reward,
            'instruction': self.current_instruction,
            'motion_history_length': len(self.motion_history),
            'step_count': self.step_count,
            'episode_step_count': self.episode_step_count,
            'avg_computation_time': np.mean(self.computation_times[-100:]) if self.computation_times else 0,
            'current_episode_success': self.current_episode_success,

            # Motion quality metrics
            'motion_smoothness': quality_metrics.get('smoothness', 0.0),
            'motion_stability': quality_metrics.get('stability', 0.0),
            'motion_naturalness': quality_metrics.get('naturalness', 0.0),
            'motion_overall_quality': quality_metrics.get('overall_quality', 0.0)
        })

        return obs, total_reward, terminated, truncated, info

    def reset(self, **kwargs):
        """Enhanced reset with episode statistics"""
        # Save video if recording
        if self.record_video and self.video_frames and len(self.video_frames) > 10:
            try:
                if self.video_path:
                    video_filename = f"{self.video_path}/episode_{self.episode_count}_{self.current_instruction.replace(' ', '_')}.mp4"
                    Path(video_filename).parent.mkdir(parents=True, exist_ok=True)
                    imageio.mimsave(video_filename, self.video_frames, fps=30)
                    print(f"Saved video: {video_filename}")
            except Exception as e:
                print(f"Video saving failed: {e}")

        # Compute episode statistics
        episode_stats = {}
        if self.episode_language_rewards:
            episode_stats = {
                'episode_avg_language_reward': np.mean(self.episode_language_rewards),
                'episode_avg_similarity': np.mean(self.episode_similarities),
                'episode_avg_success_rate': np.mean(self.episode_success_rates),
                'episode_final_similarity': self.episode_similarities[-1] if self.episode_similarities else 0.0,
                'episode_success': self.current_episode_success,
                'episode_length': self.episode_step_count
            }

        # Track success history
        self.success_history.append(self.current_episode_success)

        # Reset environment
        obs, info = self.env.reset(**kwargs)

        # Reset tracking
        self.motion_history.clear()
        self.observation_history.clear()
        self.prev_language_reward = 0.0
        self.episode_step_count = 0
        self.current_episode_success = False

        # Reset video recording
        if self.record_video:
            self.video_frames = []
        self.episode_count += 1

        # Add initial motion
        motion_features = self._extract_motion_features(obs)
        self.motion_history.append(motion_features)
        self.observation_history.append(obs.copy())

        # Enhanced info
        info.update({
            'instruction': self.current_instruction,
            'episode_start': True,
            'episode_count': self.episode_count,
            'recent_success_rate': np.mean(list(self.success_history)) if self.success_history else 0.0,
            **episode_stats
        })

        # Reset episode tracking lists
        self.episode_language_rewards = []
        self.episode_similarities = []
        self.episode_success_rates = []

        return obs, info


class EnhancedMotionLanguageAgent:
    """
    Enhanced Motion-Language Agent with proper MotionGPT integration and evaluation
    """

    def __init__(self,
                 env_name: str = "Humanoid-v4",
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 motion_model_config: str = None,
                 motion_checkpoint: str = None):

        self.env_name = env_name
        self.device = device

        print(f"Initializing Motion-Language Agent")
        print(f"Environment: {env_name}")
        print(f"Device: {device}")

        # Initialize enhanced MotionGPT tokenizer
        print("Loading MotionGPT tokenizer...")
        self.motion_tokenizer = MotionTokenizer(
            model_config_path=motion_model_config,
            checkpoint_path=motion_checkpoint,
            device=device
        )

        # Create base environment for testing
        self.base_env = gym.make(env_name)

        # RL agent placeholder
        self.rl_agent = None

        # Enhanced training configuration
        self.training_config = {
            'learning_rate': 3e-4,
            'n_steps': 2048,
            'batch_size': 64,
            'n_epochs': 10,
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'clip_range': 0.2,
            'ent_coef': 0.01,
            'vf_coef': 0.5,
            'max_grad_norm': 0.5
        }

        print("Motion-Language Agent initialized successfully")
        self._test_environment_creation()

    def _test_environment_creation(self):
        """Test environment creation"""
        try:
            test_env = gym.make(self.env_name)
            test_env.close()
            print(f"✓ Environment {self.env_name} is working correctly")
        except Exception as e:
            print(f"⚠️  Warning: {self.env_name} failed to load: {e}")
            alternatives = ["HalfCheetah-v4", "Ant-v4", "Walker2d-v4", "CartPole-v1"]
            for alt_env in alternatives:
                try:
                    test_env = gym.make(alt_env)
                    test_env.close()
                    print(f"✓ Alternative environment {alt_env} is working")
                    self.env_name = alt_env
                    break
                except Exception:
                    continue
            else:
                print("❌ No working MuJoCo environments found")
                raise RuntimeError("No compatible environments available")

    def create_training_environment(self,
                                    instruction: str = "walk forward",
                                    language_reward_weight: float = 0.5,
                                    n_envs: int = 1,
                                    use_multiprocessing: bool = False,
                                    record_video: bool = False,
                                    video_path: str = None):
        """Create enhanced vectorized environment"""

        def make_env():
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
                video_path=video_path
            )
            env.language_reward_weight = language_reward_weight
            return env

        if n_envs == 1:
            return DummyVecEnv([make_env])
        else:
            if use_multiprocessing:
                return SubprocVecEnv([make_env for _ in range(n_envs)])
            else:
                return DummyVecEnv([make_env for _ in range(n_envs)])

    def train_on_instruction(self,
                             instruction: str = "walk forward",
                             total_timesteps: int = 100000,
                             language_reward_weight: float = 0.5,
                             save_path: str = "./checkpoints/",
                             eval_freq: int = 10000,
                             n_envs: int = 4,
                             verbose: int = 1,
                             record_training_videos: bool = False):
        """Enhanced training with video recording and better metrics"""

        print(f"Training on instruction: '{instruction}'")
        print(f"Language reward weight: {language_reward_weight}")
        print(f"Parallel environments: {n_envs}")

        # Create training environment
        video_path = f"{save_path}/training_videos" if record_training_videos else None
        vec_env = self.create_training_environment(
            instruction=instruction,
            language_reward_weight=language_reward_weight,
            n_envs=n_envs,
            use_multiprocessing=(n_envs > 1),
            record_video=record_training_videos,
            video_path=video_path
        )

        # Create RL agent
        self.rl_agent = PPO(
            "MlpPolicy",
            vec_env,
            learning_rate=self.training_config['learning_rate'],
            n_steps=self.training_config['n_steps'],
            batch_size=self.training_config['batch_size'],
            n_epochs=self.training_config['n_epochs'],
            gamma=self.training_config['gamma'],
            gae_lambda=self.training_config['gae_lambda'],
            clip_range=self.training_config['clip_range'],
            ent_coef=self.training_config['ent_coef'],
            vf_coef=self.training_config['vf_coef'],
            max_grad_norm=self.training_config['max_grad_norm'],
            verbose=verbose,
            device=self.device,
            tensorboard_log="./logs/"
        )

        # Setup callbacks
        Path(save_path).mkdir(parents=True, exist_ok=True)

        checkpoint_callback = CheckpointCallback(
            save_freq=eval_freq // n_envs,
            save_path=save_path,
            name_prefix=f"motion_lang_{instruction.replace(' ', '_')}"
        )

        # Create evaluation environment
        eval_env = self.create_training_environment(
            instruction=instruction,
            language_reward_weight=language_reward_weight,
            n_envs=1
        )

        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=save_path,
            log_path=save_path,
            eval_freq=eval_freq // n_envs,
            deterministic=True,
            render=False,
            n_eval_episodes=5
        )

        # Train the agent
        print(f"Starting training for {total_timesteps} timesteps...")
        print("Using DIRECT motion-language learning (no pixel rendering)")

        start_time = time.time()

        self.rl_agent.learn(
            total_timesteps=total_timesteps,
            callback=[checkpoint_callback, eval_callback],
            progress_bar=True
        )

        training_time = time.time() - start_time

        # Save final model
        final_path = f"{save_path}/final_model_{instruction.replace(' ', '_')}.zip"
        self.rl_agent.save(final_path)

        print(f"Training completed in {training_time:.2f} seconds!")
        print(f"Final model saved to: {final_path}")

        # Close environments
        vec_env.close()
        eval_env.close()

        return final_path

    def evaluate_instruction(self,
                             instruction: str = "walk forward",
                             model_path: str = None,
                             num_episodes: int = 10,
                             language_reward_weight: float = 0.5,
                             render: bool = False,
                             deterministic: bool = True,
                             record_video: bool = False,
                             video_path: str = None):
        """Enhanced evaluation with comprehensive metrics and video recording"""

        print(f"Evaluating on instruction: '{instruction}'")

        # Load model if provided
        if model_path and Path(model_path).exists():
            if self.rl_agent is None:
                dummy_env = self.create_training_environment(instruction, language_reward_weight)
                self.rl_agent = PPO.load(model_path, env=dummy_env)
                dummy_env.close()
            else:
                self.rl_agent = PPO.load(model_path)
            print(f"Loaded model from: {model_path}")

        if self.rl_agent is None:
            print("No trained agent available! Train first or provide model_path.")
            return {}

        # Create evaluation environment
        eval_env = gym.make(self.env_name)
        if render:
            eval_env = gym.make(self.env_name, render_mode="human")

        eval_env = DirectMotionLanguageWrapper(
            eval_env,
            self.motion_tokenizer,
            instruction=instruction,
            record_video=record_video,
            video_path=video_path
        )
        eval_env.language_reward_weight = language_reward_weight

        # Run evaluation episodes
        episode_results = []
        total_success_count = 0

        for episode in range(num_episodes):
            obs, info = eval_env.reset()
            episode_data = {
                'total_reward': 0,
                'language_reward': 0,
                'env_reward': 0,
                'steps': 0,
                'similarities': [],
                'success_rates': [],
                'computation_times': [],
                'motion_quality': {
                    'smoothness': [],
                    'stability': [],
                    'naturalness': [],
                    'overall_quality': []
                }
            }

            step_count = 0
            episode_success = False

            while step_count < 1000:  # Max episode length
                action, _ = self.rl_agent.predict(obs, deterministic=deterministic)
                obs, reward, terminated, truncated, info = eval_env.step(action)

                episode_data['total_reward'] += reward
                episode_data['language_reward'] += info.get('language_reward', 0)
                episode_data['env_reward'] += info.get('original_reward', 0)
                episode_data['similarities'].append(info.get('motion_language_similarity', 0))
                episode_data['success_rates'].append(info.get('success_rate', 0))
                episode_data['computation_times'].append(info.get('avg_computation_time', 0))

                # Track motion quality
                episode_data['motion_quality']['smoothness'].append(info.get('motion_smoothness', 0))
                episode_data['motion_quality']['stability'].append(info.get('motion_stability', 0))
                episode_data['motion_quality']['naturalness'].append(info.get('motion_naturalness', 0))
                episode_data['motion_quality']['overall_quality'].append(info.get('motion_overall_quality', 0))

                # Check for episode success
                if info.get('current_episode_success', False):
                    episode_success = True

                step_count += 1

                if render:
                    eval_env.render()

                if terminated or truncated:
                    break

            episode_data['steps'] = step_count
            episode_data['episode_success'] = episode_success
            if episode_success:
                total_success_count += 1

            episode_results.append(episode_data)

            print(f"Episode {episode+1}: Total={episode_data['total_reward']:.2f}, "
                  f"Language={episode_data['language_reward']:.2f}, "
                  f"Similarity={np.mean(episode_data['similarities']):.3f}, "
                  f"Success={episode_success}, Steps={step_count}")

        eval_env.close()

        # Compute comprehensive statistics
        results = {
            'instruction': instruction,
            'num_episodes': num_episodes,
            'mean_total_reward': np.mean([ep['total_reward'] for ep in episode_results]),
            'std_total_reward': np.std([ep['total_reward'] for ep in episode_results]),
            'mean_language_reward': np.mean([ep['language_reward'] for ep in episode_results]),
            'mean_env_reward': np.mean([ep['env_reward'] for ep in episode_results]),
            'mean_similarity': np.mean([np.mean(ep['similarities']) for ep in episode_results]),
            'mean_success_rate': np.mean([np.mean(ep['success_rates']) for ep in episode_results]),
            'mean_episode_length': np.mean([ep['steps'] for ep in episode_results]),
            'mean_computation_time': np.mean([np.mean(ep['computation_times']) for ep in episode_results if ep['computation_times']]),

            # Success metrics
            'episode_success_rate': total_success_count / num_episodes,
            'total_successful_episodes': total_success_count,

            # Motion quality metrics
            'mean_motion_smoothness': np.mean([np.mean(ep['motion_quality']['smoothness']) for ep in episode_results if ep['motion_quality']['smoothness']]),
            'mean_motion_stability': np.mean([np.mean(ep['motion_quality']['stability']) for ep in episode_results if ep['motion_quality']['stability']]),
            'mean_motion_naturalness': np.mean([np.mean(ep['motion_quality']['naturalness']) for ep in episode_results if ep['motion_quality']['naturalness']]),
            'mean_motion_overall_quality': np.mean([np.mean(ep['motion_quality']['overall_quality']) for ep in episode_results if ep['motion_quality']['overall_quality']]),

            'episode_data': episode_results
        }

        print(f"\nEvaluation Results for '{instruction}':")
        print(f"  Mean Total Reward: {results['mean_total_reward']:.2f} ± {results['std_total_reward']:.2f}")
        print(f"  Mean Language Reward: {results['mean_language_reward']:.2f}")
        print(f"  Mean Motion-Language Similarity: {results['mean_similarity']:.3f}")
        print(f"  Episode Success Rate: {results['episode_success_rate']:.1%}")
        print(f"  Mean Episode Length: {results['mean_episode_length']:.1f}")
        print(f"  Mean Computation Time: {results['mean_computation_time']:.4f}s per step")
        print(f"  Mean Motion Quality: {results['mean_motion_overall_quality']:.3f}")

        return results


# Alias for backward compatibility
MotionLanguageAgent = EnhancedMotionLanguageAgent


def test_enhanced_agent():
    """Test the enhanced agent"""
    print("Testing Enhanced Motion-Language Agent")
    print("=" * 45)

    # Create agent
    agent = EnhancedMotionLanguageAgent("Humanoid-v4")

    # Quick training test
    print("\n1. Quick training test...")
    model_path = agent.train_on_instruction(
        instruction="walk forward",
        total_timesteps=5000,
        save_path="./test_checkpoints/",
        record_training_videos=True
    )

    # Enhanced evaluation test
    print("\n2. Enhanced evaluation test...")
    results = agent.evaluate_instruction(
        instruction="walk forward",
        num_episodes=3,
        model_path=model_path,
        record_video=True,
        video_path="./test_videos/"
    )

    print("\nEnhanced agent test completed")
    return agent


if __name__ == "__main__":
    agent = test_enhanced_agent()