"""
Hierarchical Motion-Language Agent
<<<<<<< HEAD
Integration with MotionGPT tokenizer and AnySkill hierarchical structure
=======
Direct integration with MotionGPT - NO pixel rendering or CLIP
This replaces the AnySkill pipeline with direct motion-language learning
>>>>>>> 8470d39 (added and changed)
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

# Fix MuJoCo OpenGL context for Windows
if os.name == 'nt':  # Windows
    os.environ['MUJOCO_GL'] = 'glfw'
elif 'MUJOCO_GL' not in os.environ:
    os.environ['MUJOCO_GL'] = 'glfw'  # Safe default for most systems

# Add project src to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root / "src"))

from models.motion_tokenizer import MotionTokenizer


class DirectMotionLanguageWrapper(gym.Wrapper):
    """
    Environment wrapper that computes rewards directly from motion-language alignment

    KEY INNOVATION: This replaces the AnySkill pipeline of:
    1. Agent acts -> 2. Render to pixels -> 3. CLIP encoding -> 4. Reward

    With direct pipeline:
    1. Agent acts -> 2. Extract motion features -> 3. MotionGPT encoding -> 4. Language alignment -> 5. Reward

    This removes the computational bottleneck and enables real-time language-conditioned RL
    """

    def __init__(self,
                 env,
                 motion_tokenizer: MotionTokenizer,
                 instruction: str = "walk forward",
                 reward_scale: float = 1.0,
                 motion_history_length: int = 20,
                 reward_aggregation: str = "weighted_recent"):

        super().__init__(env)
        self.motion_tokenizer = motion_tokenizer
        self.current_instruction = instruction
        self.reward_scale = reward_scale
        self.motion_history_length = motion_history_length
        self.reward_aggregation = reward_aggregation

        # Motion tracking with efficient storage
        self.motion_history = deque(maxlen=motion_history_length)
        self.observation_history = deque(maxlen=motion_history_length)

        # Environment-specific settings
        self.env_name = env.spec.id if hasattr(env, 'spec') else "Unknown"

        # Reward computation settings
        self.language_reward_weight = 0.5  # Balance with environment reward
        self.success_threshold = 0.6
        self.reward_smoothing = 0.9  # For temporal smoothing
        self.prev_language_reward = 0.0

        # Performance tracking
        self.step_count = 0
        self.episode_language_rewards = []
        self.computation_times = []

        print(f"DirectMotionLanguageWrapper initialized:")
        print(f"  Environment: {self.env_name}")
        print(f"  Instruction: '{instruction}'")
        print(f"  Motion history length: {motion_history_length}")
        print(f"  Reward aggregation: {reward_aggregation}")

    def set_instruction(self, instruction: str):
        """Change the current instruction and reset motion history"""
        self.current_instruction = instruction
        self.motion_history.clear()
        self.observation_history.clear()
        self.prev_language_reward = 0.0
        print(f"Instruction changed to: '{instruction}'")

    def _extract_motion_features(self, obs: np.ndarray) -> np.ndarray:
        """Extract motion features using environment-specific extraction"""
        motion_features = self.motion_tokenizer.extract_motion_from_obs(obs, self.env_name)
        return motion_features

    def _compute_direct_motion_language_reward(self) -> float:
        """
        Compute reward directly from motion-language alignment
        This is the KEY function that replaces CLIP-based rewards
        """
        if len(self.motion_history) < 5:  # Need minimum motion history
            return 0.0

        start_time = time.time()

        try:
            # Prepare motion sequence for MotionGPT
            if self.reward_aggregation == "recent_window":
                # Use only recent motion window
                motion_sequence = np.array(list(self.motion_history)[-10:])
            elif self.reward_aggregation == "full_sequence":
                # Use full motion history
                motion_sequence = np.array(list(self.motion_history))
            elif self.reward_aggregation == "weighted_recent":
                # Use weighted combination favoring recent motion
                motion_list = list(self.motion_history)
                weights = np.exp(np.linspace(-1, 0, len(motion_list)))  # Exponential weighting
                weighted_motion = np.average(motion_list, axis=0, weights=weights)
                motion_sequence = np.expand_dims(weighted_motion, axis=0)
            else:
                # Default: simple recent window
                motion_sequence = np.array(list(self.motion_history)[-10:])

            # Convert to tensor and ensure correct shape
            motion_tensor = torch.from_numpy(motion_sequence).float()
            if motion_tensor.dim() == 2:
                motion_tensor = motion_tensor.unsqueeze(0)  # Add batch dimension

            # Compute motion-language similarity using MotionGPT
            similarity = self.motion_tokenizer.compute_motion_language_similarity(
                motion_tensor,
                self.current_instruction,
                temporal_aggregation="mean"
            )

            # Advanced reward shaping
            if similarity > self.success_threshold:
                # Exponential bonus for high similarity
                reward = self.reward_scale * (1.0 + np.exp(similarity - self.success_threshold) - 1.0)
            else:
                # Smooth reward below threshold
                reward = self.reward_scale * (similarity ** 1.5)  # Slightly nonlinear

            # Temporal smoothing to reduce noise
            smoothed_reward = (self.reward_smoothing * self.prev_language_reward +
                             (1 - self.reward_smoothing) * reward)
            self.prev_language_reward = smoothed_reward

            # Track computation time
            computation_time = time.time() - start_time
            self.computation_times.append(computation_time)

            return smoothed_reward

        except Exception as e:
            print(f"Motion-language reward computation failed: {e}")
            return 0.0

    def _compute_instruction_progress_bonus(self) -> float:
        """
        Compute additional reward based on instruction-specific progress
        This provides more fine-grained feedback for complex instructions
        """
        if len(self.motion_history) < 10:
            return 0.0

        try:
            # Get recent motion trend
            recent_motion = np.array(list(self.motion_history)[-10:])

            # Instruction-specific progress rewards
            instruction_lower = self.current_instruction.lower()

            if "forward" in instruction_lower or "walk" in instruction_lower:
                # Reward forward movement (x-axis progression)
                if recent_motion.shape[1] > 0:  # Check if position data available
                    x_movement = recent_motion[-1, 0] - recent_motion[0, 0]  # x position change
                    return max(0.0, x_movement * 0.1)  # Small bonus for forward movement

            elif "turn" in instruction_lower:
                # Reward rotational movement
                if recent_motion.shape[1] > 2:  # Check if orientation data available
                    orientation_change = np.var(recent_motion[:, 2:5])  # Variance in orientation
                    return min(0.2, orientation_change * 0.5)  # Cap bonus for turning

            elif "jump" in instruction_lower:
                # Reward vertical movement
                if recent_motion.shape[1] > 1:  # Check if z position available
                    z_movement = np.max(recent_motion[-5:, 1]) - np.min(recent_motion[-5:, 1])
                    return min(0.3, z_movement * 0.2)  # Bonus for vertical motion

            elif "stop" in instruction_lower or "still" in instruction_lower:
                # Reward low movement
                movement_variance = np.var(recent_motion[-5:], axis=0).mean()
                return max(0.0, 0.2 - movement_variance * 10)  # Inverse bonus for stillness

            return 0.0

        except Exception as e:
            print(f"Progress bonus computation failed: {e}")
            return 0.0

    def step(self, action):
        """Step function with direct motion-language reward computation"""
        # Execute action in environment
        obs, env_reward, terminated, truncated, info = self.env.step(action)

        # Extract and store motion features
        motion_features = self._extract_motion_features(obs)
        self.motion_history.append(motion_features)
        self.observation_history.append(obs.copy())

        # Compute direct motion-language reward (no pixel rendering!)
        language_reward = self._compute_direct_motion_language_reward()

        # Compute instruction-specific progress bonus
        progress_bonus = self._compute_instruction_progress_bonus()

        # Total language reward
        total_language_reward = language_reward + progress_bonus

        # Combine with environment reward
        if self.language_reward_weight > 0:
            total_reward = ((1 - self.language_reward_weight) * env_reward +
                           self.language_reward_weight * total_language_reward)
        else:
            total_reward = env_reward  # Pure environment reward

        # Track performance
        self.step_count += 1
        self.episode_language_rewards.append(total_language_reward)

        # Info logging
        info.update({
            'language_reward': total_language_reward,
            'motion_language_similarity': language_reward / self.reward_scale if self.reward_scale > 0 else 0,
            'progress_bonus': progress_bonus,
            'original_reward': env_reward,
            'instruction': self.current_instruction,
            'motion_history_length': len(self.motion_history),
            'step_count': self.step_count,
            'avg_computation_time': np.mean(self.computation_times[-100:]) if self.computation_times else 0
        })

        return obs, total_reward, terminated, truncated, info

    def reset(self, **kwargs):
        """Reset environment and motion tracking"""
        obs, info = self.env.reset(**kwargs)

        # Reset motion tracking
        self.motion_history.clear()
        self.observation_history.clear()
        self.prev_language_reward = 0.0
        self.step_count = 0

        # Add initial motion
        motion_features = self._extract_motion_features(obs)
        self.motion_history.append(motion_features)
        self.observation_history.append(obs.copy())

        # Info
        info.update({
            'instruction': self.current_instruction,
            'episode_start': True
        })

        # Log episode statistics
        if self.episode_language_rewards:
            avg_lang_reward = np.mean(self.episode_language_rewards)
            info['prev_episode_avg_language_reward'] = avg_lang_reward
            self.episode_language_rewards = []  # Reset for new episode

        return obs, info


class MotionLanguageAgent:
    """
    Agent that learns directly from natural language using MotionGPT

    This implementation completely removes the pixel rendering bottleneck from AnySkill
    and enables real-time language-conditioned reinforcement learning
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

        # Initialize MotionGPT tokenizer
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

        # Training configuration
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

        # Test environment creation
        self._test_environment_creation()

    def _test_environment_creation(self):
        """Test if the environment can be created and suggest alternatives if not"""
        try:
            test_env = gym.make(self.env_name)
            test_env.close()
            print(f"✓ Environment {self.env_name} is working correctly")
        except Exception as e:
            print(f"⚠️  Warning: {self.env_name} failed to load: {e}")
            print("Trying alternative environments...")

            # Try alternative environments
            alternatives = ["HalfCheetah-v4", "Ant-v4", "Walker2d-v4", "CartPole-v1"]
            for alt_env in alternatives:
                try:
                    test_env = gym.make(alt_env)
                    test_env.close()
                    print(f"✓ Alternative environment {alt_env} is working")
                    self.env_name = alt_env
                    print(f"Switched to {alt_env} for this session")
                    break
                except Exception:
                    continue
            else:
                print("❌ No working MuJoCo environments found")
                print("Please check your MuJoCo installation or run:")
                print("  conda install -c conda-forge mujoco")
                print("  or pip install mujoco")
                raise RuntimeError("No compatible environments available")

    def create_training_environment(self,
                                    instruction: str = "walk forward",
                                    language_reward_weight: float = 0.5,
                                    n_envs: int = 1,
                                    use_multiprocessing: bool = False):
        """Create vectorized environment for training with direct motion-language rewards"""

        def make_env():
            env = gym.make(self.env_name)
            env = Monitor(env)  # Add monitoring
            env = DirectMotionLanguageWrapper(
                env,
                self.motion_tokenizer,
                instruction=instruction,
                reward_scale=1.0,
                motion_history_length=20,
                reward_aggregation="weighted_recent"
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
                             verbose: int = 1):
        """
        Train agent on a specific instruction using direct motion-language learning

        Args:
            instruction: Natural language instruction
            total_timesteps: Total training timesteps
            language_reward_weight: Balance between env reward and language reward (0-1)
            save_path: Path to save checkpoints
            eval_freq: Frequency of evaluation
            n_envs: Number of parallel environments
            verbose: Verbosity level
        """

        print(f"Training on instruction: '{instruction}'")
        print(f"Language reward weight: {language_reward_weight}")
        print(f"Parallel environments: {n_envs}")

        # Create training environment
        vec_env = self.create_training_environment(
            instruction=instruction,
            language_reward_weight=language_reward_weight,
            n_envs=n_envs,
            use_multiprocessing=(n_envs > 1)
        )

        # Create RL agent with configuration
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
            save_freq=eval_freq // n_envs,  # Adjust for parallel envs
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
                             deterministic: bool = True):
        """Evaluate agent on instruction with detailed analysis"""

        print(f"Evaluating on instruction: '{instruction}'")

        # Load model if provided
        if model_path and Path(model_path).exists():
            if self.rl_agent is None:
                # Create dummy env for loading
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
        eval_env = DirectMotionLanguageWrapper(
            eval_env,
            self.motion_tokenizer,
            instruction=instruction
        )

        # Set language reward weight
        eval_env.language_reward_weight = language_reward_weight

        # Run evaluation episodes
        episode_results = []

        for episode in range(num_episodes):
            obs, info = eval_env.reset()
            episode_data = {
                'total_reward': 0,
                'language_reward': 0,
                'env_reward': 0,
                'steps': 0,
                'similarities': [],
                'computation_times': []
            }

            step_count = 0
            while step_count < 1000:  # Max episode length
                action, _ = self.rl_agent.predict(obs, deterministic=deterministic)
                obs, reward, terminated, truncated, info = eval_env.step(action)

                episode_data['total_reward'] += reward
                episode_data['language_reward'] += info.get('language_reward', 0)
                episode_data['env_reward'] += info.get('original_reward', 0)
                episode_data['similarities'].append(info.get('motion_language_similarity', 0))
                episode_data['computation_times'].append(info.get('avg_computation_time', 0))

                step_count += 1

                if render and episode == 0:  # Render first episode only
                    eval_env.render()

                if terminated or truncated:
                    break

            episode_data['steps'] = step_count
            episode_results.append(episode_data)

            print(f"Episode {episode+1}: Total={episode_data['total_reward']:.2f}, "
                  f"Language={episode_data['language_reward']:.2f}, "
                  f"Similarity={np.mean(episode_data['similarities']):.3f}, "
                  f"Steps={step_count}")

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
            'mean_episode_length': np.mean([ep['steps'] for ep in episode_results]),
            'mean_computation_time': np.mean([np.mean(ep['computation_times']) for ep in episode_results if ep['computation_times']]),
            'episode_data': episode_results
        }

        print(f"\nEvaluation Results for '{instruction}':")
        print(f"  Mean Total Reward: {results['mean_total_reward']:.2f} ± {results['std_total_reward']:.2f}")
        print(f"  Mean Language Reward: {results['mean_language_reward']:.2f}")
        print(f"  Mean Motion-Language Similarity: {results['mean_similarity']:.3f}")
        print(f"  Mean Episode Length: {results['mean_episode_length']:.1f}")
        print(f"  Mean Computation Time: {results['mean_computation_time']:.4f}s per step")

        return results

    def demo_instruction_following(self,
                                   instructions: list = None,
                                   model_path: str = None,
                                   max_steps: int = 1000,
                                   render: bool = True,
                                   save_video: bool = True):
        """Demonstrate the agent following different instructions"""

        if instructions is None:
            instructions = ["walk forward", "turn left", "turn right", "stop"]

        print(f"Demonstrating instruction following...")

        # Load model if provided
        if model_path and Path(model_path).exists():
            vec_env = self.create_training_environment("walk forward")
            self.rl_agent = PPO.load(model_path, env=vec_env)
            print(f"Loaded model: {model_path}")

        if self.rl_agent is None:
            print("No trained agent! Training a quick baseline...")
            self.train_on_instruction("walk forward", total_timesteps=10000)

        # Demo each instruction
        for instruction in instructions:
            print(f"\nFollowing: '{instruction}'")

            # Create demo environment with live viewing
            demo_env = gym.make(self.env_name, render_mode="human")
            demo_env = DirectMotionLanguageWrapper(
                demo_env,
                self.motion_tokenizer,
                instruction=instruction
            )

            # Create separate environment for video recording if needed
            if save_video:
                video_env = gym.make(self.env_name, render_mode="rgb_array")
                video_env = DirectMotionLanguageWrapper(
                    video_env,
                    self.motion_tokenizer,
                    instruction=instruction
                )
                frames = []

            obs, info = demo_env.reset()
            if save_video:
                video_obs, _ = video_env.reset()

            total_reward = 0
            language_reward = 0

            for step in range(max_steps):
                action, _ = self.rl_agent.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = demo_env.step(action)

                if save_video:
                    _, _, _, _, _ = video_env.step(action)
                    frames.append(video_env.render())

                if render:
                    demo_env.render()

                total_reward += reward
                language_reward += info.get('language_reward', 0)

                if step % 50 == 0:
                    print(f"  Step {step}: Total reward={total_reward:.2f}, "
                          f"Language reward={language_reward:.2f}")

                if terminated or truncated:
                    print(f"  Episode ended at step {step} (terminated={terminated}, truncated={truncated})")
                    break

            # Save video if requested
            if save_video and frames:
                import imageio
                video_path = f"demo_{instruction.replace(' ', '_')}.mp4"
                imageio.mimsave(video_path, frames, fps=30)
                print(f"Video saved: {video_path}")

            demo_env.close()
            if save_video:
                video_env.close()


def test_hierarchical_agent():
    """Test the hierarchical agent"""
    print("Testing Hierarchical Motion-Language Agent")
    print("=" * 45)

    # Create agent
    agent = MotionLanguageAgent("Humanoid-v4")

    # Quick training test
    print("\n1. Quick training test...")
    model_path = agent.train_on_instruction(
        instruction="walk forward",
        total_timesteps=5000,  # Short for testing
        save_path="./test_checkpoints/"
    )

    # Evaluation test
    print("\n2. Evaluation test...")
    results = agent.evaluate_instruction(
        instruction="walk forward",
        num_episodes=3,
        model_path=model_path
    )

    # Demo test
    print("\n3. Demo test...")
    agent.demo_instruction_following(
        instructions=["walk forward", "turn left"],
        model_path=model_path,
        max_steps=100
    )

    print("\nHierarchical agent test completed")
    return agent


if __name__ == "__main__":
    agent = test_hierarchical_agent()
