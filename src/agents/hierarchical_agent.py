"""
Hierarchical Motion-Language Agent
Integration with MotionGPT tokenizer and AnySkill hierarchical structure
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from transformers import T5Tokenizer, T5EncoderModel

# Add project src to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root / "src"))

from models.motion_tokenizer import MotionTokenizer


class LanguageMotionRewardWrapper(gym.Wrapper):
    """
    Environment wrapper that adds language-based rewards using MotionGPT
    This replaces AnySkill's CLIP-based rewards with motion-language alignment
    """

    def __init__(self, env, motion_tokenizer, instruction="walk forward", reward_scale=1.0):
        super().__init__(env)
        self.motion_tokenizer = motion_tokenizer
        self.current_instruction = instruction
        self.reward_scale = reward_scale

        # Motion tracking
        self.motion_history = []
        self.max_history_length = 50

        # Language encoder for instruction embedding
        self.language_tokenizer = T5Tokenizer.from_pretrained('t5-small')
        self.language_encoder = T5EncoderModel.from_pretrained('t5-small')
        self.language_encoder.eval()
        self.use_t5 = True

        self.device = next(self.motion_tokenizer.vqvae_model.parameters()).device if (
            self.motion_tokenizer.vqvae_model and
            hasattr(self.motion_tokenizer.vqvae_model, 'parameters')
        ) else motion_tokenizer.device

        if self.use_t5:
            self.language_encoder.to(self.device)

        print(f"LanguageMotionRewardWrapper initialized with instruction: '{instruction}'")

    def set_instruction(self, instruction: str):
        """Change the current instruction"""
        self.current_instruction = instruction
        print(f"Instruction set to: '{instruction}'")

    def _encode_instruction(self, instruction: str) -> torch.Tensor:
        """Encode instruction to embedding using T5"""
        try:
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
                # Use mean of last hidden states as instruction embedding
                instruction_embedding = outputs.last_hidden_state.mean(dim=1).squeeze()

            return instruction_embedding

        except Exception as e:
            print(f"Instruction encoding failed: {e}")
            # Fallback to random embedding
            return torch.randn(512, device=self.device)

    def _extract_motion_features(self, obs: np.ndarray) -> np.ndarray:
        """Extract motion-relevant features from observation"""
        # For humanoid, extract joint positions/velocities
        # This is a simplified extraction - in practice you'd want proper joint mapping

        if len(obs) >= 200:
            # Humanoid has ~376 dimensions, extract relevant motion features
            motion_features = obs[:200]  # First 200 features (joint positions/orientations)
        else:
            # For simpler environments, use all observations
            motion_features = obs

        # Pad or truncate to match MotionGPT's expected motion dimension (263)
        if len(motion_features) < 263:
            # Pad with zeros
            motion_features = np.pad(motion_features, (0, 263 - len(motion_features)))
        else:
            # Truncate
            motion_features = motion_features[:263]

        return motion_features.astype(np.float32)

    def _compute_motion_language_reward(self) -> float:
        """Compute reward based on motion-language alignment"""
        if len(self.motion_history) < 10:  # Need some motion history
            return 0.0

        try:
            # Convert motion history to tensor
            motion_sequence = np.array(self.motion_history[-20:])  # Last 20 frames
            motion_tensor = torch.from_numpy(motion_sequence).float()

            if motion_tensor.dim() == 2:
                motion_tensor = motion_tensor.unsqueeze(0)  # Add batch dim

            # Get motion embedding
            motion_embedding = self.motion_tokenizer.get_motion_embedding(motion_tensor)
            if motion_embedding.dim() == 3:
                motion_embedding = motion_embedding.mean(dim=1)  # Average over sequence, keep 2D
            if motion_embedding.dim() == 2 and motion_embedding.shape[0] == 1:
                motion_embedding = motion_embedding.squeeze(0)  # Remove batch dim if batch size is 1

            # Get instruction embedding  
            instruction_embedding = self._encode_instruction(self.current_instruction)

            # Ensure same device and dimension
            motion_embedding = motion_embedding.to(self.device)
            instruction_embedding = instruction_embedding.to(self.device)

            # Handle dimension mismatch - project motion to language space
            if motion_embedding.shape[-1] != instruction_embedding.shape[-1]:
                # Simple projection to match dimensions
                if motion_embedding.shape[-1] == 256 and instruction_embedding.shape[-1] == 512:
                    # Pad motion embedding to 512 dimensions
                    padding_size = 512 - 256
                    if motion_embedding.dim() == 1:
                        padding = torch.zeros(padding_size, device=self.device)
                        motion_embedding = torch.cat([motion_embedding, padding], dim=0)
                    else:
                        padding = torch.zeros(motion_embedding.shape[0], padding_size, device=self.device)
                        motion_embedding = torch.cat([motion_embedding, padding], dim=1)
                elif motion_embedding.shape[-1] == 512 and instruction_embedding.shape[-1] == 256:
                    # Truncate motion embedding to 256 dimensions
                    motion_embedding = motion_embedding[:256]
                else:
                    # Generic case: use linear projection
                    if not hasattr(self, 'motion_projector'):
                        self.motion_projector = torch.nn.Linear(
                            motion_embedding.shape[-1],
                            instruction_embedding.shape[-1]
                        ).to(self.device)
                    motion_embedding = self.motion_projector(motion_embedding)

            # Compute cosine similarity
            similarity = torch.cosine_similarity(
                motion_embedding.unsqueeze(0),
                instruction_embedding.unsqueeze(0),
                dim=1
            )
            if similarity.numel() == 1:
                similarity = similarity.item()
            else:
                similarity = similarity.mean().item()

            # Normalize to [0, 1] and scale
            reward = (similarity + 1) / 2 * self.reward_scale

            return reward

        except Exception as e:
            if "size" in str(e).lower() or "dimension" in str(e).lower():
                print(f"Dimension mismatch in reward computation: {e}")
            elif "device" in str(e).lower():
                print(f"Device mismatch in reward computation: {e}")
            else:
                print(f"Motion-language reward computation failed: {e}")
            return 0.0

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Extract and store motion features
        motion_features = self._extract_motion_features(obs)
        self.motion_history.append(motion_features)

        # Keep history bounded
        if len(self.motion_history) > self.max_history_length:
            self.motion_history.pop(0)

        # Compute motion-language alignment reward
        language_reward = self._compute_motion_language_reward()

        # Add to original reward
        total_reward = reward + language_reward

        # Add info
        info['language_reward'] = language_reward
        info['original_reward'] = reward
        info['instruction'] = self.current_instruction
        info['motion_history_length'] = len(self.motion_history)

        return obs, total_reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.motion_history = []

        # Add initial motion
        motion_features = self._extract_motion_features(obs)
        self.motion_history.append(motion_features)

        info['instruction'] = self.current_instruction
        return obs, info


class MotionLanguageAgent:
    """
    Main agent that combines MotionGPT tokenization with hierarchical RL
    """

    def __init__(self,
                 env_name: str = "Humanoid-v4",
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):

        self.env_name = env_name
        self.device = device

        # Initialize MotionGPT tokenizer
        print("Initializing MotionGPT tokenizer...")
        self.motion_tokenizer = MotionTokenizer(device=device)

        # Create base environment
        self.base_env = gym.make(env_name)

        # Wrap with language-motion rewards
        self.env = LanguageMotionRewardWrapper(
            self.base_env,
            self.motion_tokenizer,
            instruction="walk forward"
        )

        # Initialize RL agent
        self.rl_agent = None

        print(f"MotionLanguageAgent initialized for {env_name}")

    def create_training_environment(self, instruction: str = "walk forward"):
        """Create vectorized environment for training"""
        def make_env():
            env = gym.make(self.env_name)
            env = LanguageMotionRewardWrapper(
                env,
                self.motion_tokenizer,
                instruction=instruction,
                reward_scale=0.0  # Disabled langauge reward for now
            )
            return env

        return DummyVecEnv([make_env])

    def train(self,
              instruction: str = "walk forward",
              total_timesteps: int = 100000,
              save_path: str = "./checkpoints/",
              eval_freq: int = 10000):
        """Train the agent on a specific instruction"""

        print(f" Training on instruction: '{instruction}'")

        # Create training environment
        vec_env = self.create_training_environment(instruction)

        # Create RL agent
        self.rl_agent = PPO(
            "MlpPolicy",
            vec_env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            verbose=1,
            device=self.device,
            tensorboard_log="./logs/"
        )

        # Setup callbacks
        Path(save_path).mkdir(parents=True, exist_ok=True)

        checkpoint_callback = CheckpointCallback(
            save_freq=eval_freq,
            save_path=save_path,
            name_prefix=f"motion_lang_agent_{instruction.replace(' ', '_')}"
        )

        # Train the agent
        print(f"Starting training for {total_timesteps} timesteps...")
        self.rl_agent.learn(
            total_timesteps=total_timesteps,
            callback=[checkpoint_callback],
            progress_bar=True
        )

        # Save final model
        final_path = f"{save_path}/final_model_{instruction.replace(' ', '_')}.zip"
        self.rl_agent.save(final_path)
        print(f" Training completed! Model saved to: {final_path}")

        return final_path

    def evaluate(self,
                 instruction: str = "walk forward",
                 num_episodes: int = 10,
                 model_path: str = None,
                 render: bool = False):
        """Evaluate the agent on a specific instruction"""

        print(f" Evaluating on instruction: '{instruction}'")

        # Load model if path provided
        if model_path and Path(model_path).exists():
            if self.rl_agent is None:
                vec_env = self.create_training_environment(instruction)
                self.rl_agent = PPO.load(model_path, env=vec_env)
            else:
                self.rl_agent = PPO.load(model_path)
            print(f"Loaded model from: {model_path}")

        if self.rl_agent is None:
            print(" No trained agent available! Train first or provide model_path.")
            return {}

        # Create evaluation environment
        eval_env = gym.make(self.env_name)
        eval_env = LanguageMotionRewardWrapper(
            eval_env,
            self.motion_tokenizer,
            instruction=instruction
        )

        # Run evaluation episodes
        episode_rewards = []
        language_rewards = []
        episode_lengths = []

        for episode in range(num_episodes):
            obs, info = eval_env.reset()
            total_reward = 0
            total_language_reward = 0
            steps = 0

            while steps < 1000:  # Max episode length
                action, _ = self.rl_agent.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = eval_env.step(action)

                total_reward += reward
                total_language_reward += info.get('language_reward', 0)
                steps += 1

                if render and episode == 0:  # Render first episode
                    eval_env.render()

                if terminated or truncated:
                    break

            episode_rewards.append(total_reward)
            language_rewards.append(total_language_reward)
            episode_lengths.append(steps)

            print(f"Episode {episode+1}: Total={total_reward:.2f}, "
                  f"Language={total_language_reward:.2f}, Steps={steps}")

        eval_env.close()

        # Compute statistics
        results = {
            'instruction': instruction,
            'num_episodes': num_episodes,
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'mean_language_reward': np.mean(language_rewards),
            'mean_episode_length': np.mean(episode_lengths),
            'episode_rewards': episode_rewards,
            'language_rewards': language_rewards
        }

        print(f"\n Evaluation Results for '{instruction}':")
        print(f"  Mean Total Reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
        print(f"  Mean Language Reward: {results['mean_language_reward']:.2f}")
        print(f"  Mean Episode Length: {results['mean_episode_length']:.1f}")

        return results

    def demo_instruction_following(self,
                                   instructions: list = None,
                                   model_path: str = None,
                                   max_steps: int = 1000,  # 1000 Steps, longer video
                                   render: bool = True,
                                   save_video: bool = True): # Save as video

        """Demonstrate the agent following different instructions"""

        if instructions is None:
            instructions = ["walk forward", "turn left", "turn right", "stop"]

        print(f" Demonstrating instruction following...")

        # Load model if provided
        if model_path and Path(model_path).exists():
            vec_env = self.create_training_environment("walk forward")
            self.rl_agent = PPO.load(model_path, env=vec_env)
            print(f"Loaded model: {model_path}")

        if self.rl_agent is None:
            print(" No trained agent! Training a quick baseline...")
            self.train("walk forward", total_timesteps=10000)

        # Demo each instruction
        for instruction in instructions:
            print(f"\n Following: '{instruction}'")

            # Create demo environment with live viewing
            demo_env = gym.make(self.env_name, render_mode="human")
            demo_env = LanguageMotionRewardWrapper(
                demo_env,
                self.motion_tokenizer,
                instruction=instruction
            )

            # Create separate environment for video recording if needed
            if save_video:
                video_env = gym.make(self.env_name, render_mode="rgb_array")
                video_env = LanguageMotionRewardWrapper(
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
                print(f" Video saved: {video_path}")

            demo_env.close()
            if save_video:
                video_env.close()


def test_hierarchical_agent():
    """Test the hierarchical agent"""
    print(" Testing Hierarchical Motion-Language Agent")
    print("=" * 45)

    # Create agent
    agent = MotionLanguageAgent("Humanoid-v4")

    # Quick training test
    print("\n1. Quick training test...")
    model_path = agent.train(
        instruction="walk forward",
        total_timesteps=5000,  # Short for testing
        save_path="./test_checkpoints/"
    )

    # Evaluation test
    print("\n2. Evaluation test...")
    results = agent.evaluate(
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

    print("\n Hierarchical agent test completed")
    return agent


if __name__ == "__main__":
    agent = test_hierarchical_agent()
