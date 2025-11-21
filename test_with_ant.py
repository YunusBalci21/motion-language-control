#!/usr/bin/env python3
"""
Test with Ant-v4 which is inherently more stable than Humanoid
WITH VIDEO RECORDING for professor review
"""

import os
import sys
from pathlib import Path
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import RecordVideo

if os.name == "nt":
    os.environ["MUJOCO_GL"] = "glfw"

# Add path for your agent
project_root = Path(__file__).parent
sys.path.append(str(project_root / "src" / "agents"))


def test_ant_stability():
    """Test Ant-v4 - should be much more stable"""
    print("Testing Ant-v4 (More Stable Alternative)")
    print("=" * 50)

    # Create video directory
    video_dir = Path("ant_demo_videos")
    video_dir.mkdir(exist_ok=True)

    # Create environment with video recording
    env = gym.make("Ant-v4", render_mode="rgb_array")
    env = RecordVideo(
        env,
        video_folder=str(video_dir),
        episode_trigger=lambda x: True,  # Record all episodes
        name_prefix="ant_stability_test"
    )

    for episode in range(2):
        print(f"\nEpisode {episode + 1}")
        obs, info = env.reset()

        for step in range(500):
            # Small random actions
            action = np.random.uniform(-0.2, 0.2, size=env.action_space.shape)
            obs, reward, terminated, truncated, info = env.step(action)

            if step % 100 == 0:
                print(f"  Step {step}: reward={reward:.3f}")

            if terminated:
                print(f"  Terminated at step {step}")
                break

        if not terminated:
            print(f"  Completed {step + 1} steps successfully!")

    env.close()
    print(f"\n✅ Videos saved to: {video_dir.absolute()}")


def test_training_with_ant():
    """Quick training test with Ant"""
    from agents.hierarchical_agent import EnhancedMotionLanguageAgent

    print("\nTesting training with Ant-v4...")

    # Create video directory for training
    video_dir = Path("ant_training_videos")
    video_dir.mkdir(exist_ok=True)

    agent = EnhancedMotionLanguageAgent("Ant-v4", use_stability_focus=False)

    # Quick training with video recording
    model_path = agent.train_on_instruction(
        instruction="walk forward",
        total_timesteps=10000,
        save_path="./ant_test/",
        n_envs=2,
        verbose=1
    )

    print(f"Training complete! Model saved to: {model_path}")

    # Now test the trained model and record it
    print("\nTesting trained model with video recording...")
    test_env = gym.make("Ant-v4", render_mode="rgb_array")
    test_env = RecordVideo(
        test_env,
        video_folder=str(video_dir),
        episode_trigger=lambda x: True,
        name_prefix="ant_trained_walk_forward"
    )

    # Load trained model and test
    from stable_baselines3 import PPO
    model = PPO.load(model_path)

    for episode in range(3):
        obs, info = test_env.reset()
        total_reward = 0

        for step in range(500):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = test_env.step(action)
            total_reward += reward

            if terminated or truncated:
                break

        print(f"Episode {episode + 1}: Total reward = {total_reward:.2f}, Steps = {step + 1}")

    test_env.close()
    print(f"\n✅ Training videos saved to: {video_dir.absolute()}")


if __name__ == "__main__":
    print("=" * 60)
    print("ANT-V4 STABILITY TEST WITH VIDEO RECORDING")
    print("=" * 60)

    test_ant_stability()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("Ant is much more stable than Humanoid!")
    print("You can train with Ant first to verify your motion-language pipeline works,")
    print("then return to debugging Humanoid stability later.")
    print("\nVideos have been recorded and saved for your professor.")

    train_ant = input("\nWould you also like to train Ant and record training results? (y/n): ")
    if train_ant.lower() == 'y':
        test_training_with_ant()
        print("\n" + "=" * 60)
        print("ALL COMPLETE!")
        print("=" * 60)
        print("Check these folders for videos to send to your professor:")
        print("  1. ant_demo_videos/ - Stability test videos")
        print("  2. ant_training_videos/ - Trained agent videos")
        print("=" * 60)