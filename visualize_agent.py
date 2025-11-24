# visualize_agent.py
"""
Script to load the trained agent and generate a video of it walking.
This is the "Proof of Concept" visualization for your thesis.
"""

import sys
import os
import numpy as np
import torch
import gymnasium as gym
import imageio
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Add src to path
sys.path.append("src")

# Import your custom wrapper and agent class
from agents.hierarchical_agent import DirectMotionLanguageWrapper, MotionTokenizer
from envs.humanoid_stable import ensure_wrapped_humanoid_registered


def make_env(render_mode='rgb_array'):
    ensure_wrapped_humanoid_registered()
    env = gym.make("HumanoidStable-v4", render_mode=render_mode)

    # We need to recreate the wrapper to ensure observations match training
    # Note: We don't need the heavy MotionGPT encoder just for visualization,
    # but the wrapper expects a tokenizer.
    tokenizer = MotionTokenizer(device='cpu')  # Load on CPU for viz

    env = DirectMotionLanguageWrapper(
        env,
        tokenizer,
        instruction="walk forward",
        record_video=False  # We'll handle video manually here
    )
    return env


def visualize():
    print("=" * 60)
    print("🎥 GENERATING THESIS DEMO VIDEO")
    print("=" * 60)

    # 1. Paths
    model_path = "./humanoid_fixed/final_model_walk_forward.zip"
    vec_norm_path = "./humanoid_fixed/vecnormalize.pkl"
    video_filename = "thesis_demo_walk_forward.mp4"

    if not os.path.exists(model_path):
        print(f"❌ Error: Could not find model at {model_path}")
        return

    # 2. Setup Environment
    print("Loading environment...")
    # We use DummyVecEnv because the model expects a vectorized environment
    env = DummyVecEnv([lambda: make_env(render_mode='rgb_array')])

    # Load normalization stats if they exist (CRITICAL for PPO performance)
    if os.path.exists(vec_norm_path):
        print("Loading normalization stats...")
        env = VecNormalize.load(vec_norm_path, env)
        env.training = False  # Don't update stats during test
        env.norm_reward = False
    else:
        print("⚠ Warning: Normalization stats not found. Performance might be jittery.")

    # 3. Load Agent
    print(f"Loading model from {model_path}...")
    model = PPO.load(model_path, env=env, device='cpu')

    # 4. Run Simulation
    print("Running simulation...")
    obs = env.reset()
    frames = []

    total_reward = 0
    steps = 0

    # Run for up to 500 steps (10 seconds)
    for i in range(500):
        # Get action from agent
        action, _ = model.predict(obs, deterministic=True)

        # Execute step
        obs, reward, done, info = env.step(action)

        # Capture frame
        frame = env.envs[0].render()
        frames.append(frame)

        total_reward += reward
        steps += 1

        if (i + 1) % 50 == 0:
            print(f"  Step {i + 1}/500 (Reward: {total_reward[0]:.1f})")

        if done[0]:
            print("  Agent fell or completed episode!")
            obs = env.reset()  # Reset just to keep loop going if needed

    env.close()

    # 5. Save Video
    print(f"Saving video to {video_filename}...")
    imageio.mimsave(video_filename, frames, fps=30)
    print(f"✅ Video saved! Download '{video_filename}' to see your agent.")


if __name__ == "__main__":
    visualize()