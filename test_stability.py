#!/usr/bin/env python3
"""
Test script to verify humanoid stability after fixes
Run this first to ensure the humanoid can stand before training walking
"""

import sys
import os
from pathlib import Path
import numpy as np
import gymnasium as gym

# Set MuJoCo GL first
if os.name == "nt":
    os.environ["MUJOCO_GL"] = "glfw"

# Add correct path - envs is directly under src/
project_root = Path(__file__).parent
sys.path.append(str(project_root / "src"))

# Now we can import from envs
from envs.humanoid_stable import ensure_wrapped_humanoid_registered, make_wrapped_humanoid_env


def test_standing_stability():
    """Test if humanoid can stand still without falling"""
    print("Testing Humanoid Stability")
    print("=" * 50)

    # Register the stable environment
    ensure_wrapped_humanoid_registered()

    # Create environment
    env = make_wrapped_humanoid_env(render_mode="human")

    # Test multiple episodes
    for episode in range(3):
        print(f"\nEpisode {episode + 1}")
        print("-" * 30)

        obs, info = env.reset()
        total_reward = 0

        for step in range(500):
            # Use small random actions to test stability
            action = np.random.uniform(-0.1, 0.1, size=env.action_space.shape)

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            # Print status every 50 steps
            if step % 50 == 0:
                z = info.get("root_z", 0)
                pitch = info.get("root_pitch", 0)
                print(f"  Step {step}: height={z:.3f}, pitch={pitch:.3f}, reward={reward:.3f}")

            if terminated:
                print(f"  FELL at step {step}! Height={info.get('root_z', 0):.3f}")
                break

        if not terminated:
            print(f"  SUCCESS! Stood for {step + 1} steps")
            print(f"  Total reward: {total_reward:.2f}")

    env.close()
    print("\nTest complete!")


def test_slow_walking():
    """Test very slow forward walking"""
    print("\nTesting Slow Walking")
    print("=" * 50)

    ensure_wrapped_humanoid_registered()
    env = make_wrapped_humanoid_env(render_mode="human")

    obs, info = env.reset()

    for step in range(500):
        # Gentle forward walking actions
        # Higher values for hip and knee joints to walk forward
        action = np.zeros(env.action_space.shape)

        # Simple walking pattern (adjust indices based on your humanoid model)
        # These are example values - you may need to tune them
        if step % 40 < 20:
            action[3] = 0.2  # Right hip
            action[6] = -0.1  # Right knee
            action[9] = -0.2  # Left hip
            action[12] = 0.1  # Left knee
        else:
            action[3] = -0.2  # Right hip
            action[6] = 0.1  # Right knee
            action[9] = 0.2  # Left hip
            action[12] = -0.1  # Left knee

        # Keep upper body stable
        action[0:3] *= 0.5  # Reduce upper body movements

        obs, reward, terminated, truncated, info = env.step(action)

        if step % 50 == 0:
            z = info.get("root_z", 0)
            print(f"Step {step}: height={z:.3f}, reward={reward:.3f}")

        if terminated:
            print(f"Fell at step {step}")
            break

    env.close()


if __name__ == "__main__":
    # First test standing
    test_standing_stability()

    # Then test walking
    input("\nPress Enter to test slow walking...")
    test_slow_walking()