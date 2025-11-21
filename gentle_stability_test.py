#!/usr/bin/env python3
"""
Gentle stability test with very small actions
Tests if the humanoid can balance with minimal disturbance
"""

import sys
import os
from pathlib import Path
import numpy as np
import gymnasium as gym

# Set MuJoCo GL first
if os.name == "nt":
    os.environ["MUJOCO_GL"] = "glfw"

# Add correct path
project_root = Path(__file__).parent
sys.path.append(str(project_root / "src"))

from envs.humanoid_stable import ensure_wrapped_humanoid_registered, make_wrapped_humanoid_env


def test_standing_with_minimal_actions():
    """Test standing with very small or zero actions"""
    print("Testing Humanoid Stability with Minimal Actions")
    print("=" * 50)

    ensure_wrapped_humanoid_registered()
    env = make_wrapped_humanoid_env(render_mode="human")

    # Test 1: Zero actions (pure balance)
    print("\nTest 1: Zero Actions (Pure Balance)")
    print("-" * 30)
    obs, info = env.reset()

    for step in range(500):
        # ZERO actions - just balance
        action = np.zeros(env.action_space.shape)

        obs, reward, terminated, truncated, info = env.step(action)

        if step % 100 == 0:
            z = info.get("root_z", 0)
            pitch = info.get("root_pitch", 0)
            print(f"  Step {step}: height={z:.3f}, pitch={pitch:.3f}, reward={reward:.3f}")

        if terminated:
            print(f"  FELL at step {step}! Height={info.get('root_z', 0):.3f}")
            break

    if not terminated:
        print(f"  SUCCESS! Balanced for {step + 1} steps with zero actions!")

    # Test 2: Tiny random actions
    print("\nTest 2: Tiny Random Actions")
    print("-" * 30)
    obs, info = env.reset()

    for step in range(500):
        # VERY small random actions
        action = np.random.uniform(-0.01, 0.01, size=env.action_space.shape)

        obs, reward, terminated, truncated, info = env.step(action)

        if step % 100 == 0:
            z = info.get("root_z", 0)
            pitch = info.get("root_pitch", 0)
            print(f"  Step {step}: height={z:.3f}, pitch={pitch:.3f}, reward={reward:.3f}")

        if terminated:
            print(f"  FELL at step {step}! Height={info.get('root_z', 0):.3f}")
            break

    if not terminated:
        print(f"  SUCCESS! Balanced for {step + 1} steps with tiny actions!")

    # Test 3: Simple standing controller
    print("\nTest 3: Simple Standing Controller")
    print("-" * 30)
    obs, info = env.reset()

    target_joints = np.zeros(env.action_space.shape)  # Target pose

    for step in range(500):
        # Simple PD-like controller to maintain upright pose
        action = target_joints * 0.1  # Small control towards neutral

        # Add tiny anti-pitch correction
        pitch = info.get("root_pitch", 0)
        if abs(pitch) > 0.1:
            # Lean back if pitching forward (adjust hip joints)
            if env.action_space.shape[0] > 5:
                action[3] -= pitch * 0.05
                action[4] -= pitch * 0.05

        obs, reward, terminated, truncated, info = env.step(action)

        if step % 100 == 0:
            z = info.get("root_z", 0)
            pitch = info.get("root_pitch", 0)
            print(f"  Step {step}: height={z:.3f}, pitch={pitch:.3f}, reward={reward:.3f}")

        if terminated:
            print(f"  FELL at step {step}! Height={info.get('root_z', 0):.3f}, Pitch={info.get('root_pitch', 0):.3f}")
            break

    if not terminated:
        print(f"  SUCCESS! Controller kept balance for {step + 1} steps!")

    env.close()

    print("\n" + "=" * 50)
    print("Test Summary:")
    print("If the humanoid can balance with zero/tiny actions,")
    print("the stability fixes are working correctly.")
    print("Next step: Train 'stand still' task first, then 'walk slowly'.")


if __name__ == "__main__":
    test_standing_with_minimal_actions()