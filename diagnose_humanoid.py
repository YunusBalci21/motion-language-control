# diagnose_humanoid.py
"""
Diagnose why humanoid is falling
Visualize joint movements and MotionGPT similarity scores
"""

import gymnasium as gym
import numpy as np
import torch
from src.agents.hierarchical_agent import EnhancedMotionLanguageAgent
from stable_baselines3 import PPO

print("Loading trained model...")

# Load your trained model
model_path = "./humanoid_fixed/best_model.zip"  # Adjust path
model = PPO.load(model_path)

# Create environment
agent = EnhancedMotionLanguageAgent(
    'Humanoid-v4',
    device='cuda',
    use_stability_focus=True,
    motion_checkpoint='external/MotionGPT/prepare/deps/t2m/t2m/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/model/finest.tar'
)

env = agent.make_single_env(
    instruction='walk forward',
    language_reward_weight=0.90
)

print("\nRunning 5 episodes to diagnose behavior...\n")

for episode in range(5):
    obs, info = env.reset()
    episode_reward = 0
    steps = 0

    prev_qpos = None
    joint_movements = []
    heights = []
    tilts = []
    velocities = []
    similarities = []

    for step in range(1000):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        episode_reward += reward
        steps += 1

        # Get state info
        try:
            qpos = env.unwrapped.unwrapped.data.qpos  # Unwrap twice for wrappers
            qvel = env.unwrapped.unwrapped.data.qvel

            z = float(qpos[2])
            quat = qpos[3:7]
            tilt = abs(1.0 - abs(quat[0]))
            vx = float(qvel[0]) if len(qvel) > 0 else 0.0

            heights.append(z)
            tilts.append(tilt)
            velocities.append(vx)

            # Check joint movement
            if prev_qpos is not None:
                joint_change = np.abs(qpos[7:19] - prev_qpos[7:19])
                avg_movement = np.mean(joint_change)
                joint_movements.append(avg_movement)

            prev_qpos = qpos.copy()

            # Get similarity if available
            if 'language_similarity' in info:
                similarities.append(info['language_similarity'])

        except Exception as e:
            pass

        if terminated or truncated:
            break

    # Print diagnostics
    print(f"=== Episode {episode + 1} ===")
    print(f"  Steps: {steps}")
    print(f"  Total reward: {episode_reward:.1f}")
    print(f"  Avg height: {np.mean(heights):.2f} (should be ~1.2-1.4)")
    print(f"  Avg tilt: {np.mean(tilts):.3f} (should be <0.15)")
    print(f"  Avg velocity: {np.mean(velocities):.2f} (target: 0.5-1.0 m/s)")
    print(f"  Avg joint movement: {np.mean(joint_movements):.4f} (should be >0.01)")

    if similarities:
        print(f"  Avg MotionGPT similarity: {np.mean(similarities):.3f} (should be >0.5)")

    # Diagnosis
    print("  Diagnosis:")
    if np.mean(joint_movements) < 0.005:
        print("    ❌ PROBLEM: Joints barely moving (stiff legs!)")
    else:
        print("    ✓ Joints are moving")

    if np.mean(heights) < 1.0:
        print("    ❌ PROBLEM: Height too low (falling)")
    else:
        print("    ✓ Height OK")

    if np.mean(tilts) > 0.2:
        print("    ❌ PROBLEM: Too much tilt (falling forward)")
    else:
        print("    ✓ Orientation OK")

    if steps < 100:
        print("    ❌ PROBLEM: Episode ended too early")

    print()

env.close()

print("\n💡 RECOMMENDATIONS:")
print("  If joints barely moving → Use train_humanoid_with_knee_rewards.py")
print("  If falling forward → Reduce language_reward_weight to 0.3-0.5")
print("  If episode ends early → Check orientation wrapper thresholds")