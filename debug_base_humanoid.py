# debug_base_humanoid.py
"""
Debug what's terminating the base Humanoid-v4 environment
"""
import gymnasium as gym
import numpy as np

print("Testing base Humanoid-v4 termination conditions...\n")

# Create raw environment
env = gym.make('Humanoid-v4')

for episode in range(5):
    obs, info = env.reset()
    print(f"\n=== Episode {episode + 1} ===")
    
    for step in range(1000):
        # Use trained model's action or random
        action = env.action_space.sample() * 0.1  # Small random actions
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Get state
        try:
            qpos = env.unwrapped.data.qpos
            qvel = env.unwrapped.data.qvel
            
            z = float(qpos[2])
            quat = qpos[3:7]
            tilt = abs(1.0 - abs(quat[0]))
            
            # Check contacts
            contact_forces = env.unwrapped.data.cfrc_ext
            has_contact = np.any(np.abs(contact_forces) > 0.1)
            
            if step % 10 == 0:
                print(f"  Step {step}: z={z:.2f}, tilt={tilt:.3f}, contact={has_contact}")
            
            if terminated:
                print(f"\n  ❌ TERMINATED at step {step}")
                print(f"     Height: {z:.3f}")
                print(f"     Tilt: {tilt:.3f}")
                print(f"     Has contact: {has_contact}")
                print(f"     Info: {info}")
                
                # Check specific body parts
                print(f"\n     Body contacts:")
                for i, force in enumerate(contact_forces[:5]):  # First 5 bodies
                    if np.any(np.abs(force) > 0.1):
                        print(f"       Body {i}: {np.linalg.norm(force):.2f}")
                
                break
            
            if truncated:
                print(f"\n  ⏰ TRUNCATED at step {step}")
                break
        
        except Exception as e:
            print(f"Error: {e}")
            break
    
    if step >= 999:
        print(f"  ✓ Completed 1000 steps successfully!")

env.close()

print("\n" + "="*60)
print("ANALYSIS:")
print("If terminating early, it's likely due to:")
print("  1. Head/torso contact with ground")
print("  2. Base env's z-height check")
print("  3. Invalid state (NaN/Inf)")
print("\nWe need to wrap the env to prevent these terminations!")
