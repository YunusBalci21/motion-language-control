# train_humanoid_lenient.py
"""
Humanoid training with LENIENT termination to allow learning
The strict orientation checks were killing episodes too early!
"""
from src.agents.hierarchical_agent import EnhancedMotionLanguageAgent
from stable_baselines3 import PPO
import gymnasium as gym
import numpy as np
import os


class LenientOrientationWrapper(gym.Wrapper):
    """MUCH more lenient - only terminate on severe failures"""

    def __init__(self, env):
        super().__init__(env)
        self.env_name = getattr(env.unwrapped, 'spec', None)
        if self.env_name:
            self.env_name = self.env_name.id
        else:
            self.env_name = "Unknown"

        self.prev_qpos = None
        self.step_count = 0

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.step_count += 1

        if "Humanoid" not in self.env_name:
            return obs, reward, terminated, truncated, info

        try:
            qpos = self.unwrapped.data.qpos
            z = float(qpos[2])  # Height
            quat = qpos[3:7]  # Orientation

            # Tilt metric
            tilt = abs(1.0 - abs(quat[0]))

            # === MUCH MORE LENIENT CHECKS ===

            # Only terminate if REALLY fallen (not just tilted)
            if z < 0.6:  # Changed from 0.8 - allow lower height
                reward -= 5.0
                terminated = True
                info['termination_reason'] = 'fell_down'
                print(f"  [Step {self.step_count}] Terminated: fell down (z={z:.2f})")

            # Only terminate if COMPLETELY sideways/upside-down
            elif tilt > 0.5:  # Changed from 0.3 - much more lenient!
                reward -= 10.0
                terminated = True
                info['termination_reason'] = 'completely_fallen'
                print(f"  [Step {self.step_count}] Terminated: completely fallen (tilt={tilt:.3f})")

            # Gentle penalties for tilting (but don't terminate!)
            elif tilt > 0.25:
                reward -= 1.0 * tilt  # Small penalty
                info['tilt_penalty'] = True

            # === JOINT MOVEMENT REWARDS ===
            if self.prev_qpos is not None and not terminated:
                current_joints = qpos[7:19]
                prev_joints = self.prev_qpos[7:19]

                joint_change = np.abs(current_joints - prev_joints)
                joint_movement = np.mean(joint_change)

                # Reward movement
                if joint_movement > 0.01:
                    movement_reward = min(joint_movement * 5.0, 1.0)  # Cap at 1.0
                    reward += movement_reward
                    info['joint_movement_reward'] = movement_reward
                else:
                    # Very small penalty for stiff legs (not harsh)
                    reward -= 0.1
                    info['stiff_penalty'] = True

                info['joint_movement'] = joint_movement

            self.prev_qpos = qpos.copy()

            # Store metrics
            info['z_height'] = z
            info['tilt'] = tilt
            info['upright_score'] = max(0, 1.0 - tilt)

        except Exception as e:
            print(f"  Error in wrapper: {e}")
            pass

        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.step_count = 0
        try:
            self.prev_qpos = self.unwrapped.data.qpos.copy()
        except:
            self.prev_qpos = None
        return obs, info


print("🏃 Training Humanoid with LENIENT termination...")
print("Allowing more freedom to explore and learn!\n")

# Create agent
agent = EnhancedMotionLanguageAgent(
    'Humanoid-v4',
    device='cuda',
    use_stability_focus=True,
    motion_checkpoint='external/MotionGPT/prepare/deps/t2m/t2m/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/model/finest.tar'
)

# Add lenient wrapper
original_make_single_env = agent.make_single_env


def make_single_env_lenient(*args, **kwargs):
    env = original_make_single_env(*args, **kwargs)
    env = LenientOrientationWrapper(env)
    return env


agent.make_single_env = make_single_env_lenient

# Better hyperparameters
agent.training_config['learning_rate'] = 3e-4
agent.training_config['ent_coef'] = 0.1  # High exploration!
agent.training_config['n_steps'] = 2048
agent.training_config['batch_size'] = 64
agent.training_config['n_epochs'] = 10
agent.training_config['gae_lambda'] = 0.95
agent.training_config['max_grad_norm'] = 0.5
agent.training_config['clip_range'] = 0.2
agent.training_config['vf_coef'] = 0.5

agent.policy_kwargs = {
    'log_std_init': 0.0,  # Standard exploration
    'ortho_init': False,
}

print("Training with:")
print(f"  - MotionGPT checkpoint ✓")
print(f"  - LENIENT termination (z < 0.6, tilt < 0.5)")
print(f"  - Joint movement rewards")
print(f"  - HIGH exploration (ent_coef=0.1)")
print(f"  - Language reward weight: 0.30 (VERY REDUCED!)\n")

os.makedirs('./humanoid_lenient', exist_ok=True)

model = agent.train_on_instruction(
    'walk forward',
    total_timesteps=500000,
    language_reward_weight=0.30,  # Much lower! Let it learn mechanics first
    save_path='./humanoid_lenient/',
    n_envs=4,
    use_multiprocessing=False
)

print("\n✅ Training complete!")
print("Episodes should now last 200+ steps!")