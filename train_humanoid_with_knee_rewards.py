# train_humanoid_with_knee_rewards.py
"""
Humanoid training with KNEE MOVEMENT REWARDS
Prevents stiff-leg falling exploit
"""
from src.agents.hierarchical_agent import EnhancedMotionLanguageAgent
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
import gymnasium as gym
import numpy as np
import os


class JointMovementWrapper(gym.Wrapper):
    """Reward knee/hip movement to encourage actual walking"""

    def __init__(self, env):
        super().__init__(env)
        self.env_name = getattr(env.unwrapped, 'spec', None)
        if self.env_name:
            self.env_name = self.env_name.id
        else:
            self.env_name = "Unknown"

        # Track previous joint positions
        self.prev_qpos = None

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        if "Humanoid" not in self.env_name:
            return obs, reward, terminated, truncated, info

        try:
            qpos = self.unwrapped.data.qpos
            z = float(qpos[2])  # Height
            quat = qpos[3:7]  # Orientation

            # === ORIENTATION CHECKS ===
            tilt = abs(1.0 - abs(quat[0]))

            if tilt > 0.3:  # Severely tilted
                reward -= 10.0
                terminated = True
                info['termination_reason'] = 'too_tilted'
            elif tilt > 0.15:  # Moderately tilted
                reward -= 2.0 * tilt
                info['tilt_penalty'] = True

            if z < 0.8:  # Fell down
                reward -= 5.0
                terminated = True
                info['termination_reason'] = 'fell_down'

            # === JOINT MOVEMENT REWARDS ===
            # Humanoid-v4 joint indices (approximate):
            # joints 7-30 are the body joints (hips, knees, ankles, etc.)
            if self.prev_qpos is not None and not terminated:
                # Get leg joints (knees, hips, ankles)
                # Indices may vary, but typically:
                # Left hip: 7-9, Left knee: 10, Left ankle: 11-12
                # Right hip: 13-15, Right knee: 16, Right ankle: 17-18

                current_joints = qpos[7:19]  # Main leg joints
                prev_joints = self.prev_qpos[7:19]

                # Calculate joint velocity (change in position)
                joint_change = np.abs(current_joints - prev_joints)
                joint_movement = np.mean(joint_change)

                # Reward significant joint movement
                if joint_movement > 0.01:  # Moving joints
                    movement_reward = min(joint_movement * 10.0, 2.0)  # Cap at 2.0
                    reward += movement_reward
                    info['joint_movement_reward'] = movement_reward
                    info['joint_movement'] = joint_movement
                else:  # Stiff legs
                    reward -= 0.5  # Small penalty for not moving
                    info['stiff_penalty'] = True

            # Store current qpos for next step
            self.prev_qpos = qpos.copy()

            # Store metrics
            info['z_height'] = z
            info['tilt'] = tilt
            info['upright_score'] = 1.0 - tilt

        except Exception as e:
            pass

        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        try:
            self.prev_qpos = self.unwrapped.data.qpos.copy()
        except:
            self.prev_qpos = None
        return obs, info


print("🏃 Training Humanoid with JOINT MOVEMENT REWARDS + MotionGPT...")
print("This encourages knee/hip movement to prevent stiff-leg falling!\n")

# Create agent with MotionGPT
agent = EnhancedMotionLanguageAgent(
    'Humanoid-v4',
    device='cuda',
    use_stability_focus=True,
    motion_checkpoint='external/MotionGPT/prepare/deps/t2m/t2m/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/model/finest.tar'
)

# Monkey-patch to add joint movement wrapper
original_make_single_env = agent.make_single_env


def make_single_env_with_rewards(*args, **kwargs):
    env = original_make_single_env(*args, **kwargs)
    env = JointMovementWrapper(env)  # Add joint movement rewards
    return env


agent.make_single_env = make_single_env_with_rewards

# IMPROVED HYPERPARAMETERS
agent.training_config['learning_rate'] = 3e-4
agent.training_config['ent_coef'] = 0.05  # Increased from 0.01 for more exploration!
agent.training_config['n_steps'] = 2048
agent.training_config['batch_size'] = 64
agent.training_config['n_epochs'] = 10
agent.training_config['gae_lambda'] = 0.95
agent.training_config['max_grad_norm'] = 0.5

agent.policy_kwargs = {
    'log_std_init': -0.5,  # Start with some exploration
    'ortho_init': False,
}

print("Training with:")
print(f"  - MotionGPT checkpoint ✓")
print(f"  - Joint movement rewards (encourages knee bending)")
print(f"  - Stiff leg penalty")
print(f"  - Orientation checks")
print(f"  - HIGHER entropy (ent_coef=0.05 for exploration)")
print(f"  - Language reward weight: 0.50 (REDUCED!)\n")  # Reduced from 0.90

os.makedirs('./humanoid_knee_rewards', exist_ok=True)

model = agent.train_on_instruction(
    'walk forward',
    total_timesteps=500000,  # Increased to 500k
    language_reward_weight=0.50,  # REDUCED from 0.90!
    save_path='./humanoid_knee_rewards/',
    n_envs=4,
    use_multiprocessing=False
)

print("\n✅ Training complete!")
print("Check if the humanoid bends knees now!")