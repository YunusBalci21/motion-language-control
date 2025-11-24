# train_humanoid_fixed.py
from src.agents.hierarchical_agent import EnhancedMotionLanguageAgent
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
import gymnasium as gym
from gymnasium import ObservationWrapper
import numpy as np
import os


class OrientationCheckWrapper(gym.Wrapper):
    """Prevent humanoid from exploiting 'falling forward' for velocity"""

    def __init__(self, env):
        super().__init__(env)
        self.env_name = getattr(env.unwrapped, 'spec', None)
        if self.env_name:
            self.env_name = self.env_name.id
        else:
            self.env_name = "Unknown"

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Only apply to Humanoid
        if "Humanoid" not in self.env_name:
            return obs, reward, terminated, truncated, info

        try:
            # Get orientation from qpos
            qpos = self.unwrapped.data.qpos

            # Root height (z position)
            z = float(qpos[2])

            # Root orientation quaternion [qw, qx, qy, qz]
            quat = qpos[3:7]

            # Check if upright: qw should be close to 1 (upright), 0 (sideways/upside down)
            # Tilt metric: 0 = perfectly upright, 1 = completely tilted
            tilt = abs(1.0 - abs(quat[0]))

            # Penalize if too tilted (leaning forward/backward to exploit)
            if tilt > 0.3:  # Severely tilted
                reward -= 10.0
                terminated = True
                info['termination_reason'] = 'too_tilted'
            elif tilt > 0.15:  # Moderately tilted
                reward -= 2.0 * tilt  # Proportional penalty
                info['tilt_penalty'] = True

            # Also check height
            if z < 0.8:
                reward -= 5.0
                terminated = True
                info['termination_reason'] = 'fell_down'

            # Store metrics
            info['z_height'] = z
            info['tilt'] = tilt
            info['upright_score'] = 1.0 - tilt

        except Exception as e:
            # If we can't get orientation, just pass through
            pass

        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


print("🏃 Training Humanoid with ORIENTATION CHECKS and MotionGPT...")
print("This prevents 'falling forward plank' exploit!\n")

# Create agent with stability focus enabled AND MotionGPT checkpoint
agent = EnhancedMotionLanguageAgent(
    'Humanoid-v4',
    device='cuda',
    use_stability_focus=True,  # Enable stability rewards
    motion_checkpoint='external/MotionGPT/prepare/deps/t2m/t2m/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/model/finest.tar'
)

# Monkey-patch the make_single_env to add orientation wrapper
original_make_single_env = agent.make_single_env


def make_single_env_with_orientation(*args, **kwargs):
    env = original_make_single_env(*args, **kwargs)
    # Wrap with orientation checker AFTER motion-language wrapper
    env = OrientationCheckWrapper(env)
    return env


agent.make_single_env = make_single_env_with_orientation

# Use HUMANOID-specific hyperparameters
agent.training_config['learning_rate'] = 3e-4  # Moderate LR
agent.training_config['ent_coef'] = 0.01  # Lower entropy (humanoid is complex enough)
agent.training_config['n_steps'] = 2048
agent.training_config['batch_size'] = 64
# ***REMOVED***: agent.training_config['log_std_init'] = 0.0  # This doesn't belong here

# Add policy_kwargs separately for PPO initialization
agent.policy_kwargs = {
    'log_std_init': 0.0,  # Start with moderate exploration
    'ortho_init': False,
}

print("Training with:")
print(f"  - MotionGPT checkpoint loaded ✓")
print(f"  - Orientation checks (prevent tipping)")
print(f"  - Height checks (z > 0.8)")
print(f"  - Stability focus enabled")
print(f"  - Language reward weight: 0.90\n")

os.makedirs('./humanoid_fixed', exist_ok=True)

model = agent.train_on_instruction(
    'walk forward',
    total_timesteps=300000,
    language_reward_weight=0.90,
    save_path='./humanoid_fixed/',
    n_envs=4,
    use_multiprocessing=False
)

print("\n✅ Humanoid training complete with orientation checks!")
print("The agent should now WALK instead of falling forward!")