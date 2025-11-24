# train_humanoid_no_base_termination.py
"""
COMPLETELY override Humanoid-v4's built-in termination
Force it to only terminate on OUR conditions
"""
from src.agents.hierarchical_agent import EnhancedMotionLanguageAgent
from stable_baselines3 import PPO
import gymnasium as gym
import numpy as np
import os


class NoBaseTerminationWrapper(gym.Wrapper):
    """
    FORCE OVERRIDE: Ignore base Humanoid-v4's termination conditions
    Only terminate on OUR conditions
    """
    
    def __init__(self, env):
        super().__init__(env)
        self.env_name = getattr(env.unwrapped, 'spec', None)
        if self.env_name:
            self.env_name = self.env_name.id
        else:
            self.env_name = "Unknown"
        
        self.prev_qpos = None
        self.step_count = 0
        self.episode_count = 0
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # === FORCE OVERRIDE BASE TERMINATION ===
        # Ignore whatever the base env said about termination
        base_terminated = terminated
        terminated = False  # Reset it!
        
        self.step_count += 1
        
        if "Humanoid" not in self.env_name:
            return obs, reward, base_terminated, truncated, info
        
        try:
            qpos = self.unwrapped.data.qpos
            qvel = self.unwrapped.data.qvel
            
            z = float(qpos[2])  # Height
            quat = qpos[3:7]    # Orientation
            tilt = abs(1.0 - abs(quat[0]))
            
            # === OUR TERMINATION CONDITIONS (VERY LENIENT) ===
            
            # Only terminate if COMPLETELY fallen (on the ground)
            if z < 0.5:  # Very low threshold
                reward -= 10.0
                terminated = True
                info['termination_reason'] = 'on_ground'
                if self.step_count % 100 == 0 or self.step_count < 50:
                    print(f"  [Episode {self.episode_count}, Step {self.step_count}] Terminated: on ground (z={z:.2f})")
            
            # Only terminate if completely upside-down
            elif tilt > 0.6:  # Almost upside down
                reward -= 10.0
                terminated = True
                info['termination_reason'] = 'upside_down'
                if self.step_count % 100 == 0 or self.step_count < 50:
                    print(f"  [Episode {self.episode_count}, Step {self.step_count}] Terminated: upside down (tilt={tilt:.3f})")
            
            # Gentle penalties for bad posture (but DON'T terminate!)
            elif z < 0.9:  # Getting low
                reward -= (0.9 - z) * 2.0  # Proportional penalty
                info['low_height_penalty'] = True
            
            elif tilt > 0.3:  # Tilting
                reward -= tilt * 1.0  # Gentle penalty
                info['tilt_penalty'] = True
            
            # === JOINT MOVEMENT REWARDS ===
            if self.prev_qpos is not None and not terminated:
                # Leg joints (approximate indices)
                current_joints = qpos[7:19]
                prev_joints = self.prev_qpos[7:19]
                
                joint_change = np.abs(current_joints - prev_joints)
                joint_movement = np.mean(joint_change)
                
                # Reward active movement
                if joint_movement > 0.015:  # Good movement
                    movement_reward = min(joint_movement * 10.0, 2.0)
                    reward += movement_reward
                    info['joint_movement_reward'] = movement_reward
                elif joint_movement < 0.005:  # Too stiff
                    reward -= 0.5
                    info['stiff_penalty'] = True
                
                info['joint_movement'] = joint_movement
            
            # === FORWARD VELOCITY REWARD ===
            if len(qvel) > 0:
                vx = float(qvel[0])
                
                # Bonus for good forward speed
                if 0.3 < vx < 2.0:  # Good walking speed
                    reward += vx * 0.5
                    info['speed_bonus'] = vx * 0.5
            
            self.prev_qpos = qpos.copy()
            
            # Store metrics
            info['z_height'] = z
            info['tilt'] = tilt
            info['step_count'] = self.step_count
            info['base_terminated'] = base_terminated  # Track what base env wanted
            
            # Log progress occasionally
            if self.step_count % 100 == 0:
                print(f"  [Episode {self.episode_count}, Step {self.step_count}] z={z:.2f}, tilt={tilt:.3f}, reward={reward:.1f}")
        
        except Exception as e:
            print(f"  Error in wrapper: {e}")
            import traceback
            traceback.print_exc()
        
        return obs, reward, terminated, truncated, info
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.step_count = 0
        self.episode_count += 1
        
        if self.episode_count % 10 == 0:
            print(f"\n🔄 Starting Episode {self.episode_count}")
        
        try:
            self.prev_qpos = self.unwrapped.data.qpos.copy()
        except:
            self.prev_qpos = None
        
        return obs, info


print("="*70)
print("🚀 Training Humanoid with BASE TERMINATION COMPLETELY DISABLED")
print("="*70)
print("\nWe're taking full control of when episodes end!")
print("Base env terminations will be IGNORED.\n")

# Create agent
agent = EnhancedMotionLanguageAgent(
    'Humanoid-v4',
    device='cuda',
    use_stability_focus=True,
    motion_checkpoint='external/MotionGPT/prepare/deps/t2m/t2m/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/model/finest.tar'
)

# Add our override wrapper
original_make_single_env = agent.make_single_env

def make_single_env_override(*args, **kwargs):
    env = original_make_single_env(*args, **kwargs)
    env = NoBaseTerminationWrapper(env)  # FORCE OVERRIDE
    return env

agent.make_single_env = make_single_env_override

# Hyperparameters optimized for exploration
agent.training_config['learning_rate'] = 3e-4
agent.training_config['ent_coef'] = 0.15  # VERY HIGH exploration
agent.training_config['n_steps'] = 2048
agent.training_config['batch_size'] = 64
agent.training_config['n_epochs'] = 10
agent.training_config['gae_lambda'] = 0.95
agent.training_config['max_grad_norm'] = 0.5
agent.training_config['clip_range'] = 0.2
agent.training_config['vf_coef'] = 0.5

agent.policy_kwargs = {
    'log_std_init': 0.0,
    'ortho_init': False,
}

print("Configuration:")
print(f"  ✓ Base termination: DISABLED")
print(f"  ✓ Our termination: z < 0.5 OR tilt > 0.6 (VERY lenient!)")
print(f"  ✓ Joint movement rewards: ENABLED")
print(f"  ✓ Exploration: VERY HIGH (ent_coef=0.15)")
print(f"  ✓ Language reward weight: 0.25 (LOW - mechanics first!)")
print(f"  ✓ Training steps: 500,000\n")

os.makedirs('./humanoid_override', exist_ok=True)

print("Starting training...\n")

model = agent.train_on_instruction(
    'walk forward',
    total_timesteps=500000,
    language_reward_weight=0.25,  # Very low - learn mechanics first!
    save_path='./humanoid_override/',
    n_envs=4,
    use_multiprocessing=False
)

print("\n" + "="*70)
print("✅ Training complete!")
print("="*70)
print("\nEpisodes should now last 500+ steps!")
print("Check if the humanoid learned proper walking!")
