# src/envs/humanoid_stable.py
"""
Stable Humanoid Environment Wrapper for Motion-Language Control

- Added reward_shaping flag to disable reward modification
- When reward_shaping=False, only handles termination (no reward conflicts)
- PD control and physics tuning still applied
"""

import math
from typing import Optional, Dict, Any, Tuple
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.wrappers import TimeLimit

try:
    import mujoco
except ImportError:
    mujoco = None


# ============================================================================
# Utility Functions
# ============================================================================

def quat_to_euler(qw: float, qx: float, qy: float, qz: float) -> Tuple[float, float, float]:
    """
    Convert quaternion to Euler angles (roll, pitch, yaw).

    Args:
        qw, qx, qy, qz: Quaternion components (w, x, y, z)

    Returns:
        Tuple of (roll, pitch, yaw) in radians
    """
    # Roll (x-axis rotation)
    sinr_cosp = 2 * (qw * qx + qy * qz)
    cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2 * (qw * qy - qz * qx)
    sinp = max(-1.0, min(1.0, sinp))
    pitch = math.asin(sinp)

    # Yaw (z-axis rotation)
    siny_cosp = 2 * (qw * qz + qx * qy)
    cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw


def get_uprightness(qw: float, qx: float, qy: float, qz: float) -> float:
    """
    Compute uprightness from quaternion (1.0 = perfectly upright).

    This is the z-component of the rotated up vector.
    """
    return 1 - 2 * (qx * qx + qy * qy)


# ============================================================================
# Humanoid Joint Configuration
# ============================================================================

class HumanoidJointConfig:
    """
    Configuration for Humanoid-v4 joints and actuators.

    Humanoid-v4 has 17 actuators in this order:
    0-2: abdomen (y, z, x rotations)
    3-5: right_hip (x, z, y rotations)
    6: right_knee
    7-9: left_hip (x, z, y rotations)
    10: left_knee
    11-12: right_shoulder (x, y rotations)
    13: right_elbow
    14-15: left_shoulder (x, y rotations)
    16: left_elbow
    """

    # Joint groups
    ABDOMEN = [0, 1, 2]
    RIGHT_HIP = [3, 4, 5]
    RIGHT_KNEE = [6]
    LEFT_HIP = [7, 8, 9]
    LEFT_KNEE = [10]
    RIGHT_SHOULDER = [11, 12]
    RIGHT_ELBOW = [13]
    LEFT_SHOULDER = [14, 15]
    LEFT_ELBOW = [16]

    # Functional groups
    CORE = ABDOMEN
    HIPS = RIGHT_HIP + LEFT_HIP
    KNEES = RIGHT_KNEE + LEFT_KNEE
    LEGS = HIPS + KNEES
    ARMS = RIGHT_SHOULDER + RIGHT_ELBOW + LEFT_SHOULDER + LEFT_ELBOW

    # Recommended PD gains per joint group
    GAINS = {
        'core': {'kp': 400.0, 'kd': 40.0},
        'hip': {'kp': 300.0, 'kd': 30.0},
        'knee': {'kp': 250.0, 'kd': 25.0},
        'shoulder': {'kp': 150.0, 'kd': 15.0},
        'elbow': {'kp': 100.0, 'kd': 10.0},
    }


# ============================================================================
# PD Controller Wrapper
# ============================================================================

class PDController(gym.ActionWrapper):
    """
    PD controller with per-joint tuning for stable humanoid control.

    Converts normalized actions [-1, 1] to joint torques using
    position-derivative control with joint-specific gains.
    """

    def __init__(
            self,
            env: gym.Env,
            action_scale: float = 0.4,
            smoothing: float = 0.7,
            use_joint_specific_gains: bool = True
    ):
        """
        Args:
            env: Base environment
            action_scale: Scale factor for action magnitudes (smaller = more stable)
            smoothing: Exponential smoothing factor for action filtering
            use_joint_specific_gains: Whether to use different gains per joint
        """
        super().__init__(env)

        m = env.unwrapped.model
        self.num_actuators = m.nu

        # Get actuator-joint mapping
        self.act_joint_ids = m.actuator_trnid[:, 0].astype(int)
        self.qpos_adr = m.jnt_qposadr[self.act_joint_ids].astype(int)
        self.qvel_adr = m.jnt_dofadr[self.act_joint_ids].astype(int)

        # Initialize gains
        self.kp = np.ones(self.num_actuators) * 200.0
        self.kd = np.ones(self.num_actuators) * 20.0

        if use_joint_specific_gains:
            self._setup_joint_specific_gains()

        self.action_scale = action_scale
        self.smoothing = smoothing

        # State for smoothing
        self.prev_action = None
        self._torque_buffer = np.zeros(self.num_actuators, dtype=np.float64)

    def _setup_joint_specific_gains(self):
        """Configure per-joint PD gains for stability."""
        cfg = HumanoidJointConfig
        gains = cfg.GAINS

        # Core (highest gains for stability)
        for i in cfg.CORE:
            if i < self.num_actuators:
                self.kp[i] = gains['core']['kp']
                self.kd[i] = gains['core']['kd']

        # Hips (high gains for stance)
        for i in cfg.HIPS:
            if i < self.num_actuators:
                self.kp[i] = gains['hip']['kp']
                self.kd[i] = gains['hip']['kd']

        # Knees
        for i in cfg.KNEES:
            if i < self.num_actuators:
                self.kp[i] = gains['knee']['kp']
                self.kd[i] = gains['knee']['kd']

        # Shoulders
        for i in cfg.RIGHT_SHOULDER + cfg.LEFT_SHOULDER:
            if i < self.num_actuators:
                self.kp[i] = gains['shoulder']['kp']
                self.kd[i] = gains['shoulder']['kd']

        # Elbows
        for i in cfg.RIGHT_ELBOW + cfg.LEFT_ELBOW:
            if i < self.num_actuators:
                self.kp[i] = gains['elbow']['kp']
                self.kd[i] = gains['elbow']['kd']

    def reset(self, **kwargs):
        """Reset smoothing state."""
        self.prev_action = None
        return self.env.reset(**kwargs)

    def action(self, action: np.ndarray) -> np.ndarray:
        """
        Convert normalized action to PD-controlled torques.

        Args:
            action: Normalized action in [-1, 1]

        Returns:
            Joint torques
        """
        # Clip input action
        action = np.clip(action, -1.0, 1.0)

        # Apply smoothing filter
        if self.prev_action is not None:
            action = self.smoothing * self.prev_action + (1 - self.smoothing) * action
        self.prev_action = action.copy()

        # Scale actions
        scaled_action = action * self.action_scale

        # Get current joint state
        d = self.env.unwrapped.data
        q = d.qpos[self.qpos_adr].copy()
        qd = d.qvel[self.qvel_adr].copy()

        # PD control: tau = kp * (target - current) - kd * velocity
        tau = self.kp * (scaled_action - q) - self.kd * qd

        # Additional velocity damping for high speeds
        high_vel_mask = np.abs(qd) > 3.0
        tau[high_vel_mask] -= 10.0 * qd[high_vel_mask]

        # Clip to actuator limits
        self._torque_buffer[:] = np.clip(
            tau,
            self.env.action_space.low,
            self.env.action_space.high
        )

        return self._torque_buffer


# ============================================================================
# Stability Reward Shaping Wrapper
# ============================================================================

class StabilityWrapper(gym.Wrapper):
    """
    Wrapper that adds stability-based termination and optional reward shaping.

    FIXED: Added reward_shaping flag to disable reward modification.
    When used with DirectMotionLanguageWrapper, set reward_shaping=False
    to avoid conflicting reward signals.

    Features:
    - Height maintenance rewards (optional)
    - Uprightness rewards (optional)
    - Early termination for falls (always active)
    """

    def __init__(
            self,
            env: gym.Env,
            target_height: float = 1.25,
            min_height: float = 0.8,
            max_pitch: float = 1.0,
            max_roll: float = 1.0,
            initial_noise_scale: float = 0.01,
            fall_penalty: float = 0.0,  # CHANGED: Default to 0 (let wrapper handle it)
            reward_shaping: bool = True,  # NEW: Flag to enable/disable reward modification
            reward_weights: Optional[Dict] = None,
    ):
        """
        Args:
            env: Base environment
            target_height: Target standing height for reward
            min_height: Minimum height before termination
            max_pitch: Maximum pitch angle before termination
            max_roll: Maximum roll angle before termination
            initial_noise_scale: Scale of noise added to initial state
            fall_penalty: Penalty applied on termination (only if reward_shaping=True)
            reward_shaping: Whether to modify rewards (set False when using with another reward wrapper)
            reward_weights: Weights for reward components
        """
        super().__init__(env)

        self.target_height = target_height
        self.min_height = min_height
        self.max_pitch = max_pitch
        self.max_roll = max_roll
        self.initial_noise_scale = initial_noise_scale
        self.fall_penalty = fall_penalty
        self.reward_shaping = reward_shaping  # NEW

        # Default reward weights
        self.reward_weights = reward_weights or {
            'height': 1.0,
            'upright': 1.0,
            'alive_bonus': 0.5,
        }

        # Episode tracking
        self.episode_steps = 0
        self.episode_return = 0.0

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset with optional initial state noise for robustness."""
        obs, info = self.env.reset(seed=seed, options=options)

        # Add small noise to initial state for robustness
        if self.initial_noise_scale > 0:
            m = self.unwrapped.model
            d = self.unwrapped.data

            # Add noise to joint positions (skip root position/orientation)
            qpos_noise = np.random.normal(0, self.initial_noise_scale, size=d.qpos.shape)
            qpos_noise[:7] = 0  # Don't perturb root
            d.qpos[:] = d.qpos + qpos_noise

            # Small velocity noise
            qvel_noise = np.random.normal(0, self.initial_noise_scale * 0.1, size=d.qvel.shape)
            d.qvel[:] = d.qvel + qvel_noise

            # Forward kinematics
            mujoco.mj_forward(m, d)

        # Get updated observation
        obs = self.unwrapped._get_obs()

        # Reset tracking
        self.episode_steps = 0
        self.episode_return = 0.0

        info['initial_height'] = float(d.qpos[2])

        return obs, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Step with stability termination and optional reward shaping."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.episode_steps += 1

        d = self.unwrapped.data

        # Extract state
        z = float(d.qpos[2])
        qw, qx, qy, qz = map(float, d.qpos[3:7])
        roll, pitch, yaw = quat_to_euler(qw, qx, qy, qz)
        uprightness = get_uprightness(qw, qx, qy, qz)

        # Check termination conditions (ALWAYS active)
        should_terminate = False
        termination_reason = None

        if z < self.min_height:
            should_terminate = True
            termination_reason = 'low_height'
        elif abs(pitch) > self.max_pitch:
            should_terminate = True
            termination_reason = 'excessive_pitch'
        elif abs(roll) > self.max_roll:
            should_terminate = True
            termination_reason = 'excessive_roll'

        if should_terminate:
            terminated = True
            info['termination_reason'] = termination_reason
            
            # Only apply fall penalty if reward shaping is enabled
            if self.reward_shaping:
                reward -= self.fall_penalty
                
        elif self.reward_shaping:
            # Only compute stability rewards if reward_shaping is enabled
            w = self.reward_weights

            # Height reward (Gaussian around target)
            height_error = abs(z - self.target_height)
            height_reward = np.exp(-3.0 * height_error ** 2)

            # Uprightness reward
            upright_reward = np.exp(-4.0 * (pitch ** 2 + roll ** 2))

            # Alive bonus
            alive_bonus = w['alive_bonus']

            # Combine
            stability_reward = (
                    w['height'] * height_reward +
                    w['upright'] * upright_reward +
                    alive_bonus
            )

            reward += stability_reward

            # Extra bonus for really good posture
            if z > self.target_height - 0.1 and abs(pitch) < 0.15 and abs(roll) < 0.15:
                reward += 1.0

        # Update tracking
        self.episode_return += reward

        # Add info (always include state info, regardless of reward_shaping)
        info.update({
            'height': z,
            'pitch': pitch,
            'roll': roll,
            'yaw': yaw,
            'uprightness': uprightness,
            'episode_steps': self.episode_steps,
            'episode_return': self.episode_return,
            'reward_shaping_active': self.reward_shaping,
        })

        return obs, float(reward), bool(terminated), bool(truncated), info


# ============================================================================
# Physics Tuning
# ============================================================================

def tune_physics(env: gym.Env, config: Optional[Dict] = None) -> gym.Env:
    """
    Apply physics tuning for stable humanoid simulation.

    Args:
        env: Base MuJoCo environment
        config: Optional configuration overrides

    Returns:
        Environment with tuned physics
    """
    config = config or {}
    m = env.unwrapped.model

    # Friction settings
    friction = m.geom_friction.copy()
    friction[:, 0] = config.get('tangent_friction', 1.5)  # Tangential
    friction[:, 1] = config.get('torsional_friction', 0.1)  # Torsional
    friction[:, 2] = config.get('rolling_friction', 0.05)  # Rolling
    m.geom_friction[:] = friction

    # Solver settings for accuracy
    m.opt.tolerance = config.get('solver_tolerance', 1e-10)
    m.opt.iterations = config.get('solver_iterations', 50)

    # Timestep (smaller = more stable but slower)
    m.opt.timestep = config.get('timestep', 0.002)

    # Joint damping
    if hasattr(m, 'dof_damping'):
        base_damping = config.get('joint_damping', 0.5)
        m.dof_damping[:] = np.maximum(m.dof_damping[:], base_damping)

    return env


# ============================================================================
# Factory Functions
# ============================================================================

def make_stable_humanoid(
        max_episode_steps: int = 1000,
        action_scale: float = 0.4,
        action_smoothing: float = 0.7,
        target_height: float = 1.25,
        min_height: float = 0.8,
        initial_noise: float = 0.01,
        reward_shaping: bool = True,  # NEW: Pass through to StabilityWrapper
        physics_config: Optional[Dict] = None,
        **kwargs
) -> gym.Env:
    """
    Create a stable humanoid environment with all wrappers applied.

    Args:
        max_episode_steps: Maximum steps per episode
        action_scale: Scale factor for actions
        action_smoothing: Smoothing factor for action filter
        target_height: Target standing height
        min_height: Minimum height before termination
        initial_noise: Scale of initial state noise
        reward_shaping: Whether to apply stability reward shaping
                       Set to False when using with DirectMotionLanguageWrapper
        physics_config: Physics tuning configuration
        **kwargs: Additional arguments for Humanoid-v4

    Returns:
        Wrapped environment
    """
    # Create base environment
    base = gym.make("Humanoid-v4", **kwargs)

    # Apply physics tuning
    env = tune_physics(base, physics_config)

    # Add PD controller
    env = PDController(
        env,
        action_scale=action_scale,
        smoothing=action_smoothing
    )

    # Add stability wrapper (with optional reward shaping)
    env = StabilityWrapper(
        env,
        target_height=target_height,
        min_height=min_height,
        initial_noise_scale=initial_noise,
        reward_shaping=reward_shaping,  # NEW: Can disable reward modification
    )

    # Add time limit
    env = TimeLimit(env, max_episode_steps=max_episode_steps)

    return env


def make_stable_humanoid_no_reward_shaping(
        max_episode_steps: int = 1000,
        action_scale: float = 0.4,
        action_smoothing: float = 0.7,
        target_height: float = 1.25,
        min_height: float = 0.8,
        initial_noise: float = 0.01,
        physics_config: Optional[Dict] = None,
        **kwargs
) -> gym.Env:
    """
    Convenience function: Create stable humanoid WITHOUT reward shaping.
    
    Use this when combining with DirectMotionLanguageWrapper to avoid
    conflicting reward signals.
    """
    return make_stable_humanoid(
        max_episode_steps=max_episode_steps,
        action_scale=action_scale,
        action_smoothing=action_smoothing,
        target_height=target_height,
        min_height=min_height,
        initial_noise=initial_noise,
        reward_shaping=False,  # Disable reward shaping
        physics_config=physics_config,
        **kwargs
    )


def make_wrapped_humanoid_env(**kwargs) -> gym.Env:
    """Entry point for gymnasium registry."""
    return make_stable_humanoid(**kwargs)


# ============================================================================
# Registration Helpers
# ============================================================================

def ensure_wrapped_humanoid_registered():
    """Register 'HumanoidStable-v4' if not already registered."""
    env_id = "HumanoidStable-v4"

    try:
        gym.spec(env_id)
        return  # Already registered
    except gym.error.NameNotFound:
        pass

    from gymnasium.envs.registration import register
    register(
        id=env_id,
        entry_point="envs.humanoid_stable:make_wrapped_humanoid_env",
        max_episode_steps=1000,
    )


def upgrade_env_name(name: str) -> str:
    """Convert standard env names to stable versions."""
    name_lower = name.lower()

    if 'humanoid' in name_lower and 'stable' not in name_lower:
        return "HumanoidStable-v4"

    return name


# ============================================================================
# Testing
# ============================================================================

if __name__ == "__main__":
    print("Testing HumanoidStable environment...")
    
    # Test WITH reward shaping
    print("\n=== Test 1: WITH reward shaping ===")
    env = make_stable_humanoid(max_episode_steps=100, reward_shaping=True)
    obs, info = env.reset()

    print(f"Observation shape: {obs.shape}")
    print(f"Action space: {env.action_space}")
    print(f"Initial height: {info.get('initial_height', 'N/A')}")

    total_reward = 0
    for i in range(50):
        action = env.action_space.sample() * 0.1
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated or truncated:
            print(f"Episode ended at step {i + 1}")
            print(f"Termination reason: {info.get('termination_reason', 'truncated')}")
            break

    print(f"Total reward (with shaping): {total_reward:.2f}")
    env.close()

    # Test WITHOUT reward shaping
    print("\n=== Test 2: WITHOUT reward shaping ===")
    env = make_stable_humanoid_no_reward_shaping(max_episode_steps=100)
    obs, info = env.reset()

    total_reward = 0
    for i in range(50):
        action = env.action_space.sample() * 0.1
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated or truncated:
            print(f"Episode ended at step {i + 1}")
            print(f"Termination reason: {info.get('termination_reason', 'truncated')}")
            break

    print(f"Total reward (no shaping): {total_reward:.2f}")
    print(f"Reward shaping active: {info.get('reward_shaping_active', 'N/A')}")
    env.close()

    print("\nTest complete!")
