# src/envs/humanoid_stable.py
import math
from pathlib import Path
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import TimeLimit

try:
    import mujoco  # noqa: F401
except Exception:
    pass


# ---------- small utilities ----------

def _quat_to_euler_xyz(qw, qx, qy, qz):
    sinr_cosp = 2 * (qw * qx + qy * qz)
    cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2 * (qw * qy - qz * qx)
    sinp = max(-1.0, min(1.0, sinp))
    pitch = math.asin(sinp)

    siny_cosp = 2 * (qw * qz + qx * qy)
    cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw


# ---------- wrappers ----------

class StrongPDController(gym.ActionWrapper):
    """Much stronger PD controller with per-joint tuning for stability."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        m = env.unwrapped.model
        self.act_joint_ids = m.actuator_trnid[:, 0].astype(int)
        self.qpos_adr = m.jnt_qposadr[self.act_joint_ids].astype(int)
        self.qvel_adr = m.jnt_dofadr[self.act_joint_ids].astype(int)
        self._u = np.zeros(m.nu, dtype=np.float64)

        # MUCH stronger gains for stability
        # Different gains for different joint types
        self.kp = np.ones(m.nu) * 300.0  # Base high gain
        self.kd = np.ones(m.nu) * 30.0  # High damping

        # Even higher gains for core/hip joints (typically first few)
        self.kp[:6] = 500.0  # Core stability
        self.kd[:6] = 50.0

        # Lower gains for extremities (typically last few)
        self.kp[-4:] = 200.0
        self.kd[-4:] = 20.0

        # Target position filter for smoothing
        self.prev_target = None
        self.alpha = 0.8  # Smoothing factor

    def action(self, action):
        # Clip and smooth actions
        action = np.clip(action, -1.0, 1.0)

        # Apply smoothing filter to reduce jerkiness
        if self.prev_target is not None:
            action = self.alpha * action + (1 - self.alpha) * self.prev_target
        self.prev_target = action.copy()

        # Scale actions to smaller range for stability
        action = action * 0.5  # Reduce action magnitude

        d = self.env.unwrapped.data
        q = d.qpos[self.qpos_adr].copy()
        qd = d.qvel[self.qvel_adr].copy()

        # PD control with per-joint gains
        tau = self.kp * (action - q) - self.kd * qd

        # Additional damping for high velocities
        high_vel_damping = np.where(np.abs(qd) > 2.0, qd * 20.0, 0.0)
        tau -= high_vel_damping

        self._u[:] = tau
        return np.clip(self._u, self.env.action_space.low, self.env.action_space.high)


class ImprovedStabilityShaping(gym.Wrapper):
    """Enhanced stability with better balance control and anti-pitch compensation."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.episode_steps = 0
        self.initial_height = 1.4  # Approximate starting height
        self.prev_pitch = 0.0
        self.pitch_integral = 0.0  # For integral control

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        d = self.unwrapped.data

        # Much smaller initial noise - only on non-critical joints
        joint_start_idx = 7
        d.qpos[joint_start_idx:] += np.random.uniform(-0.002, 0.002,
                                                      size=d.qpos[joint_start_idx:].shape)
        d.qvel[:] = 0.0

        # Set a good initial posture if possible
        # This helps start from a stable configuration
        d.qpos[2] = 1.4  # Set good initial height
        d.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]  # Perfect upright quaternion

        self.episode_steps = 0
        self.prev_pitch = 0.0
        self.pitch_integral = 0.0

        return obs, info

    def step(self, action):
        # Anti-pitch compensation
        d = self.unwrapped.data
        qw, qx, qy, qz = map(float, d.qpos[3:7])
        roll, pitch, _ = _quat_to_euler_xyz(qw, qx, qy, qz)

        # If pitching forward, adjust hip joints to lean back
        if abs(pitch) > 0.1:
            # Simple proportional-integral control
            pitch_correction = -pitch * 0.3 - self.pitch_integral * 0.1
            # Apply correction to hip joints (typically indices 3-5)
            if len(action) > 5:
                action[3] += pitch_correction
                action[4] += pitch_correction
                action[5] += pitch_correction * 0.5

            # Update integral
            self.pitch_integral += pitch * 0.01
            self.pitch_integral = np.clip(self.pitch_integral, -0.5, 0.5)

        obs, rew, terminated, truncated, info = self.env.step(action)
        self.episode_steps += 1

        z = float(d.qpos[2])

        # Stricter termination conditions
        if z < 0.7 or abs(pitch) > 1.2 or abs(roll) > 1.2:
            terminated = True
            rew -= 20.0
        else:
            # Height maintenance reward
            height_error = abs(z - self.initial_height)
            height_reward = np.exp(-3.0 * height_error)

            # Uprightness reward (stronger weight on pitch)
            upright_reward = np.exp(-5.0 * (pitch ** 2)) * np.exp(-3.0 * (roll ** 2))

            # Velocity penalty (discourage fast movements initially)
            vel_penalty = 0.01 * np.sum(np.square(d.qvel))

            # Combine rewards
            rew += 2.0 * height_reward + 3.0 * upright_reward - vel_penalty

            # Extra reward for maintaining good posture
            if z > 1.3 and abs(pitch) < 0.2 and abs(roll) < 0.2:
                rew += 2.0

        info["root_z"] = z
        info["root_pitch"] = pitch
        info["root_roll"] = roll
        info["episode_steps"] = self.episode_steps

        self.prev_pitch = pitch

        return obs, float(rew), bool(terminated), bool(truncated), info


def aggressive_physics_tuning(env: gym.Env):
    """Very aggressive physics settings for maximum stability."""
    m = env.unwrapped.model

    # Maximum friction for no slipping
    fr = m.geom_friction.copy()
    fr[:, 0] = 3.0  # Very high tangential friction
    fr[:, 1] = 0.2  # Torsional friction
    fr[:, 2] = 0.1  # Rolling friction
    m.geom_friction[:] = fr

    # Better solver settings
    m.opt.tolerance = 1e-12
    m.opt.iterations = 100  # Many iterations
    m.opt.timestep = 0.001  # Smaller timestep for stability

    # Add significant joint damping
    if hasattr(m, 'dof_damping'):
        m.dof_damping[:] = np.maximum(m.dof_damping[:], 2.0)

    # Reduce joint limits slightly to prevent extreme poses
    if hasattr(m, 'jnt_range'):
        # Shrink joint ranges by 10% to avoid extremes
        ranges = m.jnt_range.copy()
        for i in range(len(ranges)):
            mid = (ranges[i, 0] + ranges[i, 1]) / 2
            span = ranges[i, 1] - ranges[i, 0]
            ranges[i, 0] = mid - span * 0.45
            ranges[i, 1] = mid + span * 0.45
        m.jnt_range[:] = ranges

    return env


# ---------- factory for the registered env ----------

def make_wrapped_humanoid_env(**kwargs):
    """Entry point used by gym registry."""
    base = gym.make("Humanoid-v4", **kwargs)
    base = aggressive_physics_tuning(base)

    # Use the stronger PD controller
    env = StrongPDController(base)
    env = ImprovedStabilityShaping(env)

    # Episode limit
    max_steps = kwargs.get("max_episode_steps", 1000)
    env = TimeLimit(env, max_episode_steps=max_steps)

    return env


# ---------- helpers used by your agent ----------

def ensure_wrapped_humanoid_registered():
    """Register 'HumanoidStable-v4' if not already registered."""
    try:
        gym.spec("HumanoidStable-v4")
        return  # already there
    except Exception:
        pass
    from gymnasium.envs.registration import register
    register(
        id="HumanoidStable-v4",
        entry_point="envs.humanoid_stable:make_wrapped_humanoid_env",
        max_episode_steps=1000,
    )


def upgrade_env_name(name: str) -> str:
    """Swap Humanoid-v4 → HumanoidStable-v4; otherwise return input."""
    if name.lower().startswith("humanoid"):
        return "HumanoidStable-v4"
    return name