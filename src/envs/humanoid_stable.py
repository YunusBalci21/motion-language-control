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

class ActionScale(gym.ActionWrapper):
    """Keep actions in [-1,1] and scale to actuator range."""
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.low = env.action_space.low
        self.high = env.action_space.high

    def action(self, act):
        act = np.clip(act, -1.0, 1.0)
        scaled = self.low + (act + 1.0) * 0.5 * (self.high - self.low)
        return np.clip(scaled, self.low, self.high)

class PDActionWrapper(gym.ActionWrapper):
    """Very light PD controller for joint targets (improves early stability)."""
    def __init__(self, env: gym.Env, kp=60.0, kd=2.0):
        super().__init__(env)
        self.kp, self.kd = kp, kd
        m = env.unwrapped.model
        self.act_joint_ids = m.actuator_trnid[:, 0].astype(int)
        self.qpos_adr = m.jnt_qposadr[self.act_joint_ids].astype(int)
        self.qvel_adr = m.jnt_dofadr[self.act_joint_ids].astype(int)
        self._u = np.zeros(m.nu, dtype=np.float64)

    def action(self, target_pos_unit):
        target_pos_unit = np.clip(target_pos_unit, -1.0, 1.0)
        d = self.env.unwrapped.data
        q = d.qpos[self.qpos_adr].copy()
        qd = d.qvel[self.qvel_adr].copy()
        tau = self.kp * (target_pos_unit - q) - self.kd * qd
        self._u[:] = tau
        return np.clip(self._u, self.env.action_space.low, self.env.action_space.high)

class HumanoidShaping(gym.Wrapper):
    """Gentle shaping: stand first, then ramp to a target speed, with fall detection."""
    def __init__(self, env: gym.Env, stand_steps=10000, target_speed=1.2, speed_ramp_steps=20000):
        super().__init__(env)
        self.total_steps = 0
        self.stand_steps = stand_steps
        self.target_speed = target_speed
        self.speed_ramp_steps = speed_ramp_steps

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        d = self.unwrapped.data
        d.qpos[:] += np.random.uniform(-0.003, 0.003, size=d.qpos.shape)
        d.qvel[:] = 0.0
        self.total_steps = 0
        return obs, info

    def step(self, action):
        obs, rew, terminated, truncated, info = self.env.step(action)
        self.total_steps += 1
        d = self.unwrapped.data
        z = float(d.qpos[2])
        qw, qx, qy, qz = map(float, d.qpos[3:7])
        _, pitch, _ = _quat_to_euler_xyz(qw, qx, qy, qz)

        # fall detection
        if (z < 0.8) or (abs(pitch) > 0.7):
            terminated = True
            rew -= 5.0

        # shaping
        if self.total_steps < self.stand_steps:
            upright = np.exp(-2.0 * (pitch ** 2))
            height = np.clip((z - 0.9) / 0.5, 0.0, 1.0)
            rew += 1.0 * upright + 0.5 * height
        else:
            t = min(1.0, (self.total_steps - self.stand_steps) / max(1, self.speed_ramp_steps))
            v_target = t * self.target_speed
            vx = float(d.qvel[0])
            rew += 1.2 * np.exp(-((vx - v_target) ** 2) / 0.25) - 1e-3 * np.sum(np.square(action))

        info["root_z"] = z
        info["root_pitch"] = pitch
        return obs, float(rew), bool(terminated), bool(truncated), info

def tune_mujoco_sim(env: gym.Env):
    """Friction & solver tweaks to reduce foot slip/instability."""
    m = env.unwrapped.model
    fr = m.geom_friction.copy()
    # increase lateral/rolling friction a bit
    fr[:, 0] = np.maximum(fr[:, 0], 1.5)
    fr[:, 1] = np.maximum(fr[:, 1], 0.05)
    fr[:, 2] = np.maximum(fr[:, 2], 0.01)
    m.geom_friction[:] = fr
    m.opt.tolerance = max(m.opt.tolerance, 1e-8)
    m.opt.iterations = max(m.opt.iterations, 10)
    m.opt.timestep = min(m.opt.timestep, 0.002)
    return env

# ---------- factory for the registered env ----------

def make_wrapped_humanoid_env(**kwargs):
    """Entry point used by gym registry."""
    base = gym.make("Humanoid-v4", **kwargs)
    base = tune_mujoco_sim(base)
    # Order: PD (optional), then shaping, then time limit
    env = PDActionWrapper(ActionScale(base))
    env = HumanoidShaping(env)
    # Ensure episode limit matches upstream default
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
    # IMPORTANT: entry_point must be 'envs.humanoid_stable:make_wrapped_humanoid_env'
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
