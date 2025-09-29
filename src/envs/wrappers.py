import gymnasium as gym
import numpy as np
from gymnasium.wrappers import TimeLimit
import mujoco
from dataclasses import dataclass

class ActionScale(gym.ActionWrapper):
    """Map policy outputs in [-1,1] to env.action_space range."""
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.low = env.action_space.low
        self.high = env.action_space.high

    def action(self, act):
        act = np.clip(act, -1.0, 1.0)
        scaled = self.low + (act + 1.0) * 0.5 * (self.high - self.low)
        return np.clip(scaled, self.low, self.high)

class PDActionWrapper(gym.ActionWrapper):
    """
    Interpret policy outputs as target joint positions in [-1,1],
    convert to torques via PD: tau = kp*(q_target - q) - kd*qdot.
    Works even when actuators are torque motors.
    """
    def __init__(self, env: gym.Env, kp=60.0, kd=2.0):
        super().__init__(env)
        self.kp = kp
        self.kd = kd
        # map actuators -> joints -> qpos/qvel indices
        model = env.unwrapped.model
        # actuator_trnid gives (joint_id, parent) — we take joint id
        self.act_joint_ids = model.actuator_trnid[:, 0].astype(int)
        self.qpos_adr = model.jnt_qposadr[self.act_joint_ids].astype(int)
        self.qvel_adr = model.jnt_dofadr[self.act_joint_ids].astype(int)
        self._u = np.zeros(model.nu, dtype=np.float64)

    def action(self, target_pos_unit):
        target_pos_unit = np.clip(target_pos_unit, -1.0, 1.0)
        data = self.env.unwrapped.data
        # Current joint angles/velocities for actuated joints
        q = data.qpos[self.qpos_adr].copy()
        qd = data.qvel[self.qvel_adr].copy()
        # Convert unit targets to approximate joint ranges
        # Assume symmetric ~[-1,1] rad if range unknown
        # If you know per-joint ranges, replace with a table.
        q_target = target_pos_unit  # ~radians
        tau = self.kp * (q_target - q) - self.kd * qd
        self._u[:] = tau
        return np.clip(self._u, self.env.action_space.low, self.env.action_space.high)

class ObservationNorm(gym.ObservationWrapper):
    """Running mean/var normalization for observations."""
    def __init__(self, env: gym.Env, eps=1e-8, clip=10.0):
        super().__init__(env)
        self.eps, self.clip = eps, clip
        shape = env.observation_space.shape
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = eps

    def observation(self, obs):
        # update running stats (Welford-style)
        batch = np.asarray(obs, dtype=np.float64)
        delta = batch - self.mean
        self.count += 1.0
        self.mean += delta / self.count
        self.var += (batch - self.mean) * delta
        std = np.sqrt(self.var / self.count) + self.eps
        out = (batch - self.mean) / std
        return np.clip(out, -self.clip, self.clip)

@dataclass
class Curriculum:
    stand_steps: int = 10000   # steps to reward standing still/balance
    target_speed: float = 1.2  # m/s after stand curriculum
    speed_ramp_steps: int = 20000

class HumanoidShaping(gym.Wrapper):
    """
    Gentle reward shaping + early termination:
    1) Phase 1: stand (keep pelvis high & upright).
    2) Phase 2: walk towards target speed.
    """
    def __init__(self, env: gym.Env, curriculum: Curriculum = Curriculum()):
        super().__init__(env)
        self.cur = curriculum
        self.total_steps = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # Lightly stabilize initial state (avoid extreme random inits)
        qpos = self.unwrapped.data.qpos
        qvel = self.unwrapped.data.qvel
        # Set pelvis height reasonable if needed
        pelvis_id = self.unwrapped.model.joint("root").id if hasattr(self.unwrapped.model, "joint") else 0
        # keep whatever default; ensure tiny noise only:
        qpos[:] += np.random.uniform(-0.005, 0.005, size=qpos.shape)
        qvel[:] *= 0.0
        return obs, info

    def step(self, action):
        obs, rew, terminated, truncated, info = self.env.step(action)
        self.total_steps += 1

        data = self.unwrapped.data
        com_z = float(data.subtree_com[1][2])  # torso COM z (subtree 1 is usually torso)
        torso_y = float(data.site_xmat[0][3]) if data.site_xmat.size else 0.0  # fallback
        # Safer indicators:
        z = float(data.qpos[2])        # root z
        pitch = float(data.qpos[4])    # approximate root pitch (depends on model DOFs)

        # Early termination if fell
        fell = (z < 0.8) or (abs(pitch) > 0.7)
        if fell:
            terminated = True
            rew -= 5.0  # small penalty

        # Phase shaping
        if self.total_steps < self.cur.stand_steps:
            # Encourage standing upright
            upright = np.exp(-2.0 * (pitch ** 2))
            height = np.clip((z - 0.9) / 0.5, 0.0, 1.0)
            rew += 1.0 * upright + 0.5 * height
        else:
            # Ramp target speed from 0 to target_speed
            t = min(1.0, (self.total_steps - self.cur.stand_steps) / max(1, self.cur.speed_ramp_steps))
            v_target = t * self.cur.target_speed
            # forward vel along +x
            vx = float(self.unwrapped.data.qvel[0])
            speed_reward = 1.5 * np.exp(-((vx - v_target) ** 2) / 0.25)
            energy_penalty = 1e-3 * np.sum(np.square(action))
            rew += speed_reward - energy_penalty

        info["z"] = z
        info["pitch"] = pitch
        return obs, float(rew), bool(terminated), bool(truncated), info

def tune_mujoco_sim(env: gym.Env):
    """Adjust friction, solver, and dt to reduce slipping/jitter."""
    model = env.unwrapped.model
    # Friction: [sliding, torsional, rolling]
    fr = model.geom_friction.copy()
    fr[:, 0] = np.maximum(fr[:, 0], 1.5)  # sliding
    fr[:, 1] = np.maximum(fr[:, 1], 0.05) # torsional
    fr[:, 2] = np.maximum(fr[:, 2], 0.01) # rolling
    model.geom_friction[:] = fr
    # Contact parameters (softer penetration)
    model.opt.tolerance = max(model.opt.tolerance, 1e-8)
    model.opt.iterations = max(model.opt.iterations, 10)
    model.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
    model.opt.timestep = min(model.opt.timestep, 0.002)  # smaller dt = more stable
    return env

def make_humanoid_env(env_id: str = "Humanoid-v4", max_episode_steps: int = 1000):
    base = gym.make(env_id)
    base = TimeLimit(base, max_episode_steps)
    base = tune_mujoco_sim(base)
    # PD + scaling + obs-norm + shaping (order matters: PD acts on actions)
    env = PDActionWrapper(ActionScale(base), kp=60.0, kd=2.0)
    env = HumanoidShaping(env)
    env = ObservationNorm(env)
    return env
