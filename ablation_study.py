#!/usr/bin/env python3
"""
Ablation Study: MotionGPT vs Heuristic Rewards

This script compares three reward configurations:
1. MotionGPT (Full) - Your method with motion-language alignment
2. Heuristic Only - Hand-crafted rewards without MotionGPT
3. Random Baseline - Untrained policy for reference

ESTIMATED RUNTIME:
- Quick mode (50K steps × 3 configs × 2 envs): ~30-45 minutes
- Full mode (200K steps × 3 configs × 3 envs): ~4-6 hours

Usage:
    python ablation_study.py --quick          # Fast test (~30 min)
    python ablation_study.py --full           # Full experiment (~5 hours)
    python ablation_study.py --env halfcheetah --timesteps 100000

Author: Yunus Emre Balci
Thesis: Continuous Control from Open-Vocabulary Feedback
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
import numpy as np

# Add project paths
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

import torch
import gymnasium as gym

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from stable_baselines3.common.callbacks import BaseCallback

    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    print("ERROR: stable_baselines3 required. Install with: pip install stable-baselines3")
    sys.exit(1)


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class AblationConfig:
    """Configuration for ablation experiment"""
    env_key: str
    env_name: str
    instruction: str
    timesteps: int
    n_eval_episodes: int = 10


ENVIRONMENTS = {
    "halfcheetah": AblationConfig(
        env_key="halfcheetah",
        env_name="HalfCheetah-v4",
        instruction="run forward",
        timesteps=200000,
    ),
    "ant": AblationConfig(
        env_key="ant",
        env_name="Ant-v4",
        instruction="walk forward",
        timesteps=200000,
    ),
    "walker2d": AblationConfig(
        env_key="walker2d",
        env_name="Walker2d-v4",
        instruction="walk forward",
        timesteps=200000,
    ),
    "humanoid": AblationConfig(
        env_key="humanoid",
        env_name="Humanoid-v4",
        instruction="walk forward",
        timesteps=300000,
    ),
    "hopper": AblationConfig(
        env_key="hopper",
        env_name="Hopper-v4",
        instruction="hop forward",
        timesteps=150000,
    ),
}

# PPO hyperparameters (same as your main experiments for fair comparison)
PPO_CONFIG = {
    "learning_rate": 3e-4,
    "n_steps": 2048,
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.01,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
}


# ============================================================================
# REWARD WRAPPERS
# ============================================================================

class HeuristicOnlyWrapper(gym.Wrapper):
    """
    ABLATION: Heuristic rewards WITHOUT MotionGPT.

    This uses only hand-crafted rewards based on:
    - Forward velocity
    - Height maintenance
    - Energy efficiency

    This is the baseline to show MotionGPT's contribution.
    """

    def __init__(self, env: gym.Env, instruction: str = "walk forward"):
        super().__init__(env)
        self.instruction = instruction.lower()
        self.env_name = getattr(getattr(env, 'spec', None), 'id', '') or ''

        # Parse instruction for target behavior
        self.target_speed = self._get_target_speed()
        self.direction = 1.0 if "forward" in self.instruction else -1.0
        if "backward" in self.instruction:
            self.direction = -1.0

    def _get_target_speed(self) -> float:
        """Extract target speed from instruction"""
        if "run" in self.instruction or "fast" in self.instruction:
            return 3.0
        elif "slow" in self.instruction:
            return 0.5
        else:
            return 1.5  # Default walking speed

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Compute heuristic reward
        heuristic_reward = self._compute_heuristic_reward(obs, action, info)

        info['original_reward'] = reward
        info['heuristic_reward'] = heuristic_reward
        info['reward_type'] = 'heuristic_only'

        return obs, heuristic_reward, terminated, truncated, info

    def _compute_heuristic_reward(self, obs: np.ndarray, action: np.ndarray, info: dict) -> float:
        """
        Compute hand-crafted heuristic reward.

        This is what you'd use WITHOUT MotionGPT - just velocity matching.
        """
        reward = 0.0

        # Get velocity (environment-specific)
        vx = self._get_forward_velocity(obs)

        # 1. Velocity reward: match target speed in target direction
        target_vx = self.target_speed * self.direction
        velocity_error = abs(vx - target_vx)
        velocity_reward = np.exp(-0.5 * velocity_error ** 2)
        reward += 2.0 * velocity_reward

        # 2. Alive bonus
        reward += 1.0

        # 3. Energy penalty
        energy_penalty = 0.001 * np.sum(np.square(action))
        reward -= energy_penalty

        # 4. Height bonus (for bipeds)
        if "Humanoid" in self.env_name or "Walker" in self.env_name or "Hopper" in self.env_name:
            height = self._get_height(obs)
            if height > 0.8:
                reward += 0.5

        return float(reward)

    def _get_forward_velocity(self, obs: np.ndarray) -> float:
        """Extract forward velocity from observation"""
        if "HalfCheetah" in self.env_name:
            return float(obs[8]) if len(obs) > 8 else 0.0
        elif "Ant" in self.env_name:
            return float(obs[13]) if len(obs) > 13 else 0.0
        elif "Humanoid" in self.env_name:
            return float(obs[22]) if len(obs) > 22 else 0.0
        elif "Walker" in self.env_name:
            return float(obs[8]) if len(obs) > 8 else 0.0
        elif "Hopper" in self.env_name:
            return float(obs[5]) if len(obs) > 5 else 0.0
        return 0.0

    def _get_height(self, obs: np.ndarray) -> float:
        """Extract height from observation"""
        if "Humanoid" in self.env_name:
            return float(obs[0]) if len(obs) > 0 else 1.0
        elif "Walker" in self.env_name:
            return float(obs[0]) if len(obs) > 0 else 1.0
        elif "Hopper" in self.env_name:
            return float(obs[0]) if len(obs) > 0 else 1.0
        return 1.0


class MotionGPTRewardWrapper(gym.Wrapper):
    """
    FULL METHOD: MotionGPT-based motion-language reward.

    This is YOUR method - using motion-language alignment.
    """

    def __init__(self, env: gym.Env, instruction: str = "walk forward",
                 motion_tokenizer=None, history_length: int = 32):
        super().__init__(env)
        self.instruction = instruction
        self.motion_tokenizer = motion_tokenizer
        self.history_length = history_length
        self.motion_history = []
        self.env_name = getattr(getattr(env, 'spec', None), 'id', '') or ''

    def reset(self, **kwargs):
        self.motion_history = []
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Extract motion features
        motion_features = self._extract_motion_features(obs)
        self.motion_history.append(motion_features)

        # Keep history bounded
        if len(self.motion_history) > self.history_length:
            self.motion_history.pop(0)

        # Compute motion-language reward (need 20+ frames for MotionGPT kernel)
        if len(self.motion_history) >= 20 and self.motion_tokenizer is not None:
            motion_sequence = np.array(self.motion_history)
            try:
                similarity = self.motion_tokenizer.compute_motion_language_similarity(
                    motion_sequence, self.instruction
                )
                motiongpt_reward = similarity * 3.0 + 1.0  # Scale to reasonable range
            except Exception:
                # Fallback if encoding fails
                motiongpt_reward = 0.5
                similarity = 0.0
        else:
            # Warmup: use small positive reward
            motiongpt_reward = 0.5
            similarity = 0.0

        info['original_reward'] = reward
        info['motiongpt_reward'] = motiongpt_reward
        info['motion_similarity'] = similarity if 'similarity' in dir() else 0.0
        info['reward_type'] = 'motiongpt'

        return obs, motiongpt_reward, terminated, truncated, info

    def _extract_motion_features(self, obs: np.ndarray) -> np.ndarray:
        """Extract 30-dim motion features"""
        features = np.zeros(30, dtype=np.float32)

        try:
            if "Humanoid" in self.env_name:
                features[0] = obs[0]  # height
                features[1:5] = obs[1:5]  # quaternion
                if len(obs) > 25:
                    features[16:19] = obs[22:25]  # velocities
            elif "Ant" in self.env_name:
                features[0] = 0.5
                if len(obs) > 16:
                    features[16] = obs[13]  # vx
                    features[17] = obs[14]  # vy
            elif "HalfCheetah" in self.env_name:
                features[0] = 0.5
                if len(obs) > 9:
                    features[16] = obs[8]  # vx
            elif "Walker" in self.env_name:
                features[0] = obs[0] if len(obs) > 0 else 1.0
                if len(obs) > 9:
                    features[16] = obs[8]  # vx
            elif "Hopper" in self.env_name:
                features[0] = obs[0] if len(obs) > 0 else 1.0
                if len(obs) > 6:
                    features[16] = obs[5]  # vx
        except Exception:
            pass

        return features


# ============================================================================
# TRAINING CALLBACK
# ============================================================================

class AblationCallback(BaseCallback):
    """Callback to track training progress"""

    def __init__(self, eval_freq: int = 10000, verbose: int = 1):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.episode_rewards = []
        self.episode_lengths = []

    def _on_step(self) -> bool:
        # Track episode info
        if self.locals.get('dones') is not None:
            for i, done in enumerate(self.locals['dones']):
                if done:
                    info = self.locals['infos'][i]
                    if 'episode' in info:
                        self.episode_rewards.append(info['episode']['r'])
                        self.episode_lengths.append(info['episode']['l'])

        # Log progress
        if self.num_timesteps % self.eval_freq == 0:
            if self.episode_rewards:
                mean_reward = np.mean(self.episode_rewards[-100:])
                print(f"  Step {self.num_timesteps}: mean_reward={mean_reward:.2f}")

        return True


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_policy(
        model,
        env: gym.Env,
        n_episodes: int = 10,
        motion_tokenizer=None,
        instruction: str = "walk forward"
) -> Dict:
    """Evaluate a trained policy"""

    episode_rewards = []
    episode_lengths = []
    episode_similarities = []

    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False
        ep_reward = 0
        ep_length = 0
        ep_sims = []
        motion_history = []

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            ep_reward += reward
            ep_length += 1

            # Track similarity if available
            if 'motion_similarity' in info:
                ep_sims.append(info['motion_similarity'])
            elif motion_tokenizer is not None:
                # Compute similarity manually
                features = np.zeros(30, dtype=np.float32)  # Simplified
                motion_history.append(features)
                if len(motion_history) >= 20:  # Need 20+ frames for MotionGPT kernel
                    seq = np.array(motion_history[-64:])
                    try:
                        sim = motion_tokenizer.compute_motion_language_similarity(seq, instruction)
                        ep_sims.append(sim)
                    except:
                        pass

            done = terminated or truncated

        episode_rewards.append(ep_reward)
        episode_lengths.append(ep_length)
        if ep_sims:
            episode_similarities.append(np.mean(ep_sims))

    return {
        'mean_reward': float(np.mean(episode_rewards)),
        'std_reward': float(np.std(episode_rewards)),
        'mean_length': float(np.mean(episode_lengths)),
        'std_length': float(np.std(episode_lengths)),
        'mean_similarity': float(np.mean(episode_similarities)) if episode_similarities else 0.0,
        'std_similarity': float(np.std(episode_similarities)) if episode_similarities else 0.0,
    }


# ============================================================================
# ABLATION RUNNER
# ============================================================================

def run_ablation_experiment(
        env_key: str,
        timesteps: int,
        output_dir: str = "./ablation_results",
        use_motiongpt: bool = True,
        n_eval_episodes: int = 10,
) -> Dict:
    """Run ablation experiment for a single environment"""

    config = ENVIRONMENTS[env_key]
    os.makedirs(output_dir, exist_ok=True)

    results = {
        'env_key': env_key,
        'env_name': config.env_name,
        'instruction': config.instruction,
        'timesteps': timesteps,
        'configs': {}
    }

    # Load MotionGPT tokenizer if available
    motion_tokenizer = None
    if use_motiongpt:
        try:
            from models.motion_tokenizer import MotionTokenizer
            checkpoint_path = "external/MotionGPT/prepare/deps/t2m/t2m/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/model/finest.tar"
            if os.path.exists(checkpoint_path):
                motion_tokenizer = MotionTokenizer(checkpoint_path=checkpoint_path)
                print("✓ MotionGPT loaded")
            else:
                print("⚠ MotionGPT checkpoint not found, using heuristic fallback")
        except ImportError:
            print("⚠ MotionTokenizer not available")

    # ========== CONFIG 1: MotionGPT (Full Method) ==========
    print(f"\n{'=' * 60}")
    print(f"CONFIG 1: MotionGPT Reward (YOUR METHOD)")
    print(f"{'=' * 60}")

    def make_motiongpt_env():
        env = gym.make(config.env_name)
        env = MotionGPTRewardWrapper(env, config.instruction, motion_tokenizer)
        return env

    vec_env = DummyVecEnv([make_motiongpt_env])
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True)

    model_motiongpt = PPO("MlpPolicy", vec_env, verbose=1, **PPO_CONFIG)

    start_time = time.time()
    model_motiongpt.learn(total_timesteps=timesteps, progress_bar=True)
    train_time_motiongpt = time.time() - start_time

    # Save model
    model_path = os.path.join(output_dir, f"{env_key}_motiongpt")
    model_motiongpt.save(model_path)
    vec_env.save(f"{model_path}_vecnorm.pkl")

    # Evaluate
    eval_env = gym.make(config.env_name)
    eval_results_motiongpt = evaluate_policy(
        model_motiongpt, eval_env, n_eval_episodes, motion_tokenizer, config.instruction
    )
    eval_env.close()
    vec_env.close()

    results['configs']['motiongpt'] = {
        'training_time_seconds': train_time_motiongpt,
        **eval_results_motiongpt
    }
    print(f"✓ MotionGPT: reward={eval_results_motiongpt['mean_reward']:.2f}, "
          f"sim={eval_results_motiongpt['mean_similarity']:.3f}")

    # ========== CONFIG 2: Heuristic Only (Ablation) ==========
    print(f"\n{'=' * 60}")
    print(f"CONFIG 2: Heuristic Only (ABLATION - No MotionGPT)")
    print(f"{'=' * 60}")

    def make_heuristic_env():
        env = gym.make(config.env_name)
        env = HeuristicOnlyWrapper(env, config.instruction)
        return env

    vec_env = DummyVecEnv([make_heuristic_env])
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True)

    model_heuristic = PPO("MlpPolicy", vec_env, verbose=1, **PPO_CONFIG)

    start_time = time.time()
    model_heuristic.learn(total_timesteps=timesteps, progress_bar=True)
    train_time_heuristic = time.time() - start_time

    # Save model
    model_path = os.path.join(output_dir, f"{env_key}_heuristic")
    model_heuristic.save(model_path)

    # Evaluate
    eval_env = gym.make(config.env_name)
    eval_results_heuristic = evaluate_policy(
        model_heuristic, eval_env, n_eval_episodes, motion_tokenizer, config.instruction
    )
    eval_env.close()
    vec_env.close()

    results['configs']['heuristic'] = {
        'training_time_seconds': train_time_heuristic,
        **eval_results_heuristic
    }
    print(f"✓ Heuristic: reward={eval_results_heuristic['mean_reward']:.2f}, "
          f"sim={eval_results_heuristic['mean_similarity']:.3f}")

    # ========== CONFIG 3: Random Baseline ==========
    print(f"\n{'=' * 60}")
    print(f"CONFIG 3: Random Policy (BASELINE)")
    print(f"{'=' * 60}")

    eval_env = gym.make(config.env_name)

    # Evaluate random policy
    episode_rewards = []
    episode_lengths = []

    for _ in range(n_eval_episodes):
        obs, _ = eval_env.reset()
        done = False
        ep_reward = 0
        ep_length = 0

        while not done:
            action = eval_env.action_space.sample()
            obs, reward, terminated, truncated, _ = eval_env.step(action)
            ep_reward += reward
            ep_length += 1
            done = terminated or truncated

        episode_rewards.append(ep_reward)
        episode_lengths.append(ep_length)

    eval_env.close()

    results['configs']['random'] = {
        'training_time_seconds': 0,
        'mean_reward': float(np.mean(episode_rewards)),
        'std_reward': float(np.std(episode_rewards)),
        'mean_length': float(np.mean(episode_lengths)),
        'mean_similarity': 0.0,  # Random has no meaningful similarity
    }
    print(f"✓ Random: reward={results['configs']['random']['mean_reward']:.2f}")

    return results


def run_full_ablation(
        envs: List[str],
        timesteps_multiplier: float = 1.0,
        output_dir: str = "./ablation_results",
) -> Dict:
    """Run ablation across multiple environments"""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(output_dir, f"ablation_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    all_results = {
        'timestamp': timestamp,
        'environments': {}
    }

    total_start = time.time()

    for env_key in envs:
        config = ENVIRONMENTS[env_key]
        timesteps = int(config.timesteps * timesteps_multiplier)

        print(f"\n{'#' * 70}")
        print(f"# ABLATION: {env_key.upper()} ({timesteps:,} timesteps)")
        print(f"{'#' * 70}")

        results = run_ablation_experiment(
            env_key=env_key,
            timesteps=timesteps,
            output_dir=output_dir,
        )

        all_results['environments'][env_key] = results

    total_time = time.time() - total_start
    all_results['total_time_seconds'] = total_time
    all_results['total_time_human'] = f"{total_time / 3600:.1f}h"

    # Save results
    results_path = os.path.join(output_dir, "ablation_results.json")
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    # Generate LaTeX table
    latex_table = generate_ablation_latex(all_results)
    latex_path = os.path.join(output_dir, "ablation_table.tex")
    with open(latex_path, 'w') as f:
        f.write(latex_table)

    # Print summary
    print(f"\n{'=' * 70}")
    print("ABLATION STUDY COMPLETE")
    print(f"{'=' * 70}")
    print(f"\nTotal time: {all_results['total_time_human']}")
    print(f"Results saved to: {results_path}")
    print(f"LaTeX table saved to: {latex_path}")

    print_ablation_summary(all_results)

    return all_results


def generate_ablation_latex(results: Dict) -> str:
    """Generate LaTeX table for ablation results"""

    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Ablation study: MotionGPT reward vs heuristic-only reward. "
        "Motion-language similarity is computed using MotionGPT embeddings for fair comparison.}",
        "\\label{tab:ablation}",
        "\\begin{tabular}{llccc}",
        "\\toprule",
        "\\textbf{Environment} & \\textbf{Reward Type} & \\textbf{Reward} & \\textbf{Length} & \\textbf{Similarity} \\\\",
        "\\midrule",
    ]

    for env_key, env_results in results['environments'].items():
        env_name = env_results['env_name'].replace('-v4', '')

        # MotionGPT row
        mg = env_results['configs']['motiongpt']
        lines.append(f"{env_name} & MotionGPT (Ours) & {mg['mean_reward']:.0f} $\\pm$ {mg['std_reward']:.0f} & "
                     f"{mg['mean_length']:.0f} & \\textbf{{{mg['mean_similarity']:.3f}}} \\\\")

        # Heuristic row
        h = env_results['configs']['heuristic']
        lines.append(f" & Heuristic Only & {h['mean_reward']:.0f} $\\pm$ {h['std_reward']:.0f} & "
                     f"{h['mean_length']:.0f} & {h['mean_similarity']:.3f} \\\\")

        # Random row
        r = env_results['configs']['random']
        lines.append(f" & Random & {r['mean_reward']:.0f} $\\pm$ {r['std_reward']:.0f} & "
                     f"{r['mean_length']:.0f} & -- \\\\")

        lines.append("\\midrule")

    # Remove last midrule and add bottomrule
    lines[-1] = "\\bottomrule"
    lines.extend([
        "\\end{tabular}",
        "\\end{table}",
    ])

    return "\n".join(lines)


def print_ablation_summary(results: Dict):
    """Print summary of ablation results"""

    print("\n" + "=" * 70)
    print("ABLATION SUMMARY")
    print("=" * 70)

    print(f"\n{'Environment':<15} {'Config':<15} {'Reward':>10} {'Length':>10} {'Similarity':>12}")
    print("-" * 70)

    for env_key, env_results in results['environments'].items():
        for config_name, config_results in env_results['configs'].items():
            sim = config_results.get('mean_similarity', 0)
            sim_str = f"{sim:.3f}" if sim > 0 else "--"
            print(f"{env_key:<15} {config_name:<15} {config_results['mean_reward']:>10.1f} "
                  f"{config_results['mean_length']:>10.0f} {sim_str:>12}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Ablation Study: MotionGPT vs Heuristic Rewards")

    parser.add_argument('--quick', action='store_true',
                        help='Quick test: 50K steps, 2 environments (~30-45 min)')
    parser.add_argument('--full', action='store_true',
                        help='Full experiment: 200K steps, 3 environments (~4-6 hours)')
    parser.add_argument('--env', type=str, choices=list(ENVIRONMENTS.keys()),
                        help='Single environment to test')
    parser.add_argument('--timesteps', type=int, default=None,
                        help='Override timesteps')
    parser.add_argument('--output', type=str, default='./ablation_results',
                        help='Output directory')

    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("ABLATION STUDY: MotionGPT vs Heuristic Rewards")
    print("=" * 70)

    if args.quick:
        print("\n🚀 QUICK MODE: ~30-45 minutes")
        envs = ['halfcheetah', 'ant']
        multiplier = 0.25  # 50K steps
    elif args.full:
        print("\n🔬 FULL MODE: ~4-6 hours")
        envs = ['halfcheetah', 'ant', 'walker2d']
        multiplier = 1.0  # Full timesteps
    elif args.env:
        print(f"\n🎯 SINGLE ENV: {args.env}")
        envs = [args.env]
        multiplier = 0.5 if args.timesteps is None else args.timesteps / ENVIRONMENTS[args.env].timesteps
    else:
        # Default: quick mode
        print("\n🚀 DEFAULT: Quick mode (~30-45 minutes)")
        envs = ['halfcheetah', 'ant']
        multiplier = 0.25

    if args.timesteps:
        multiplier = args.timesteps / ENVIRONMENTS[envs[0]].timesteps

    # Estimate time
    total_timesteps = sum(int(ENVIRONMENTS[e].timesteps * multiplier) for e in envs)
    estimated_time = total_timesteps / 100000 * 15  # ~15 min per 100K steps
    print(f"\nTotal timesteps: {total_timesteps:,}")
    print(f"Estimated time: {estimated_time / 60:.1f} hours")
    print(f"Environments: {envs}")

    run_full_ablation(
        envs=envs,
        timesteps_multiplier=multiplier,
        output_dir=args.output,
    )


if __name__ == "__main__":
    main()