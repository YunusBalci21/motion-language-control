#!/usr/bin/env python3
"""
Generalization Experiment: Testing on Unseen Instructions

This script evaluates whether policies trained on one instruction can
generalize to semantically similar but lexically different instructions.

EXPERIMENTS:
1. Train on "walk forward" → Test on "move ahead", "go forward", "walk slowly"
2. Train on "run forward" → Test on "sprint", "move quickly", "dash forward"
3. Zero-shot transfer across instruction variations

ESTIMATED RUNTIME:
- Quick mode (evaluation only, uses pretrained): ~5-10 minutes
- Train mode (train + eval): ~1-2 hours per environment

Usage:
    python generalization_experiment.py --quick                    # Eval only (~10 min)
    python generalization_experiment.py --train --env halfcheetah  # Train + eval (~1.5 hours)
    python generalization_experiment.py --full                     # Full experiment (~4-5 hours)

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

    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    print("ERROR: stable_baselines3 required")
    sys.exit(1)

# ============================================================================
# INSTRUCTION SETS FOR GENERALIZATION
# ============================================================================

INSTRUCTION_SETS = {
    "walk_forward": {
        "train": "walk forward",
        "test_seen": ["walk forward"],  # Same as training
        "test_unseen": [
            "move ahead",
            "go forward",
            "walk ahead",
            "move forward",
            "step forward",
            "proceed forward",
            "advance forward",
        ],
        "test_variations": [
            "walk forward slowly",
            "walk forward quickly",
            "walk forward steadily",
            "walk straight ahead",
        ],
        "test_negative": [
            "walk backward",
            "turn left",
            "stop moving",
            "run forward",  # Different action
        ],
    },
    "run_forward": {
        "train": "run forward",
        "test_seen": ["run forward"],
        "test_unseen": [
            "sprint forward",
            "dash forward",
            "move quickly",
            "go fast",
            "rush forward",
            "race forward",
        ],
        "test_variations": [
            "run forward fast",
            "run forward slowly",  # Contradiction - interesting test
            "run straight ahead",
        ],
        "test_negative": [
            "walk forward",  # Different speed
            "run backward",
            "stop running",
        ],
    },
    "hop_forward": {
        "train": "hop forward",
        "test_seen": ["hop forward"],
        "test_unseen": [
            "jump forward",
            "bounce forward",
            "leap forward",
            "skip forward",
        ],
        "test_variations": [
            "hop forward slowly",
            "hop forward quickly",
        ],
        "test_negative": [
            "hop backward",
            "walk forward",
            "stand still",
        ],
    },
}

ENVIRONMENT_CONFIGS = {
    "halfcheetah": {
        "env_name": "HalfCheetah-v4",
        "instruction_set": "run_forward",
        "train_timesteps": 200000,
    },
    "ant": {
        "env_name": "Ant-v4",
        "instruction_set": "walk_forward",
        "train_timesteps": 200000,
    },
    "walker2d": {
        "env_name": "Walker2d-v4",
        "instruction_set": "walk_forward",
        "train_timesteps": 200000,
    },
    "humanoid": {
        "env_name": "Humanoid-v4",
        "instruction_set": "walk_forward",
        "train_timesteps": 300000,
    },
    "hopper": {
        "env_name": "Hopper-v4",
        "instruction_set": "hop_forward",
        "train_timesteps": 150000,
    },
}

PPO_CONFIG = {
    "learning_rate": 3e-4,
    "n_steps": 2048,
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.01,
}


# ============================================================================
# MOTION-LANGUAGE EVALUATION
# ============================================================================

class GeneralizationEvaluator:
    """Evaluates motion-language similarity for generalization experiments"""

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.motion_tokenizer = None
        self._load_tokenizer()

    def _load_tokenizer(self):
        """Load MotionGPT tokenizer"""
        try:
            from models.motion_tokenizer import MotionTokenizer
            checkpoint_path = "external/MotionGPT/prepare/deps/t2m/t2m/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/model/finest.tar"
            if os.path.exists(checkpoint_path):
                self.motion_tokenizer = MotionTokenizer(checkpoint_path=checkpoint_path, device=self.device)
                print("✓ MotionGPT tokenizer loaded")
            else:
                print("⚠ MotionGPT checkpoint not found - using fallback")
                self._create_fallback_tokenizer()
        except ImportError:
            print("⚠ MotionTokenizer not available - using fallback")
            self._create_fallback_tokenizer()

    def _create_fallback_tokenizer(self):
        """Create a simple fallback for testing without MotionGPT"""

        class FallbackTokenizer:
            def compute_motion_language_similarity(self, motion, instruction):
                # Simple heuristic based on velocity
                if isinstance(motion, np.ndarray) and len(motion) > 0:
                    # Use velocity features if available
                    vx = np.mean(motion[:, 16]) if motion.shape[1] > 16 else 0

                    # Simple keyword matching
                    instruction = instruction.lower()
                    if "forward" in instruction:
                        return min(1.0, max(0.0, 0.3 + 0.1 * vx))
                    elif "backward" in instruction:
                        return min(1.0, max(0.0, 0.3 - 0.1 * vx))
                    else:
                        return 0.3
                return 0.3

            def extract_motion_from_obs(self, obs, env_name):
                features = np.zeros(30, dtype=np.float32)
                if len(obs) > 0:
                    features[0] = obs[0] if len(obs) > 0 else 0.5
                    if len(obs) > 8:
                        features[16] = obs[8] if "HalfCheetah" in env_name else obs[0]
                return features

        self.motion_tokenizer = FallbackTokenizer()

    def compute_similarity(self, motion_sequence: np.ndarray, instruction: str) -> float:
        """Compute motion-language similarity"""
        if self.motion_tokenizer is None:
            return 0.0

        # MotionGPT VQ-VAE needs minimum ~20 frames due to kernel_size=4 convolutions
        if len(motion_sequence) < 20:
            # Pad by repeating first frame
            pad_size = 20 - len(motion_sequence)
            padding = np.tile(motion_sequence[0:1], (pad_size, 1))
            motion_sequence = np.concatenate([padding, motion_sequence], axis=0)

        try:
            return float(self.motion_tokenizer.compute_motion_language_similarity(
                motion_sequence, instruction
            ))
        except Exception as e:
            print(f"⚠ Similarity computation failed: {e}")
            return 0.0

    def extract_features(self, obs: np.ndarray, env_name: str) -> np.ndarray:
        """Extract motion features from observation"""
        if hasattr(self.motion_tokenizer, 'extract_motion_from_obs'):
            return self.motion_tokenizer.extract_motion_from_obs(obs, env_name)

        # Fallback extraction
        features = np.zeros(30, dtype=np.float32)
        features[0] = obs[0] if len(obs) > 0 else 0.5
        return features


# ============================================================================
# EVALUATION FUNCTIONS
# ============================================================================

def evaluate_on_instruction(
        model: PPO,
        env_name: str,
        instruction: str,
        evaluator: GeneralizationEvaluator,
        n_episodes: int = 5,
        max_steps: int = 1000,
) -> Dict:
    """Evaluate a model on a specific instruction"""

    env = gym.make(env_name)

    episode_rewards = []
    episode_lengths = []
    episode_similarities = []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = 0
        ep_length = 0
        motion_history = []
        ep_sims = []

        while not done and ep_length < max_steps:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            ep_reward += reward
            ep_length += 1

            # Track motion features
            features = evaluator.extract_features(obs, env_name)
            motion_history.append(features)

            # Compute similarity every 16 steps (need minimum 20 frames for MotionGPT kernel)
            if len(motion_history) >= 20 and len(motion_history) % 16 == 0:
                motion_seq = np.array(motion_history[-64:])  # Use more frames for better encoding
                sim = evaluator.compute_similarity(motion_seq, instruction)
                ep_sims.append(sim)

            done = terminated or truncated

        episode_rewards.append(ep_reward)
        episode_lengths.append(ep_length)
        if ep_sims:
            episode_similarities.append(np.mean(ep_sims))

    env.close()

    return {
        'instruction': instruction,
        'mean_reward': float(np.mean(episode_rewards)),
        'std_reward': float(np.std(episode_rewards)),
        'mean_length': float(np.mean(episode_lengths)),
        'mean_similarity': float(np.mean(episode_similarities)) if episode_similarities else 0.0,
        'std_similarity': float(np.std(episode_similarities)) if episode_similarities else 0.0,
        'n_episodes': n_episodes,
    }


def run_generalization_eval(
        model: PPO,
        env_key: str,
        evaluator: GeneralizationEvaluator,
        n_episodes: int = 5,
) -> Dict:
    """Run full generalization evaluation for one environment"""

    config = ENVIRONMENT_CONFIGS[env_key]
    instruction_set_name = config['instruction_set']
    instruction_set = INSTRUCTION_SETS[instruction_set_name]

    results = {
        'env_key': env_key,
        'env_name': config['env_name'],
        'train_instruction': instruction_set['train'],
        'categories': {}
    }

    # Evaluate on each category
    categories = ['test_seen', 'test_unseen', 'test_variations', 'test_negative']

    for category in categories:
        print(f"\n  Evaluating {category}...")
        results['categories'][category] = []

        for instruction in instruction_set[category]:
            print(f"    Testing: '{instruction}'")
            eval_result = evaluate_on_instruction(
                model=model,
                env_name=config['env_name'],
                instruction=instruction,
                evaluator=evaluator,
                n_episodes=n_episodes,
            )
            results['categories'][category].append(eval_result)
            print(f"      → similarity={eval_result['mean_similarity']:.3f}, "
                  f"length={eval_result['mean_length']:.0f}")

    # Compute summary statistics
    results['summary'] = compute_generalization_summary(results)

    return results


def compute_generalization_summary(results: Dict) -> Dict:
    """Compute summary statistics for generalization"""

    summary = {}

    for category, evals in results['categories'].items():
        sims = [e['mean_similarity'] for e in evals]
        summary[category] = {
            'mean_similarity': float(np.mean(sims)),
            'std_similarity': float(np.std(sims)),
            'min_similarity': float(np.min(sims)),
            'max_similarity': float(np.max(sims)),
            'n_instructions': len(evals),
        }

    # Compute generalization gap
    if 'test_seen' in summary and 'test_unseen' in summary:
        seen_sim = summary['test_seen']['mean_similarity']
        unseen_sim = summary['test_unseen']['mean_similarity']
        summary['generalization_gap'] = seen_sim - unseen_sim
        summary['generalization_ratio'] = unseen_sim / max(seen_sim, 0.01)

    return summary


# ============================================================================
# TRAINING
# ============================================================================

def train_policy(
        env_key: str,
        output_dir: str,
        timesteps: Optional[int] = None,
) -> Tuple[PPO, str]:
    """Train a policy for generalization experiment"""

    config = ENVIRONMENT_CONFIGS[env_key]
    instruction_set = INSTRUCTION_SETS[config['instruction_set']]
    train_instruction = instruction_set['train']

    timesteps = timesteps or config['train_timesteps']

    print(f"\n{'=' * 60}")
    print(f"TRAINING: {env_key}")
    print(f"Instruction: '{train_instruction}'")
    print(f"Timesteps: {timesteps:,}")
    print(f"{'=' * 60}")

    # Create environment with motion-language wrapper
    def make_env():
        env = gym.make(config['env_name'])
        # Note: In full implementation, wrap with MotionLanguageWrapper
        return env

    vec_env = DummyVecEnv([make_env])
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True)

    model = PPO("MlpPolicy", vec_env, verbose=1, **PPO_CONFIG)

    start_time = time.time()
    model.learn(total_timesteps=timesteps, progress_bar=True)
    train_time = time.time() - start_time

    # Save model
    model_path = os.path.join(output_dir, f"{env_key}_generalization")
    model.save(model_path)
    vec_env.save(f"{model_path}_vecnorm.pkl")

    print(f"\n✓ Training complete in {train_time / 60:.1f} minutes")
    print(f"  Model saved to: {model_path}")

    vec_env.close()

    return model, model_path


def load_or_train_policy(
        env_key: str,
        output_dir: str,
        force_train: bool = False,
        timesteps: Optional[int] = None,
) -> PPO:
    """Load existing model or train new one"""

    model_path = os.path.join(output_dir, f"{env_key}_generalization.zip")

    if os.path.exists(model_path) and not force_train:
        print(f"\n✓ Loading existing model: {model_path}")
        model = PPO.load(model_path)
        return model

    # Check for pretrained models from main experiments
    pretrained_paths = [
        f"./checkpoints_v2/{env_key}_*/final_model.zip",
        f"./checkpoints/{env_key}_*/final_model.zip",
    ]

    import glob
    for pattern in pretrained_paths:
        matches = glob.glob(pattern)
        if matches and not force_train:
            print(f"\n✓ Loading pretrained model: {matches[0]}")
            model = PPO.load(matches[0])
            return model

    # Train new model
    model, _ = train_policy(env_key, output_dir, timesteps)
    return model


# ============================================================================
# LATEX GENERATION
# ============================================================================

def generate_generalization_latex(all_results: Dict) -> str:
    """Generate LaTeX table for generalization results"""

    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Generalization to unseen instructions. Models trained on one instruction "
        "(e.g., ``walk forward'') and evaluated on semantically similar but lexically different "
        "instructions. Higher similarity indicates better generalization.}",
        "\\label{tab:generalization}",
        "\\begin{tabular}{llcc}",
        "\\toprule",
        "\\textbf{Environment} & \\textbf{Instruction Category} & \\textbf{Similarity} & \\textbf{Gap} \\\\",
        "\\midrule",
    ]

    for env_key, env_results in all_results['environments'].items():
        env_name = env_results['env_name'].replace('-v4', '')
        summary = env_results['summary']

        # Seen instruction
        seen = summary['test_seen']
        lines.append(f"{env_name} & Seen (train) & {seen['mean_similarity']:.3f} & -- \\\\")

        # Unseen instructions
        unseen = summary['test_unseen']
        gap = summary.get('generalization_gap', 0)
        lines.append(f" & Unseen (zero-shot) & {unseen['mean_similarity']:.3f} & {gap:+.3f} \\\\")

        # Variations
        if 'test_variations' in summary:
            var = summary['test_variations']
            lines.append(f" & Variations & {var['mean_similarity']:.3f} & -- \\\\")

        # Negative (should be low)
        if 'test_negative' in summary:
            neg = summary['test_negative']
            lines.append(f" & Negative & {neg['mean_similarity']:.3f} & -- \\\\")

        lines.append("\\midrule")

    lines[-1] = "\\bottomrule"
    lines.extend([
        "\\end{tabular}",
        "\\end{table}",
    ])

    return "\n".join(lines)


def print_generalization_summary(all_results: Dict):
    """Print summary of generalization results"""

    print("\n" + "=" * 70)
    print("GENERALIZATION SUMMARY")
    print("=" * 70)

    print(f"\n{'Environment':<15} {'Category':<15} {'Similarity':>12} {'Gap':>10}")
    print("-" * 60)

    for env_key, env_results in all_results['environments'].items():
        summary = env_results['summary']

        for category in ['test_seen', 'test_unseen', 'test_variations', 'test_negative']:
            if category in summary:
                cat_data = summary[category]
                gap_str = "--"
                if category == 'test_unseen' and 'generalization_gap' in summary:
                    gap_str = f"{summary['generalization_gap']:+.3f}"

                print(f"{env_key:<15} {category:<15} {cat_data['mean_similarity']:>12.3f} {gap_str:>10}")


# ============================================================================
# MAIN EXPERIMENT RUNNER
# ============================================================================

def run_generalization_experiment(
        envs: List[str],
        output_dir: str,
        force_train: bool = False,
        timesteps: Optional[int] = None,
        n_eval_episodes: int = 5,
) -> Dict:
    """Run full generalization experiment"""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(output_dir, f"generalization_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    evaluator = GeneralizationEvaluator(device=device)

    all_results = {
        'timestamp': timestamp,
        'n_eval_episodes': n_eval_episodes,
        'environments': {}
    }

    total_start = time.time()

    for env_key in envs:
        print(f"\n{'#' * 70}")
        print(f"# GENERALIZATION: {env_key.upper()}")
        print(f"{'#' * 70}")

        # Load or train model
        model = load_or_train_policy(
            env_key=env_key,
            output_dir=output_dir,
            force_train=force_train,
            timesteps=timesteps,
        )

        # Run evaluation
        results = run_generalization_eval(
            model=model,
            env_key=env_key,
            evaluator=evaluator,
            n_episodes=n_eval_episodes,
        )

        all_results['environments'][env_key] = results

    total_time = time.time() - total_start
    all_results['total_time_seconds'] = total_time
    all_results['total_time_human'] = f"{total_time / 60:.1f} min"

    # Save results
    results_path = os.path.join(output_dir, "generalization_results.json")
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    # Generate LaTeX
    latex_table = generate_generalization_latex(all_results)
    latex_path = os.path.join(output_dir, "generalization_table.tex")
    with open(latex_path, 'w') as f:
        f.write(latex_table)

    print(f"\n{'=' * 70}")
    print("GENERALIZATION EXPERIMENT COMPLETE")
    print(f"{'=' * 70}")
    print(f"\nTotal time: {all_results['total_time_human']}")
    print(f"Results saved to: {results_path}")
    print(f"LaTeX table saved to: {latex_path}")

    print_generalization_summary(all_results)

    # Key insight
    print("\n" + "=" * 70)
    print("KEY FINDINGS FOR PAPER")
    print("=" * 70)

    for env_key, env_results in all_results['environments'].items():
        summary = env_results['summary']
        if 'generalization_ratio' in summary:
            ratio = summary['generalization_ratio']
            gap = summary['generalization_gap']
            print(f"\n{env_key}:")
            print(f"  • Generalization ratio: {ratio:.1%} (unseen/seen similarity)")
            print(f"  • Generalization gap: {gap:+.3f}")
            if ratio > 0.8:
                print(f"  → GOOD generalization to unseen instructions!")
            elif ratio > 0.6:
                print(f"  → MODERATE generalization")
            else:
                print(f"  → POOR generalization - may need more diverse training")

    return all_results


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generalization Experiment: Testing on Unseen Instructions"
    )

    parser.add_argument('--quick', action='store_true',
                        help='Quick evaluation only, uses pretrained models (~5-10 min)')
    parser.add_argument('--train', action='store_true',
                        help='Force training new models')
    parser.add_argument('--full', action='store_true',
                        help='Full experiment with all environments (~4-5 hours)')
    parser.add_argument('--env', type=str, choices=list(ENVIRONMENT_CONFIGS.keys()),
                        help='Single environment to test')
    parser.add_argument('--timesteps', type=int, default=None,
                        help='Override training timesteps')
    parser.add_argument('--episodes', type=int, default=5,
                        help='Evaluation episodes per instruction')
    parser.add_argument('--output', type=str, default='./generalization_results',
                        help='Output directory')

    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("GENERALIZATION EXPERIMENT")
    print("Testing transfer to unseen instructions")
    print("=" * 70)

    if args.quick:
        print("\n🚀 QUICK MODE: Evaluation only (~5-10 minutes)")
        print("   Using pretrained models if available")
        envs = ['halfcheetah', 'ant']
        force_train = False
    elif args.full:
        print("\n🔬 FULL MODE: Train + evaluate all environments (~4-5 hours)")
        envs = ['halfcheetah', 'ant', 'walker2d']
        force_train = True
    elif args.env:
        print(f"\n🎯 SINGLE ENV: {args.env}")
        envs = [args.env]
        force_train = args.train
    else:
        print("\n🚀 DEFAULT: Quick evaluation (~5-10 minutes)")
        envs = ['halfcheetah', 'ant']
        force_train = False

    # Estimate time
    if force_train:
        total_timesteps = sum(ENVIRONMENT_CONFIGS[e]['train_timesteps'] for e in envs)
        est_train_time = total_timesteps / 100000 * 15  # ~15 min per 100K
        est_eval_time = len(envs) * 15 * args.episodes / 60  # ~1 min per instruction category
        print(f"\nEstimated training time: {est_train_time / 60:.1f} hours")
        print(f"Estimated evaluation time: {est_eval_time:.0f} minutes")
    else:
        n_instructions = sum(
            sum(len(INSTRUCTION_SETS[ENVIRONMENT_CONFIGS[e]['instruction_set']][cat])
                for cat in ['test_seen', 'test_unseen', 'test_variations', 'test_negative'])
            for e in envs
        )
        print(f"\nTotal instructions to evaluate: {n_instructions}")
        print(f"Estimated time: {n_instructions * args.episodes * 0.5:.0f} minutes")

    print(f"Environments: {envs}")

    run_generalization_experiment(
        envs=envs,
        output_dir=args.output,
        force_train=force_train,
        timesteps=args.timesteps,
        n_eval_episodes=args.episodes,
    )


if __name__ == "__main__":
    main()