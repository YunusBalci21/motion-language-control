#!/usr/bin/env python3
"""
train_all_envs_v2.py - Improved Training Pipeline

CHANGES FROM v1:
- Higher entropy (0.005 → 0.02) for more exploration
- Higher learning rate (1e-4 → 3e-4) for faster learning
- Larger batch size (64 → 128) for more stable gradients
- More steps per update (2048 → 4096) for better value estimates
- Adjusted reward weights to prioritize forward motion

Author: Yunus Emre Balci
"""

import sys
import os
import json
import argparse
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import numpy as np

# Add src to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

import torch

CUDA_AVAILABLE = torch.cuda.is_available()
if CUDA_AVAILABLE:
    GPU_NAME = torch.cuda.get_device_name(0)
    GPU_MEM = torch.cuda.get_device_properties(0).total_memory / 1e9
else:
    GPU_NAME = "N/A"
    GPU_MEM = 0

from agents.hierarchical_agent import EnhancedMotionLanguageAgent

try:
    import matplotlib.pyplot as plt
    import matplotlib

    matplotlib.use('Agg')
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

# ============================================================================
# CONFIGURATION
# ============================================================================

# MotionGPT checkpoint
MOTION_CHECKPOINT = "./external/MotionGPT/prepare/deps/t2m/t2m/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/model/finest.tar"

# IMPROVED HYPERPARAMETERS
IMPROVED_TRAINING_CONFIG = {
    "learning_rate": 3e-4,  # ↑ Was 1e-4 - faster learning
    "n_steps": 4096,  # ↑ Was 2048 - more data per update
    "batch_size": 128,  # ↑ Was 64 - more stable gradients
    "n_epochs": 10,  # ↑ Was 5 - more passes over data
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,  # ↑ Was 0.1 - allow larger policy updates
    "ent_coef": 0.02,  # ↑↑ Was 0.005 - MUCH more exploration
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "target_kl": 0.05,  # ↑ Was 0.03 - allow more KL divergence
}

# Environment configs
ENVIRONMENT_CONFIGS = {
    "humanoid": {
        "env_name": "Humanoid-v4",
        "instruction": "walk forward",
        "timesteps": 1_500_000,    
        "n_envs": 4,
        "description": "27-DoF humanoid robot walking",
    },
    "ant": {
        "env_name": "Ant-v4",
        "instruction": "walk forward",
        "timesteps": 800_000,      
        "n_envs": 4,
        "description": "4-legged ant locomotion",
    },
    "halfcheetah": {
        "env_name": "HalfCheetah-v4",
        "instruction": "run forward",
        "timesteps": 800_000,      
        "n_envs": 4,
        "description": "2D running cheetah",
    },
    "walker2d": {
        "env_name": "Walker2d-v4",
        "instruction": "walk forward",
        "timesteps": 700_000,      
        "n_envs": 4,
        "description": "2D bipedal walker",
    },
    "hopper": {
        "env_name": "Hopper-v4",
        "instruction": "hop forward",
        "timesteps": 500_000,      
        "n_envs": 4,
        "description": "Single-legged hopper",
    },
}

QUICK_TIMESTEPS = {
    "humanoid": 100_000,
    "ant": 50_000,
    "halfcheetah": 50_000,
    "walker2d": 50_000,
    "hopper": 30_000,
}

# Output directories
BASE_CHECKPOINT_DIR = "./checkpoints_v2"
RESULTS_DIR = "./thesis_results"
PLOTS_DIR = "./thesis_results/plots"
VIDEOS_DIR = "./thesis_results/videos"


# ============================================================================
# TRAINING WITH IMPROVED HYPERPARAMETERS
# ============================================================================

def train_environment(
        env_key: str,
        config: Dict,
        checkpoint_dir: str,
        device: str = "cuda",
        quick_mode: bool = False,
) -> Tuple[str, Dict]:
    """Train a single environment with improved hyperparameters."""

    env_name = config["env_name"]
    instruction = config["instruction"]
    timesteps = QUICK_TIMESTEPS[env_key] if quick_mode else config["timesteps"]
    n_envs = config["n_envs"]

    print(f"\n{'=' * 70}")
    print(f"🤖 TRAINING: {env_name} (IMPROVED HYPERPARAMETERS)")
    print(f"{'=' * 70}")
    print(f"  Instruction:    '{instruction}'")
    print(f"  Timesteps:      {timesteps:,}")
    print(f"  Parallel envs:  {n_envs}")
    print(f"  Device:         {device}")
    print(f"  Save path:      {checkpoint_dir}")
    print(f"  ")
    print(f"  HYPERPARAMETER CHANGES:")
    print(f"    learning_rate: 3e-4 (was 1e-4)")
    print(f"    ent_coef:      0.02 (was 0.005) ← MORE EXPLORATION")
    print(f"    clip_range:    0.2  (was 0.1)")
    print(f"    n_steps:       4096 (was 2048)")
    print(f"    batch_size:    128  (was 64)")
    print(f"{'=' * 70}\n")

    start_time = time.time()

    # Initialize agent
    agent = EnhancedMotionLanguageAgent(
        env_name=env_name,
        device=device,
        motion_checkpoint=MOTION_CHECKPOINT,
        use_stability_focus=True,
    )

    # OVERRIDE with improved hyperparameters
    agent.training_config = IMPROVED_TRAINING_CONFIG.copy()

    print("✓ Applied improved hyperparameters")

    # Train
    try:
        model_path = agent.train_on_instruction(
            instruction=instruction,
            total_timesteps=timesteps,
            n_envs=n_envs,
            save_path=checkpoint_dir,
            record_training_videos=False,
        )
    except KeyboardInterrupt:
        print("\n⚠ Training interrupted by user")
        model_path = f"{checkpoint_dir}/final_model.zip"
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return None, {"error": str(e)}

    training_time = time.time() - start_time

    training_info = {
        "env_name": env_name,
        "env_key": env_key,
        "instruction": instruction,
        "timesteps": timesteps,
        "training_time_seconds": training_time,
        "training_time_human": format_time(training_time),
        "model_path": model_path,
        "hyperparameters": IMPROVED_TRAINING_CONFIG,
    }

    print(f"\n✅ {env_name} training complete!")
    print(f"   Time: {format_time(training_time)}")
    print(f"   Model: {model_path}")

    return model_path, training_info


def train_all_environments(
        envs_to_train: List[str],
        device: str = "cuda",
        quick_mode: bool = False,
) -> Dict:
    """Train multiple environments with improved hyperparameters."""

    print("\n" + "=" * 70)
    print("🚀 IMPROVED TRAINING PIPELINE v2")
    print("=" * 70)
    print(f"  Environments: {', '.join(envs_to_train)}")
    print(f"  Device: {device}")
    if CUDA_AVAILABLE:
        print(f"  GPU: {GPU_NAME} ({GPU_MEM:.1f} GB)")
    print(f"  Mode: {'Quick' if quick_mode else 'Full'}")
    print()
    print("  KEY IMPROVEMENTS:")
    print("    • 4x more exploration (ent_coef: 0.005 → 0.02)")
    print("    • 3x faster learning rate (1e-4 → 3e-4)")
    print("    • Larger batches for stability")
    print("=" * 70 + "\n")

    results = {
        "start_time": datetime.now().isoformat(),
        "device": device,
        "quick_mode": quick_mode,
        "hyperparameters": IMPROVED_TRAINING_CONFIG,
        "environments": {},
    }

    overall_start = time.time()

    for i, env_key in enumerate(envs_to_train, 1):
        print(f"\n[{i}/{len(envs_to_train)}] Training {env_key}...")

        config = ENVIRONMENT_CONFIGS[env_key]
        checkpoint_dir = f"{BASE_CHECKPOINT_DIR}/{env_key}_{config['instruction'].replace(' ', '_')}"

        model_path, training_info = train_environment(
            env_key=env_key,
            config=config,
            checkpoint_dir=checkpoint_dir,
            device=device,
            quick_mode=quick_mode,
        )

        results["environments"][env_key] = training_info

    results["total_time_seconds"] = time.time() - overall_start
    results["total_time_human"] = format_time(results["total_time_seconds"])
    results["end_time"] = datetime.now().isoformat()

    # Save results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    results_file = f"{RESULTS_DIR}/training_results_v2_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 70)
    print("🎉 ALL TRAINING COMPLETE!")
    print("=" * 70)
    print(f"  Total time: {results['total_time_human']}")
    print(f"  Results: {results_file}")
    print(f"  Models: {BASE_CHECKPOINT_DIR}/")
    print("=" * 70)

    return results


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_environment(
        env_key: str,
        model_path: str,
        instruction: str,
        n_episodes: int = 10,
        device: str = "cuda",
        record_video: bool = True,
) -> Dict:
    """Evaluate a trained model."""

    env_name = ENVIRONMENT_CONFIGS[env_key]["env_name"]

    print(f"\n📊 Evaluating {env_name}...")
    print(f"   Model: {model_path}")

    if not os.path.exists(model_path):
        print(f"   ❌ Model not found!")
        return {"error": "Model not found"}

    agent = EnhancedMotionLanguageAgent(
        env_name=env_name,
        device=device,
        motion_checkpoint=MOTION_CHECKPOINT,
        use_stability_focus=True,
    )

    video_path = f"{VIDEOS_DIR}/{env_key}" if record_video else None
    if video_path:
        os.makedirs(video_path, exist_ok=True)

    try:
        results = agent.evaluate(
            model_path=model_path,
            instruction=instruction,
            n_episodes=n_episodes,
            record_video=record_video,
            video_path=video_path,
        )

        results["env_key"] = env_key
        results["env_name"] = env_name
        results["instruction"] = instruction

        print(f"   ✅ Mean Reward: {results['mean_reward']:.1f} ± {results['std_reward']:.1f}")
        print(f"      Mean Length: {results['mean_length']:.0f}")
        print(f"      Similarity:  {results['mean_similarity']:.3f}")

        return results

    except Exception as e:
        print(f"   ❌ Evaluation failed: {e}")
        return {"error": str(e)}


def evaluate_all_environments(
        envs_to_eval: List[str],
        n_episodes: int = 10,
        device: str = "cuda",
        record_video: bool = True,
) -> Dict:
    """Evaluate all trained models."""

    print("\n" + "=" * 70)
    print("📊 EVALUATION")
    print("=" * 70)

    results = {
        "timestamp": datetime.now().isoformat(),
        "n_episodes": n_episodes,
        "environments": {},
    }

    for env_key in envs_to_eval:
        config = ENVIRONMENT_CONFIGS[env_key]

        # Try v2 checkpoint first, then v1
        checkpoint_dir_v2 = f"{BASE_CHECKPOINT_DIR}/{env_key}_{config['instruction'].replace(' ', '_')}"
        checkpoint_dir_v1 = f"./checkpoints/{env_key}_{config['instruction'].replace(' ', '_')}"

        model_path = None
        for checkpoint_dir in [checkpoint_dir_v2, checkpoint_dir_v1]:
            for name in ["final_model.zip", f"final_model_{config['instruction'].replace(' ', '_')}.zip"]:
                path = f"{checkpoint_dir}/{name}"
                if os.path.exists(path):
                    model_path = path
                    break
            if model_path:
                break

        if not model_path:
            print(f"⚠ No model found for {env_key}")
            results["environments"][env_key] = {"error": "No model found"}
            continue

        eval_results = evaluate_environment(
            env_key=env_key,
            model_path=model_path,
            instruction=config["instruction"],
            n_episodes=n_episodes,
            device=device,
            record_video=record_video,
        )

        results["environments"][env_key] = eval_results

    # Save
    os.makedirs(RESULTS_DIR, exist_ok=True)
    results_file = f"{RESULTS_DIR}/evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n✅ Results saved: {results_file}")

    return results


# ============================================================================
# THESIS OUTPUTS
# ============================================================================

def generate_comparison_plots(eval_results: Dict) -> None:
    """Generate plots for thesis."""

    if not MATPLOTLIB_AVAILABLE:
        print("⚠ matplotlib not available")
        return

    os.makedirs(PLOTS_DIR, exist_ok=True)

    envs, rewards, lengths, similarities, stds = [], [], [], [], []

    for env_key, data in eval_results["environments"].items():
        if "error" not in data:
            envs.append(ENVIRONMENT_CONFIGS[env_key]["env_name"].replace("-v4", ""))
            rewards.append(data["mean_reward"])
            stds.append(data["std_reward"])
            lengths.append(data["mean_length"])
            similarities.append(data["mean_similarity"])

    if not envs:
        print("⚠ No valid results")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    colors = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c', '#f39c12']

    # Rewards
    ax = axes[0]
    bars = ax.bar(range(len(envs)), rewards, yerr=stds, color=colors[:len(envs)],
                  alpha=0.8, capsize=5, edgecolor='black')
    ax.set_xticks(range(len(envs)))
    ax.set_xticklabels(envs, rotation=45, ha='right')
    ax.set_ylabel('Mean Episode Reward', fontweight='bold')
    ax.set_title('(a) Reward Comparison', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # Episode Lengths
    ax = axes[1]
    ax.bar(range(len(envs)), lengths, color=colors[:len(envs)], alpha=0.8, edgecolor='black')
    ax.set_xticks(range(len(envs)))
    ax.set_xticklabels(envs, rotation=45, ha='right')
    ax.set_ylabel('Mean Episode Length', fontweight='bold')
    ax.set_title('(b) Episode Duration', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # Similarity
    ax = axes[2]
    ax.bar(range(len(envs)), similarities, color=colors[:len(envs)], alpha=0.8, edgecolor='black')
    ax.set_xticks(range(len(envs)))
    ax.set_xticklabels(envs, rotation=45, ha='right')
    ax.set_ylabel('Motion-Language Similarity', fontweight='bold')
    ax.set_title('(c) MotionGPT Alignment', fontweight='bold')
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle('Motion-Language Control Results (Improved Hyperparameters)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    plt.savefig(f"{PLOTS_DIR}/results_comparison.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{PLOTS_DIR}/results_comparison.pdf", bbox_inches='tight')
    plt.close()

    print(f"✅ Saved plots to {PLOTS_DIR}/")


def generate_latex_table(eval_results: Dict) -> None:
    """Generate LaTeX table."""

    latex = r"""\begin{table}[h]
\centering
\caption{Performance across MuJoCo environments with motion-language control.}
\label{tab:results}
\begin{tabular}{lcccc}
\toprule
\textbf{Environment} & \textbf{Instruction} & \textbf{Reward} & \textbf{Length} & \textbf{Similarity} \\
\midrule
"""

    for env_key, data in eval_results["environments"].items():
        if "error" not in data:
            env_name = ENVIRONMENT_CONFIGS[env_key]["env_name"].replace("-v4", "")
            instruction = data.get("instruction", "")
            reward = f"{data['mean_reward']:.0f} $\\pm$ {data['std_reward']:.0f}"
            length = f"{data['mean_length']:.0f}"
            sim = f"{data['mean_similarity']:.3f}"
            latex += f"{env_name} & {instruction} & {reward} & {length} & {sim} \\\\\n"

    latex += r"""\bottomrule
\end{tabular}
\end{table}
"""

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(f"{RESULTS_DIR}/results_table.tex", 'w') as f:
        f.write(latex)

    print(f"✅ Saved LaTeX table")


# ============================================================================
# UTILITIES
# ============================================================================

def format_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds / 60:.1f}m"
    else:
        return f"{seconds / 3600:.1f}h"


def print_system_info():
    print("\n" + "=" * 70)
    print("💻 SYSTEM INFO")
    print("=" * 70)
    print(f"  Python:  {sys.version.split()[0]}")
    print(f"  PyTorch: {torch.__version__}")
    print(f"  CUDA:    {'Available' if CUDA_AVAILABLE else 'Not available'}")
    if CUDA_AVAILABLE:
        print(f"  GPU:     {GPU_NAME} ({GPU_MEM:.1f} GB)")
    print("=" * 70)


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train with improved hyperparameters")

    parser.add_argument('--env', nargs='+',
                        choices=list(ENVIRONMENT_CONFIGS.keys()),
                        default=list(ENVIRONMENT_CONFIGS.keys()))
    parser.add_argument('--quick', action='store_true', help='Quick test mode')
    parser.add_argument('--eval-only', action='store_true', help='Only evaluate')
    parser.add_argument('--skip-eval', action='store_true', help='Skip evaluation')
    parser.add_argument('--device', default='cuda' if CUDA_AVAILABLE else 'cpu')
    parser.add_argument('--n-eval-episodes', type=int, default=10)
    parser.add_argument('--no-video', action='store_true')

    args = parser.parse_args()

    print_system_info()

    training_results = None

    if not args.eval_only:
        training_results = train_all_environments(
            envs_to_train=args.env,
            device=args.device,
            quick_mode=args.quick,
        )

    if not args.skip_eval:
        eval_results = evaluate_all_environments(
            envs_to_eval=args.env,
            n_episodes=args.n_eval_episodes,
            device=args.device,
            record_video=not args.no_video,
        )

        generate_comparison_plots(eval_results)
        generate_latex_table(eval_results)

        print("\n" + "=" * 70)
        print("🎓 THESIS OUTPUTS READY")
        print("=" * 70)
        print(f"  📊 Plots:  {PLOTS_DIR}/")
        print(f"  🎥 Videos: {VIDEOS_DIR}/")
        print(f"  📄 LaTeX:  {RESULTS_DIR}/results_table.tex")
        print("=" * 70)


if __name__ == "__main__":
    main()