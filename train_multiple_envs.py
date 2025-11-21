"""
Multi-Environment Training Script
Train motion-language control on Hopper, Ant, Walker2d, HalfCheetah
"""

import sys
from pathlib import Path

src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from agents.hierarchical_agent import EnhancedMotionLanguageAgent


def train_hopper():
    """Train Hopper to hop forward"""
    print("\n" + "="*60)
    print("TRAINING: Hopper-v4")
    print("="*60)
    
    agent = EnhancedMotionLanguageAgent(
        env_name="Hopper-v4",
        use_stability_focus=True
    )
    
    agent.train_on_instruction_stable(
        instruction="hop forward",
        total_timesteps=100000,
        n_envs=4,
        save_path="./checkpoints/hopper_forward",
        language_reward_weight=0.5
    )
    
    print("✓ Hopper training complete")
    return "./checkpoints/hopper_forward/final_model_hop_forward.zip"


def train_ant():
    """Train Ant to walk forward"""
    print("\n" + "="*60)
    print("TRAINING: Ant-v4")
    print("="*60)
    
    agent = EnhancedMotionLanguageAgent(
        env_name="Ant-v4",
        use_stability_focus=True
    )
    
    agent.train_on_instruction_stable(
        instruction="walk forward",
        total_timesteps=100000,
        n_envs=4,
        save_path="./checkpoints/ant_forward",
        language_reward_weight=0.5
    )
    
    print("✓ Ant training complete")
    return "./checkpoints/ant_forward/final_model_walk_forward.zip"


def train_walker2d():
    """Train Walker2d to walk forward"""
    print("\n" + "="*60)
    print("TRAINING: Walker2d-v4")
    print("="*60)
    
    agent = EnhancedMotionLanguageAgent(
        env_name="Walker2d-v4",
        use_stability_focus=True
    )
    
    agent.train_on_instruction_stable(
        instruction="walk forward",
        total_timesteps=100000,
        n_envs=4,
        save_path="./checkpoints/walker2d_forward",
        language_reward_weight=0.5
    )
    
    print("✓ Walker2d training complete")
    return "./checkpoints/walker2d_forward/final_model_walk_forward.zip"


def train_halfcheetah():
    """Train HalfCheetah to run forward"""
    print("\n" + "="*60)
    print("TRAINING: HalfCheetah-v4")
    print("="*60)
    
    agent = EnhancedMotionLanguageAgent(
        env_name="HalfCheetah-v4",
        use_stability_focus=True
    )
    
    agent.train_on_instruction_stable(
        instruction="run forward",
        total_timesteps=100000,
        n_envs=4,
        save_path="./checkpoints/halfcheetah_forward",
        language_reward_weight=0.5
    )
    
    print("✓ HalfCheetah training complete")
    return "./checkpoints/halfcheetah_forward/final_model_run_forward.zip"


def train_all_environments():
    """Train all environments sequentially"""
    
    print("="*60)
    print("MULTI-ENVIRONMENT TRAINING")
    print("Training on: Hopper, Ant, Walker2d, HalfCheetah")
    print("Estimated time: 4-6 hours total")
    print("="*60)
    
    models = {}
    
    # Train each environment
    try:
        models['hopper'] = train_hopper()
    except Exception as e:
        print(f"⚠ Hopper training failed: {e}")
    
    try:
        models['ant'] = train_ant()
    except Exception as e:
        print(f"⚠ Ant training failed: {e}")
    
    try:
        models['walker2d'] = train_walker2d()
    except Exception as e:
        print(f"⚠ Walker2d training failed: {e}")
    
    try:
        models['halfcheetah'] = train_halfcheetah()
    except Exception as e:
        print(f"⚠ HalfCheetah training failed: {e}")
    
    # Summary
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print("Trained models:")
    for env, path in models.items():
        print(f"  {env}: {path}")
    
    print("\nNext step: Run evaluation script")
    print("  python evaluate_all_environments.py")
    
    return models


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, choices=['hopper', 'ant', 'walker2d', 'halfcheetah', 'all'],
                       default='all', help='Environment to train')
    args = parser.parse_args()
    
    if args.env == 'all':
        train_all_environments()
    elif args.env == 'hopper':
        train_hopper()
    elif args.env == 'ant':
        train_ant()
    elif args.env == 'walker2d':
        train_walker2d()
    elif args.env == 'halfcheetah':
        train_halfcheetah()
