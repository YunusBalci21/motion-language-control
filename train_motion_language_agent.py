#!/usr/bin/env python3
"""
Training Script for Motion-Language Control Agent
"""

import sys
import argparse
from pathlib import Path
import yaml
import json
from datetime import datetime

# Add src to path
project_root = Path(__file__).parent
sys.path.append(str(project_root / "src"))

from agents.hierarchical_agent import MotionLanguageAgent


def load_config(config_path: str) -> dict:
    """Load training configuration"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_experiment_dir(base_dir: str, experiment_name: str) -> Path:
    """Create directory for experiment outputs"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = Path(base_dir) / f"{experiment_name}_{timestamp}"
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Create subdirectories
    (exp_dir / "checkpoints").mkdir(exist_ok=True)
    (exp_dir / "logs").mkdir(exist_ok=True)
    (exp_dir / "results").mkdir(exist_ok=True)

    return exp_dir


def train_single_instruction(agent: MotionLanguageAgent,
                             instruction: str,
                             config: dict,
                             exp_dir: Path) -> str:
    """Train agent on a single instruction"""
    print(f"\n Training on: '{instruction}'")

    checkpoint_dir = exp_dir / "checkpoints" / instruction.replace(" ", "_")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    model_path = agent.train(
        instruction=instruction,
        total_timesteps=config['training']['total_timesteps'],
        save_path=str(checkpoint_dir),
        eval_freq=config['training'].get('eval_freq', 10000)
    )

    print(f" Training completed for: '{instruction}'")
    return model_path


def evaluate_instruction(agent: MotionLanguageAgent,
                         instruction: str,
                         model_path: str,
                         config: dict,
                         exp_dir: Path) -> dict:
    """Evaluate agent on instruction"""
    print(f"\n Evaluating: '{instruction}'")

    results = agent.evaluate(
        instruction=instruction,
        num_episodes=config['evaluation']['n_eval_episodes'],
        model_path=model_path,
        render=False
    )

    # Save results
    results_file = exp_dir / "results" / f"{instruction.replace(' ', '_')}_results.json"
    results_file.parent.mkdir(exist_ok=True)

    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    return results


def train_curriculum(agent: MotionLanguageAgent,
                     instructions: list,
                     config: dict,
                     exp_dir: Path) -> dict:
    """Train agent on curriculum of instructions"""
    print(f"\n Training curriculum with {len(instructions)} instructions")

    all_results = {}
    model_paths = {}

    for i, instruction in enumerate(instructions):
        print(f"\n{'=' * 50}")
        print(f"Curriculum Step {i + 1}/{len(instructions)}: '{instruction}'")
        print(f"{'=' * 50}")

        # Train on this instruction
        model_path = train_single_instruction(agent, instruction, config, exp_dir)
        model_paths[instruction] = model_path

        # Evaluate on this instruction
        results = evaluate_instruction(agent, instruction, model_path, config, exp_dir)
        all_results[instruction] = results

        # Cross-evaluation: test on other instructions
        if config.get('cross_evaluation', False) and i > 0:
            print(f"\n Cross-evaluating '{instruction}' model on previous instructions...")
            for prev_instruction in instructions[:i]:
                cross_results = agent.evaluate(
                    instruction=prev_instruction,
                    num_episodes=3,  # Fewer episodes for cross-eval
                    model_path=model_path
                )

                cross_key = f"{instruction}_model_on_{prev_instruction}"
                all_results[cross_key] = cross_results

    return all_results, model_paths


def run_final_demo(agent: MotionLanguageAgent,
                   instructions: list,
                   model_paths: dict,
                   config: dict):
    """Run final demonstration with all trained models"""
    print(f"\n Final Demonstration")
    print("=" * 30)

    for instruction in instructions:
        if instruction in model_paths:
            print(f"\nDemonstrating: '{instruction}'")
            agent.demo_instruction_following(
                instructions=[instruction],
                model_path=model_paths[instruction],
                max_steps=config.get('demo_steps', 200)
            )


def main():
    parser = argparse.ArgumentParser(
        description="Train motion-language control agents",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                        help='Path to configuration file')
    parser.add_argument('--env', type=str, default='Humanoid-v4',
                        help='MuJoCo environment name')
    parser.add_argument('--instruction', type=str, default=None,
                        help='Single instruction to train on (overrides config)')
    parser.add_argument('--output-dir', type=str, default='./experiments',
                        help='Output directory for experiments')
    parser.add_argument('--experiment-name', type=str, default='motion_lang_training',
                        help='Name for this experiment')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (cuda/cpu/auto)')
    parser.add_argument('--quick-test', action='store_true',
                        help='Run quick test with reduced timesteps')
    parser.add_argument('--eval-only', type=str, default=None,
                        help='Path to trained model for evaluation only')
    parser.add_argument('--demo-only', type=str, default=None,
                        help='Path to trained model for demo only')

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    print(f" Loaded config: {args.config}")

    # Override config with command line args
    if args.quick_test:
        config['training']['total_timesteps'] = 5000
        config['evaluation']['n_eval_episodes'] = 3
        print(" Quick test mode: reduced timesteps")

    # Set device
    if args.device == 'auto':
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    print(f" Using device: {device}")

    # Create experiment directory
    exp_dir = create_experiment_dir(args.output_dir, args.experiment_name)
    print(f" Experiment directory: {exp_dir}")

    # Save config to experiment directory
    with open(exp_dir / "config.yaml", 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    # Create agent
    print(f" Creating agent for environment: {args.env}")
    agent = MotionLanguageAgent(env_name=args.env, device=device)

    # Determine instructions to work with
    if args.instruction:
        instructions = [args.instruction]
        print(f" Single instruction mode: '{args.instruction}'")
    else:
        instructions = config.get('instructions', ['walk forward', 'turn left', 'turn right'])
        print(f" Multi-instruction mode: {instructions}")

    # Evaluation-only mode
    if args.eval_only:
        print(f" Evaluation-only mode with model: {args.eval_only}")
        all_results = {}
        for instruction in instructions:
            results = evaluate_instruction(agent, instruction, args.eval_only, config, exp_dir)
            all_results[instruction] = results

        # Print summary
        print(f"\n Evaluation Summary:")
        for instruction, results in all_results.items():
            print(f"  '{instruction}': {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")

        return

    # Demo-only mode
    if args.demo_only:
        print(f" Demo-only mode with model: {args.demo_only}")
        agent.demo_instruction_following(
            instructions=instructions,
            model_path=args.demo_only,
            max_steps=config.get('demo_steps', 200)
        )
        return

    # Full training mode
    print(f"\n Starting training...")
    print(f"Environment: {args.env}")
    print(f"Instructions: {instructions}")
    print(f"Total timesteps per instruction: {config['training']['total_timesteps']}")

    # Train curriculum
    all_results, model_paths = train_curriculum(agent, instructions, config, exp_dir)

    # Save all results
    final_results_file = exp_dir / "results" / "final_results.json"
    with open(final_results_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    # Print summary
    print(f"\n Training Summary:")
    print("=" * 50)
    for instruction in instructions:
        if instruction in all_results:
            results = all_results[instruction]
            print(f"'{instruction}': {results['mean_reward']:.2f} ± {results['std_reward']:.2f} "
                  f"(Language: {results['mean_language_reward']:.2f})")

    # Run final demo
    if config.get('run_final_demo', True):
        run_final_demo(agent, instructions, model_paths, config)

    print(f"\n Training completed!")
    print(f" Results saved to: {exp_dir}")
    print(f" Model checkpoints: {exp_dir / 'checkpoints'}")

    # Print usage instructions
    print(f"\n To evaluate your trained models:")
    for instruction in instructions:
        if instruction in model_paths:
            print(
                f"  python train_motion_language_agent.py --eval-only {model_paths[instruction]} --instruction '{instruction}'")

    print(f"\n To run demos:")
    for instruction in instructions:
        if instruction in model_paths:
            print(
                f"  python train_motion_language_agent.py --demo-only {model_paths[instruction]} --instruction '{instruction}'")


if __name__ == "__main__":
    main()