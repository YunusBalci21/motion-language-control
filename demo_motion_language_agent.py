#!/usr/bin/env python3
"""
Interactive Demo Script for Enhanced Motion-Language Control
Demonstrates the agent following natural language instructions in real-time
"""

import sys
import argparse
import time
from pathlib import Path
import numpy as np
import gymnasium as gym
import torch
from stable_baselines3 import PPO

# Add src to path
project_root = Path(__file__).parent
sys.path.append(str(project_root / "src"))

from agents.hierarchical_agent import EnhancedMotionLanguageAgent, DirectMotionLanguageWrapper
from models.motion_tokenizer import EnhancedMotionTokenizer


class InteractiveDemo:
    """Interactive demonstration of motion-language control"""

    def __init__(self,
                 model_path: str,
                 env_name: str = "Humanoid-v4",
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):

        self.model_path = model_path
        self.env_name = env_name
        self.device = device

        print(f"Initializing Interactive Demo")
        print(f"Model: {model_path}")
        print(f"Environment: {env_name}")
        print(f"Device: {device}")

        # Initialize components
        self.motion_tokenizer = EnhancedMotionTokenizer(device=device)
        self.agent = None
        self.env = None

        # Load trained model
        self._load_model()

        # Demo statistics
        self.demo_stats = {
            'instructions_tested': [],
            'similarities': [],
            'rewards': [],
            'computation_times': []
        }

    def _load_model(self):
        """Load the trained PPO model"""
        try:
            # Create dummy environment for model loading
            dummy_env = gym.make(self.env_name)
            dummy_env = DirectMotionLanguageWrapper(
                dummy_env,
                self.motion_tokenizer,
                instruction="walk forward"
            )

            # Load model
            self.agent = PPO.load(self.model_path, env=dummy_env)
            print(f"Successfully loaded model from {self.model_path}")

            dummy_env.close()

        except Exception as e:
            print(f"Failed to load model: {e}")
            sys.exit(1)

    def create_demo_environment(self, instruction: str):
        """Create environment for demonstration"""
        env = gym.make(self.env_name, render_mode="human")
        env = DirectMotionLanguageWrapper(
            env,
            self.motion_tokenizer,
            instruction=instruction,
            language_reward_weight=0.7
        )
        return env

    def run_single_instruction_demo(self,
                                    instruction: str,
                                    max_steps: int = 500,
                                    verbose: bool = True):
        """Run demo for a single instruction"""

        if verbose:
            print(f"\n{'=' * 60}")
            print(f"DEMONSTRATING: '{instruction}'")
            print(f"{'=' * 60}")

        # Create environment
        env = self.create_demo_environment(instruction)

        # Reset environment
        obs, info = env.reset()

        # Demo statistics
        total_reward = 0
        language_reward = 0
        similarities = []
        step_times = []

        for step in range(max_steps):
            step_start = time.time()

            # Predict action
            action, _ = self.agent.predict(obs, deterministic=True)

            # Execute action
            obs, reward, terminated, truncated, info = env.step(action)

            step_time = time.time() - step_start
            step_times.append(step_time)

            # Track statistics
            total_reward += reward
            language_reward += info.get('language_reward', 0)
            similarities.append(info.get('motion_language_similarity', 0))

            # Render environment
            env.render()

            # Print step info
            if verbose and step % 50 == 0:
                print(f"Step {step}: Reward={reward:.2f}, "
                      f"Similarity={info.get('motion_language_similarity', 0):.3f}, "
                      f"Time={step_time * 1000:.1f}ms")

            # Check termination
            if terminated or truncated:
                if verbose:
                    print(f"Episode ended at step {step}")
                break

            # Small delay for visualization
            time.sleep(0.01)

        env.close()

        # Compute statistics
        demo_result = {
            'instruction': instruction,
            'steps': step + 1,
            'total_reward': total_reward,
            'language_reward': language_reward,
            'mean_similarity': np.mean(similarities) if similarities else 0.0,
            'final_similarity': similarities[-1] if similarities else 0.0,
            'mean_step_time': np.mean(step_times) * 1000,  # Convert to ms
            'success': np.mean(similarities[-50:]) > 0.6 if len(similarities) >= 50 else False
        }

        if verbose:
            print(f"\nDemo Results for '{instruction}':")
            print(f"  Steps: {demo_result['steps']}")
            print(f"  Total Reward: {demo_result['total_reward']:.2f}")
            print(f"  Language Reward: {demo_result['language_reward']:.2f}")
            print(f"  Mean Similarity: {demo_result['mean_similarity']:.3f}")
            print(f"  Final Similarity: {demo_result['final_similarity']:.3f}")
            print(f"  Mean Step Time: {demo_result['mean_step_time']:.1f}ms")
            print(f"  Success: {demo_result['success']}")

        # Update demo statistics
        self.demo_stats['instructions_tested'].append(instruction)
        self.demo_stats['similarities'].append(demo_result['mean_similarity'])
        self.demo_stats['rewards'].append(demo_result['total_reward'])
        self.demo_stats['computation_times'].append(demo_result['mean_step_time'])

        return demo_result

    def run_instruction_sequence(self, instructions: list, steps_per_instruction: int = 300):
        """Run sequence of instructions in same environment"""

        print(f"\n{'=' * 80}")
        print(f"INSTRUCTION SEQUENCE DEMO")
        print(f"Instructions: {instructions}")
        print(f"Steps per instruction: {steps_per_instruction}")
        print(f"{'=' * 80}")

        # Create environment with first instruction
        env = self.create_demo_environment(instructions[0])
        obs, info = env.reset()

        sequence_results = []

        for i, instruction in enumerate(instructions):
            print(f"\n[{i + 1}/{len(instructions)}] Switching to: '{instruction}'")

            # Change instruction
            env.set_instruction(instruction)

            # Run for specified steps
            instruction_stats = {
                'instruction': instruction,
                'rewards': [],
                'similarities': []
            }

            for step in range(steps_per_instruction):
                action, _ = self.agent.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)

                instruction_stats['rewards'].append(reward)
                instruction_stats['similarities'].append(info.get('motion_language_similarity', 0))

                env.render()

                if step % 50 == 0:
                    print(f"  Step {step}: Similarity={info.get('motion_language_similarity', 0):.3f}")

                if terminated or truncated:
                    obs, info = env.reset()

                time.sleep(0.01)

            # Compute instruction results
            result = {
                'instruction': instruction,
                'mean_reward': np.mean(instruction_stats['rewards']),
                'mean_similarity': np.mean(instruction_stats['similarities']),
                'final_similarity': np.mean(instruction_stats['similarities'][-50:])
            }

            sequence_results.append(result)

            print(f"  Results: Similarity={result['mean_similarity']:.3f}, "
                  f"Reward={result['mean_reward']:.2f}")

        env.close()

        print(f"\nSequence Demo Completed!")
        print(f"Summary:")
        for result in sequence_results:
            print(f"  '{result['instruction']}': {result['mean_similarity']:.3f} similarity")

        return sequence_results

    def run_interactive_demo(self):
        """Run interactive demo where user inputs instructions"""

        print(f"\n{'=' * 80}")
        print(f"INTERACTIVE MOTION-LANGUAGE DEMO")
        print(f"{'=' * 80}")
        print(f"Enter natural language instructions for the agent to follow.")
        print(f"Examples: 'walk forward', 'turn left', 'jump up and down'")
        print(f"Type 'quit' to exit, 'help' for more examples.")
        print(f"{'=' * 80}")

        # Create environment
        env = self.create_demo_environment("walk forward")
        obs, info = env.reset()

        instruction_count = 0

        while True:
            # Get instruction from user
            try:
                instruction = input(f"\n[{instruction_count + 1}] Enter instruction: ").strip()
            except KeyboardInterrupt:
                print("\nExiting interactive demo...")
                break

            if instruction.lower() in ['quit', 'exit', 'q']:
                print("Exiting interactive demo...")
                break

            elif instruction.lower() == 'help':
                print(f"\nExample instructions:")
                print(f"  Basic: walk forward, walk backward, turn left, turn right")
                print(f"  Advanced: run forward quickly, walk in a circle, jump up")
                print(f"  Complex: wave your hand while walking, crouch down low")
                continue

            elif instruction == '':
                print("Please enter an instruction.")
                continue

            # Run demo for this instruction
            print(f"\nExecuting: '{instruction}'")
            env.set_instruction(instruction)

            # Run for 200 steps
            similarities = []
            for step in range(200):
                action, _ = self.agent.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)

                similarities.append(info.get('motion_language_similarity', 0))
                env.render()

                if terminated or truncated:
                    obs, info = env.reset()

                time.sleep(0.02)  # Slightly slower for interaction

            # Show results
            mean_similarity = np.mean(similarities)
            final_similarity = np.mean(similarities[-50:])

            print(f"Results: Mean similarity = {mean_similarity:.3f}, "
                  f"Final similarity = {final_similarity:.3f}")

            if final_similarity > 0.7:
                print("Excellent performance!")
            elif final_similarity > 0.5:
                print("Good performance!")
            else:
                print("Needs improvement. Try a different instruction.")

            instruction_count += 1

        env.close()

    def run_benchmark_demo(self, instructions: list = None):
        """Run benchmark demonstration comparing multiple instructions"""

        if instructions is None:
            instructions = [
                "walk forward",
                "walk backward",
                "turn left",
                "turn right",
                "run forward",
                "walk in a circle",
                "jump up and down"
            ]

        print(f"\n{'=' * 80}")
        print(f"BENCHMARK DEMO")
        print(f"Testing {len(instructions)} instructions")
        print(f"{'=' * 80}")

        results = []

        for i, instruction in enumerate(instructions):
            print(f"\n[{i + 1}/{len(instructions)}] Testing '{instruction}'...")

            result = self.run_single_instruction_demo(
                instruction,
                max_steps=300,
                verbose=False
            )

            results.append(result)

            print(f"  Similarity: {result['mean_similarity']:.3f}")
            print(f"  Success: {'YES' if result['success'] else 'NO'}")

        # Print summary
        print(f"\n{'=' * 80}")
        print(f"BENCHMARK SUMMARY")
        print(f"{'=' * 80}")

        successful_instructions = [r for r in results if r['success']]
        mean_similarity = np.mean([r['mean_similarity'] for r in results])

        print(f"Instructions tested: {len(instructions)}")
        print(
            f"Successful: {len(successful_instructions)} ({100 * len(successful_instructions) / len(instructions):.1f}%)")
        print(f"Mean similarity: {mean_similarity:.3f}")
        print(f"Mean computation time: {np.mean(self.demo_stats['computation_times']):.1f}ms")

        print(f"\nDetailed Results:")
        for result in sorted(results, key=lambda x: x['mean_similarity'], reverse=True):
            status = "✓" if result['success'] else "✗"
            print(f"  {status} '{result['instruction']}': {result['mean_similarity']:.3f}")

        return results


def main():
    parser = argparse.ArgumentParser(
        description="Interactive Demo for Enhanced Motion-Language Control",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to trained model (.zip file)')
    parser.add_argument('--env', type=str, default='Humanoid-v4',
                        help='MuJoCo environment name')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device (cuda/cpu/auto)')

    # Demo modes
    parser.add_argument('--interactive', action='store_true',
                        help='Run interactive demo with user input')
    parser.add_argument('--benchmark', action='store_true',
                        help='Run benchmark on multiple instructions')
    parser.add_argument('--sequence', action='store_true',
                        help='Run instruction sequence demo')

    # Specific instructions
    parser.add_argument('--instructions', nargs='+',
                        default=['walk forward', 'turn left', 'jump up'],
                        help='Specific instructions to demo')
    parser.add_argument('--steps', type=int, default=500,
                        help='Steps per instruction')

    args = parser.parse_args()

    # Check if model exists
    if not Path(args.model_path).exists():
        print(f"Error: Model file not found: {args.model_path}")
        sys.exit(1)

    # Set device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    print(f"Enhanced Motion-Language Control Demo")
    print(f"Device: {device}")

    # Initialize demo
    demo = InteractiveDemo(args.model_path, args.env, device)

    # Run appropriate demo mode
    if args.interactive:
        demo.run_interactive_demo()

    elif args.benchmark:
        demo.run_benchmark_demo()

    elif args.sequence:
        demo.run_instruction_sequence(args.instructions, args.steps)

    else:
        # Default: run specific instructions
        print(f"Running demo for instructions: {args.instructions}")

        for instruction in args.instructions:
            demo.run_single_instruction_demo(instruction, args.steps)

            # Small delay between instructions
            time.sleep(2)

    print(f"\nDemo completed!")


if __name__ == "__main__":
    main()