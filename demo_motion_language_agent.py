#!/usr/bin/env python3
"""
Enhanced Interactive Demo Script for Motion-Language Control
With video recording, live viewing, and comprehensive evaluation metrics
"""

import sys
import argparse
import time
import json
from pathlib import Path
import numpy as np
import gymnasium as gym
import torch
from stable_baselines3 import PPO
import imageio
import cv2

# Add src to path
project_root = Path(__file__).parent
sys.path.append(str(project_root / "src"))

from agents.hierarchical_agent import EnhancedMotionLanguageAgent, DirectMotionLanguageWrapper
from models.motion_tokenizer import MotionTokenizer


class EnhancedInteractiveDemo:
    """Enhanced demonstration with video recording and comprehensive metrics"""

    def __init__(self,
                 model_path: str,
                 env_name: str = "Humanoid-v4",
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 output_dir: str = "./demo_outputs"):

        self.model_path = model_path
        self.env_name = env_name
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        print(f"Initializing Enhanced Interactive Demo")
        print(f"Model: {model_path}")
        print(f"Environment: {env_name}")
        print(f"Device: {device}")
        print(f"Output directory: {output_dir}")

        # Initialize components
        self.motion_tokenizer = MotionTokenizer(device=device)
        self.agent = None
        self.env = None

        # Load trained model
        self._load_model()

        # Demo statistics
        self.demo_stats = {
            'instructions_tested': [],
            'similarities': [],
            'success_rates': [],
            'rewards': [],
            'computation_times': [],
            'motion_quality_scores': [],
            'videos_recorded': []
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

    def create_demo_environment(self, instruction: str, record_video: bool = False):
        """Create environment for demonstration - Fixed for original wrapper"""
        render_mode = "rgb_array" if record_video else "human"
        env = gym.make(self.env_name, render_mode=render_mode)

        # Use the original DirectMotionLanguageWrapper interface
        env = DirectMotionLanguageWrapper(
            env,
            self.motion_tokenizer,
            instruction=instruction,
            reward_scale=1.0,
            motion_history_length=20,
            reward_aggregation="weighted_recent"
        )

        # Set language reward weight after creation
        env.language_reward_weight = 0.7

        return env

    def run_single_instruction_demo(self,
                                    instruction: str,
                                    max_steps: int = 500,
                                    record_video: bool = True,
                                    show_live: bool = True,
                                    verbose: bool = True):
        """Run enhanced demo for a single instruction with video recording"""

        if verbose:
            print(f"\n{'=' * 60}")
            print(f"DEMONSTRATING: '{instruction}'")
            print(f"{'=' * 60}")

        # Create environment
        env = self.create_demo_environment(instruction, record_video=record_video)

        # Setup video recording
        video_frames = []
        video_filename = None
        if record_video:
            safe_instruction = instruction.replace(' ', '_').replace('/', '_')
            video_filename = self.output_dir / "videos" / f"demo_{safe_instruction}_{int(time.time())}.mp4"
            video_filename.parent.mkdir(parents=True, exist_ok=True)

        # Setup live display
        if show_live and not record_video:
            cv2.namedWindow('Motion-Language Demo', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Motion-Language Demo', 800, 600)

        # Reset environment
        obs, info = env.reset()

        # Demo statistics
        total_reward = 0
        language_reward = 0
        similarities = []
        success_rates = []
        step_times = []
        motion_quality_scores = []

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
            success_rates.append(info.get('success_rate', 0))
            motion_quality_scores.append(info.get('motion_overall_quality', 0))

            # Render and record
            frame = env.render()
            if frame is not None:
                if record_video:
                    video_frames.append(frame)

                if show_live and not record_video:
                    # Convert RGB to BGR for OpenCV
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                    # Add text overlay with metrics
                    overlay_text = [
                        f"Instruction: {instruction}",
                        f"Step: {step}/{max_steps}",
                        f"Similarity: {similarities[-1]:.3f}",
                        f"Success Rate: {success_rates[-1]:.3f}",
                        f"Quality: {motion_quality_scores[-1]:.3f}",
                        f"Reward: {reward:.2f}"
                    ]

                    for i, text in enumerate(overlay_text):
                        cv2.putText(frame_bgr, text, (10, 30 + i * 25),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                    cv2.imshow('Motion-Language Demo', frame_bgr)

                    # Break on 'q' key press
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("Demo interrupted by user")
                        break

            # Print step info
            if verbose and step % 50 == 0:
                print(f"Step {step}: Reward={reward:.2f}, "
                      f"Similarity={similarities[-1]:.3f}, "
                      f"Success={success_rates[-1]:.3f}, "
                      f"Quality={motion_quality_scores[-1]:.3f}, "
                      f"Time={step_time * 1000:.1f}ms")

            # Check termination
            if terminated or truncated:
                if verbose:
                    print(f"Episode ended at step {step}")
                break

            # Small delay for visualization
            time.sleep(0.01)

        # Cleanup display
        if show_live and not record_video:
            cv2.destroyAllWindows()

        env.close()

        # Save video
        if record_video and video_frames and len(video_frames) > 10:
            try:
                print(f"Saving video with {len(video_frames)} frames...")
                imageio.mimsave(str(video_filename), video_frames, fps=30)
                print(f"Video saved: {video_filename}")
                self.demo_stats['videos_recorded'].append(str(video_filename))
            except Exception as e:
                print(f"Video saving failed: {e}")

        # Compute statistics
        demo_result = {
            'instruction': instruction,
            'steps': step + 1,
            'total_reward': total_reward,
            'language_reward': language_reward,
            'mean_similarity': np.mean(similarities) if similarities else 0.0,
            'final_similarity': similarities[-1] if similarities else 0.0,
            'mean_success_rate': np.mean(success_rates) if success_rates else 0.0,
            'final_success_rate': success_rates[-1] if success_rates else 0.0,
            'mean_motion_quality': np.mean(motion_quality_scores) if motion_quality_scores else 0.0,
            'mean_step_time': np.mean(step_times) * 1000,  # Convert to ms
            'success': np.mean(similarities[-50:]) > 0.6 if len(similarities) >= 50 else False,
            'video_path': str(video_filename) if record_video and video_frames else None
        }

        if verbose:
            print(f"\nDemo Results for '{instruction}':")
            print(f"  Steps: {demo_result['steps']}")
            print(f"  Total Reward: {demo_result['total_reward']:.2f}")
            print(f"  Language Reward: {demo_result['language_reward']:.2f}")
            print(f"  Mean Similarity: {demo_result['mean_similarity']:.3f}")
            print(f"  Final Similarity: {demo_result['final_similarity']:.3f}")
            print(f"  Mean Success Rate: {demo_result['mean_success_rate']:.3f}")
            print(f"  Mean Motion Quality: {demo_result['mean_motion_quality']:.3f}")
            print(f"  Mean Step Time: {demo_result['mean_step_time']:.1f}ms")
            print(f"  Overall Success: {demo_result['success']}")
            if demo_result['video_path']:
                print(f"  Video: {demo_result['video_path']}")

        # Update demo statistics
        self.demo_stats['instructions_tested'].append(instruction)
        self.demo_stats['similarities'].append(demo_result['mean_similarity'])
        self.demo_stats['success_rates'].append(demo_result['mean_success_rate'])
        self.demo_stats['rewards'].append(demo_result['total_reward'])
        self.demo_stats['computation_times'].append(demo_result['mean_step_time'])
        self.demo_stats['motion_quality_scores'].append(demo_result['mean_motion_quality'])

        return demo_result

    def run_instruction_sequence(self,
                                 instructions: list,
                                 steps_per_instruction: int = 300,
                                 record_video: bool = True):
        """Run sequence of instructions with video recording"""

        print(f"\n{'=' * 80}")
        print(f"INSTRUCTION SEQUENCE DEMO")
        print(f"Instructions: {instructions}")
        print(f"Steps per instruction: {steps_per_instruction}")
        print(f"Record video: {record_video}")
        print(f"{'=' * 80}")

        # Create environment with first instruction
        env = self.create_demo_environment(instructions[0], record_video=record_video)
        obs, info = env.reset()

        # Setup video recording for sequence
        video_frames = []
        video_filename = None
        if record_video:
            video_filename = self.output_dir / "videos" / f"sequence_demo_{int(time.time())}.mp4"
            video_filename.parent.mkdir(parents=True, exist_ok=True)

        sequence_results = []

        for i, instruction in enumerate(instructions):
            print(f"\n[{i + 1}/{len(instructions)}] Switching to: '{instruction}'")

            # Change instruction
            env.set_instruction(instruction)

            # Run for specified steps
            instruction_stats = {
                'instruction': instruction,
                'rewards': [],
                'similarities': [],
                'success_rates': [],
                'motion_quality': []
            }

            for step in range(steps_per_instruction):
                action, _ = self.agent.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)

                instruction_stats['rewards'].append(reward)
                instruction_stats['similarities'].append(info.get('motion_language_similarity', 0))
                instruction_stats['success_rates'].append(info.get('success_rate', 0))
                instruction_stats['motion_quality'].append(info.get('motion_overall_quality', 0))

                # Record frame
                frame = env.render()
                if frame is not None and record_video:
                    # Add instruction overlay
                    frame_with_text = frame.copy()

                    # Convert to PIL Image for text overlay
                    try:
                        from PIL import Image, ImageDraw, ImageFont
                        pil_image = Image.fromarray(frame_with_text)
                        draw = ImageDraw.Draw(pil_image)

                        # Add text overlay
                        text = f"Instruction {i + 1}/{len(instructions)}: {instruction}"
                        try:
                            font = ImageFont.truetype("arial.ttf", 20)
                        except:
                            font = ImageFont.load_default()

                        draw.text((10, 10), text, fill=(255, 255, 255), font=font)
                        frame_with_text = np.array(pil_image)
                    except ImportError:
                        # Fallback if PIL not available
                        pass

                    video_frames.append(frame_with_text)

                if step % 50 == 0:
                    print(f"  Step {step}: Similarity={info.get('motion_language_similarity', 0):.3f}, "
                          f"Success={info.get('success_rate', 0):.3f}")

                if terminated or truncated:
                    obs, info = env.reset()

                time.sleep(0.01)

            # Compute instruction results
            result = {
                'instruction': instruction,
                'mean_reward': np.mean(instruction_stats['rewards']),
                'mean_similarity': np.mean(instruction_stats['similarities']),
                'mean_success_rate': np.mean(instruction_stats['success_rates']),
                'mean_motion_quality': np.mean(instruction_stats['motion_quality']),
                'final_similarity': np.mean(instruction_stats['similarities'][-50:])
            }

            sequence_results.append(result)

            print(f"  Results: Similarity={result['mean_similarity']:.3f}, "
                  f"Success={result['mean_success_rate']:.3f}, "
                  f"Quality={result['mean_motion_quality']:.3f}, "
                  f"Reward={result['mean_reward']:.2f}")

        env.close()

        # Save sequence video
        if record_video and video_frames and len(video_frames) > 10:
            try:
                print(f"Saving sequence video with {len(video_frames)} frames...")
                imageio.mimsave(str(video_filename), video_frames, fps=30)
                print(f"Sequence video saved: {video_filename}")
                self.demo_stats['videos_recorded'].append(str(video_filename))
            except Exception as e:
                print(f"Sequence video saving failed: {e}")

        print(f"\nSequence Demo Completed!")
        print(f"Summary:")
        for result in sequence_results:
            print(f"  '{result['instruction']}': {result['mean_similarity']:.3f} similarity, "
                  f"{result['mean_success_rate']:.3f} success")

        # Save sequence results
        sequence_data = {
            'instructions': instructions,
            'results': sequence_results,
            'video_path': str(video_filename) if record_video and video_frames else None,
            'timestamp': time.time()
        }

        results_file = self.output_dir / f"sequence_results_{int(time.time())}.json"
        with open(results_file, 'w') as f:
            json.dump(sequence_data, f, indent=2, default=str)

        return sequence_results

    def run_interactive_demo(self, record_videos: bool = True):
        """Run interactive demo with video recording"""

        print(f"\n{'=' * 80}")
        print(f"INTERACTIVE MOTION-LANGUAGE DEMO")
        print(f"{'=' * 80}")
        print(f"Enter natural language instructions for the agent to follow.")
        print(f"Examples: 'walk forward', 'turn left', 'jump up and down'")
        print(f"Type 'quit' to exit, 'help' for examples, 'stats' for statistics.")
        print(f"Videos will be recorded: {record_videos}")
        print(f"{'=' * 80}")

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

            elif instruction.lower() == 'stats':
                self._print_demo_statistics()
                continue

            elif instruction == '':
                print("Please enter an instruction.")
                continue

            # Run demo for this instruction
            print(f"\nExecuting: '{instruction}'")

            result = self.run_single_instruction_demo(
                instruction,
                max_steps=200,
                record_video=record_videos,
                show_live=not record_videos,  # Show live if not recording
                verbose=True
            )

            # Show results
            print(f"Results: Similarity = {result['mean_similarity']:.3f}, "
                  f"Success = {result['mean_success_rate']:.3f}, "
                  f"Quality = {result['mean_motion_quality']:.3f}")

            if result['mean_similarity'] > 0.7:
                print("Excellent performance!")
            elif result['mean_similarity'] > 0.5:
                print("Good performance!")
            else:
                print("Needs improvement. Try a different instruction.")

            instruction_count += 1

        # Save final statistics
        self._save_demo_statistics()

    def run_benchmark_demo(self, instructions: list = None, record_videos: bool = True):
        """Run comprehensive benchmark demonstration"""

        if instructions is None:
            instructions = [
                "walk forward",
                "walk backward",
                "turn left",
                "turn right",
                "stop moving",
                "run forward",
                "walk in a circle",
                "jump up and down"
            ]

        print(f"\n{'=' * 80}")
        print(f"COMPREHENSIVE BENCHMARK DEMO")
        print(f"Testing {len(instructions)} instructions")
        print(f"Recording videos: {record_videos}")
        print(f"{'=' * 80}")

        results = []

        for i, instruction in enumerate(instructions):
            print(f"\n[{i + 1}/{len(instructions)}] Testing '{instruction}'...")

            result = self.run_single_instruction_demo(
                instruction,
                max_steps=400,
                record_video=record_videos,
                show_live=False,
                verbose=False
            )

            results.append(result)

            print(f"  Similarity: {result['mean_similarity']:.3f}")
            print(f"  Success Rate: {result['mean_success_rate']:.3f}")
            print(f"  Motion Quality: {result['mean_motion_quality']:.3f}")
            print(f"  Overall Success: {'YES' if result['success'] else 'NO'}")

        # Print comprehensive summary
        print(f"\n{'=' * 80}")
        print(f"BENCHMARK SUMMARY")
        print(f"{'=' * 80}")

        successful_instructions = [r for r in results if r['success']]
        mean_similarity = np.mean([r['mean_similarity'] for r in results])
        mean_success_rate = np.mean([r['mean_success_rate'] for r in results])
        mean_motion_quality = np.mean([r['mean_motion_quality'] for r in results])

        print(f"Instructions tested: {len(instructions)}")
        print(
            f"Successful: {len(successful_instructions)} ({100 * len(successful_instructions) / len(instructions):.1f}%)")
        print(f"Mean similarity: {mean_similarity:.3f}")
        print(f"Mean success rate: {mean_success_rate:.3f}")
        print(f"Mean motion quality: {mean_motion_quality:.3f}")
        print(f"Mean computation time: {np.mean(self.demo_stats['computation_times']):.1f}ms")

        print(f"\nDetailed Results:")
        for result in sorted(results, key=lambda x: x['mean_similarity'], reverse=True):
            status = "✓" if result['success'] else "✗"
            print(f"  {status} '{result['instruction']}': "
                  f"Sim={result['mean_similarity']:.3f}, "
                  f"Succ={result['mean_success_rate']:.3f}, "
                  f"Qual={result['mean_motion_quality']:.3f}")

        # Save benchmark results
        benchmark_data = {
            'instructions': instructions,
            'results': results,
            'summary': {
                'total_instructions': len(instructions),
                'successful_instructions': len(successful_instructions),
                'success_percentage': len(successful_instructions) / len(instructions),
                'mean_similarity': mean_similarity,
                'mean_success_rate': mean_success_rate,
                'mean_motion_quality': mean_motion_quality,
                'mean_computation_time': np.mean(self.demo_stats['computation_times'])
            },
            'timestamp': time.time()
        }

        results_file = self.output_dir / f"benchmark_results_{int(time.time())}.json"
        with open(results_file, 'w') as f:
            json.dump(benchmark_data, f, indent=2, default=str)

        print(f"\nBenchmark results saved: {results_file}")

        if self.demo_stats['videos_recorded']:
            print(f"Videos recorded: {len(self.demo_stats['videos_recorded'])}")
            for video in self.demo_stats['videos_recorded'][-3:]:  # Show last 3
                print(f"  {video}")

        return results

    def _print_demo_statistics(self):
        """Print current demo statistics"""
        if not self.demo_stats['instructions_tested']:
            print("No demo statistics available yet.")
            return

        print(f"\n{'=' * 60}")
        print(f"DEMO STATISTICS")
        print(f"{'=' * 60}")
        print(f"Instructions tested: {len(self.demo_stats['instructions_tested'])}")
        print(f"Mean similarity: {np.mean(self.demo_stats['similarities']):.3f}")
        print(f"Mean success rate: {np.mean(self.demo_stats['success_rates']):.3f}")
        print(f"Mean motion quality: {np.mean(self.demo_stats['motion_quality_scores']):.3f}")
        print(f"Mean computation time: {np.mean(self.demo_stats['computation_times']):.1f}ms")
        print(f"Videos recorded: {len(self.demo_stats['videos_recorded'])}")

        print(f"\nInstructions tested:")
        for i, instruction in enumerate(self.demo_stats['instructions_tested']):
            print(f"  {i + 1}. {instruction} (sim: {self.demo_stats['similarities'][i]:.3f})")

    def _save_demo_statistics(self):
        """Save demo statistics to file"""
        stats_file = self.output_dir / f"demo_statistics_{int(time.time())}.json"
        with open(stats_file, 'w') as f:
            json.dump(self.demo_stats, f, indent=2, default=str)
        print(f"\nDemo statistics saved: {stats_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Enhanced Interactive Demo for Motion-Language Control",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to trained model (.zip file)')
    parser.add_argument('--env', type=str, default='Humanoid-v4',
                        help='MuJoCo environment name')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device (cuda/cpu/auto)')
    parser.add_argument('--output-dir', type=str, default='./demo_outputs',
                        help='Output directory for videos and results')

    # Demo modes
    parser.add_argument('--interactive', action='store_true',
                        help='Run interactive demo with user input')
    parser.add_argument('--benchmark', action='store_true',
                        help='Run benchmark on multiple instructions')
    parser.add_argument('--sequence', action='store_true',
                        help='Run instruction sequence demo')

    # Video and display options
    parser.add_argument('--no-video', action='store_true',
                        help='Disable video recording')
    parser.add_argument('--live-view', action='store_true',
                        help='Show live visualization (disables video recording)')

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

    # Video recording settings
    record_videos = not args.no_video and not args.live_view

    print(f"Enhanced Motion-Language Control Demo")
    print(f"Device: {device}")
    print(f"Video recording: {record_videos}")
    print(f"Live view: {args.live_view}")

    # Initialize demo
    demo = EnhancedInteractiveDemo(args.model_path, args.env, device, args.output_dir)

    # Run appropriate demo mode
    if args.interactive:
        demo.run_interactive_demo(record_videos=record_videos)

    elif args.benchmark:
        demo.run_benchmark_demo(record_videos=record_videos)

    elif args.sequence:
        demo.run_instruction_sequence(args.instructions, args.steps, record_video=record_videos)

    else:
        # Default: run specific instructions
        print(f"Running demo for instructions: {args.instructions}")

        for instruction in args.instructions:
            demo.run_single_instruction_demo(
                instruction,
                args.steps,
                record_video=record_videos,
                show_live=args.live_view
            )

            # Small delay between instructions
            time.sleep(2)

    print(f"\nDemo completed!")
    print(f"Output directory: {demo.output_dir}")


if __name__ == "__main__":
    main()