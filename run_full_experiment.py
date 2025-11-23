#!/usr/bin/env python3
"""
Full Multi-Environment Experiment Runner
Tests the conversational robot system across multiple MuJoCo environments
Generates: Videos, Plots, and Results Summary (.txt)

Author: Yunus Emre Balci
Course: IADM805
"""

import sys
import os
import time
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import numpy as np

# Add src to path
project_root = Path(__file__).parent
sys.path.append(str(project_root / "src"))

# Suppress warnings
import warnings

warnings.filterwarnings('ignore')

# Import plotting libraries
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')  # Non-interactive backend for saving plots
import seaborn as sns

# Set plot style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11

# Import project components
try:
    from conversation.deepseek_llm import DeepSeekLLM
    from conversation.task_planner import TaskPlanner
    from conversation.feedback_system import FeedbackSystem
    from agents.hierarchical_agent import EnhancedMotionLanguageAgent
    from models.motion_tokenizer import MotionTokenizer

    FULL_SYSTEM_AVAILABLE = True
except ImportError as e:
    print(f"⚠ Some components not available: {e}")
    FULL_SYSTEM_AVAILABLE = False

import torch
import gymnasium as gym


class MultiEnvironmentExperiment:
    """
    Comprehensive experiment runner for testing conversational robot
    across multiple MuJoCo environments
    """

    # Environment configurations - ALL ENVIRONMENTS
    ENVIRONMENTS = {
        "humanoid": {
            "name": "Humanoid-v4",
            "description": "27-DoF humanoid robot - most complex",
            "test_commands": [
                "walk forward",
                "walk forward slowly",
                "turn left",
                "turn right",
                "stop moving",
            ],
            "max_steps": 500,
            "difficulty": "hard"
        },
        "ant": {
            "name": "Ant-v4",
            "description": "4-legged ant robot - quadruped locomotion",
            "test_commands": [
                "walk forward",
                "walk forward slowly",
                "turn left",
                "turn right",
                "stop moving",
            ],
            "max_steps": 500,
            "difficulty": "medium"
        },
        "halfcheetah": {
            "name": "HalfCheetah-v4",
            "description": "2D running cheetah - fast locomotion",
            "test_commands": [
                "run forward",
                "walk forward",
                "walk forward slowly",
                "stop moving",
            ],
            "max_steps": 500,
            "difficulty": "medium"
        },
        "walker2d": {
            "name": "Walker2d-v4",
            "description": "2D bipedal walker - balance focused",
            "test_commands": [
                "walk forward",
                "walk forward slowly",
                "stop moving",
            ],
            "max_steps": 500,
            "difficulty": "medium"
        },
        "hopper": {
            "name": "Hopper-v4",
            "description": "Single-legged hopper - jumping locomotion",
            "test_commands": [
                "hop forward",
                "jump forward",
                "walk forward",
                "stop moving",
            ],
            "max_steps": 400,
            "difficulty": "easy"
        },
        "swimmer": {
            "name": "Swimmer-v4",
            "description": "3-link swimming robot - fluid locomotion",
            "test_commands": [
                "swim forward",
                "move forward",
                "stop moving",
            ],
            "max_steps": 500,
            "difficulty": "easy"
        },
        "inverted_pendulum": {
            "name": "InvertedPendulum-v4",
            "description": "Balance a pole - classic control",
            "test_commands": [
                "balance",
                "stand still",
                "stop moving",
            ],
            "max_steps": 200,
            "difficulty": "easy"
        }
    }

    def __init__(self,
                 output_dir: str = "./experiment_results",
                 device: str = "auto",
                 use_deepseek: bool = True,
                 record_videos: bool = True,
                 verbose: int = 1):

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        self.videos_dir = self.output_dir / "videos"
        self.plots_dir = self.output_dir / "plots"
        self.logs_dir = self.output_dir / "logs"

        for d in [self.videos_dir, self.plots_dir, self.logs_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # Device selection
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.use_deepseek = use_deepseek
        self.record_videos = record_videos
        self.verbose = verbose
        self.show_live = False  # Will be set by run_full_experiment

        # Initialize components
        self.llm = None
        self.planner = None
        self.motion_tokenizer = None
        self.agents: Dict[str, EnhancedMotionLanguageAgent] = {}

        # Results storage
        self.experiment_results: Dict[str, Dict] = {}
        self.experiment_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        print("=" * 70)
        print("  MULTI-ENVIRONMENT EXPERIMENT RUNNER")
        print("=" * 70)
        print(f"  Output directory: {self.output_dir}")
        print(f"  Device: {self.device}")
        print(f"  DeepSeek LLM: {'Enabled' if use_deepseek else 'Disabled'}")
        print(f"  Video recording: {'Enabled' if record_videos else 'Disabled'}")
        print("=" * 70)

    def initialize_components(self):
        """Initialize LLM and shared components"""
        print("\n🔧 Initializing components...")

        # Initialize DeepSeek LLM
        if self.use_deepseek and FULL_SYSTEM_AVAILABLE:
            try:
                print("  Loading DeepSeek LLM...")
                self.llm = DeepSeekLLM(use_real_model=True)
                print("  ✓ DeepSeek LLM loaded")
            except Exception as e:
                print(f"  ⚠ DeepSeek failed: {e}")
                print("  Using simulated LLM responses")
                self.llm = DeepSeekLLM(use_real_model=False)
        else:
            self.llm = DeepSeekLLM(use_real_model=False) if FULL_SYSTEM_AVAILABLE else None

        # Initialize task planner
        if FULL_SYSTEM_AVAILABLE:
            self.planner = TaskPlanner()
            self.motion_tokenizer = MotionTokenizer(device=self.device)
            print("  ✓ Task planner and motion tokenizer initialized")

    def create_agent_for_env(self, env_key: str) -> Optional[EnhancedMotionLanguageAgent]:
        """Create or retrieve agent for specific environment"""
        if env_key in self.agents:
            return self.agents[env_key]

        env_config = self.ENVIRONMENTS[env_key]
        env_name = env_config["name"]

        try:
            print(f"  Creating agent for {env_name}...")
            agent = EnhancedMotionLanguageAgent(
                env_name=env_name,
                device=self.device,
                use_stability_focus=True
            )
            self.agents[env_key] = agent
            print(f"  ✓ Agent created for {env_name}")
            return agent
        except Exception as e:
            print(f"  ✗ Failed to create agent for {env_name}: {e}")
            return None

    def run_single_environment_test(self,
                                    env_key: str,
                                    commands: List[str] = None,
                                    training_steps: int = 10000) -> Dict:
        """
        Run comprehensive test on a single environment
        """
        env_config = self.ENVIRONMENTS[env_key]
        env_name = env_config["name"]

        print(f"\n{'=' * 70}")
        print(f"  TESTING: {env_name} ({env_config['description']})")
        print(f"  Difficulty: {env_config['difficulty']}")
        print(f"{'=' * 70}")

        # Use default commands if not specified
        if commands is None:
            commands = env_config["test_commands"]

        results = {
            "environment": env_name,
            "env_key": env_key,
            "description": env_config["description"],
            "difficulty": env_config["difficulty"],
            "commands_tested": commands,
            "command_results": [],
            "aggregate_metrics": {},
            "llm_interactions": [],
            "training_info": {},
            "timestamp": datetime.now().isoformat()
        }

        # Create agent
        agent = self.create_agent_for_env(env_key)
        if agent is None:
            results["error"] = "Failed to create agent"
            return results

        # Quick training for each command
        all_similarities = []
        all_success_rates = []
        all_rewards = []

        for cmd_idx, command in enumerate(commands):
            print(f"\n  [{cmd_idx + 1}/{len(commands)}] Testing: '{command}'")

            cmd_result = {
                "command": command,
                "llm_response": None,
                "chain_of_thought": None,
                "training_metrics": {},
                "evaluation_metrics": {},
                "video_path": None
            }

            # Process with LLM
            if self.llm:
                try:
                    llm_turn = self.llm.process_user_request(command)
                    cmd_result["llm_response"] = llm_turn.llm_response
                    cmd_result["chain_of_thought"] = llm_turn.chain_of_thought
                    cmd_result["extracted_actions"] = llm_turn.extracted_actions

                    results["llm_interactions"].append({
                        "command": command,
                        "response": llm_turn.llm_response,
                        "reasoning": llm_turn.chain_of_thought,
                        "actions": llm_turn.extracted_actions
                    })
                except Exception as e:
                    print(f"    ⚠ LLM processing failed: {e}")

            # Quick training
            try:
                print(f"    Training ({training_steps} steps)...")
                train_start = time.time()

                save_path = str(self.logs_dir / f"{env_key}_{command.replace(' ', '_')}")
                Path(save_path).mkdir(parents=True, exist_ok=True)

                model_path = agent.train_on_instruction(
                    instruction=command,
                    total_timesteps=training_steps,
                    language_reward_weight=0.6,
                    save_path=save_path,
                    eval_freq=training_steps // 2,
                    n_envs=1,
                    verbose=0,
                    record_training_videos=False,
                    use_vecnormalize=False
                )

                train_time = time.time() - train_start
                cmd_result["training_metrics"]["training_time"] = train_time
                cmd_result["training_metrics"]["model_path"] = model_path
                print(f"    ✓ Training completed in {train_time:.1f}s")

            except Exception as e:
                print(f"    ✗ Training failed: {e}")
                cmd_result["training_metrics"]["error"] = str(e)
                results["command_results"].append(cmd_result)
                continue

            # Evaluation
            try:
                print(f"    Evaluating...")

                video_path = str(
                    self.videos_dir / f"{env_key}_{command.replace(' ', '_')}") if self.record_videos else None

                eval_results = agent.evaluate_instruction(
                    instruction=command,
                    model_path=model_path,
                    num_episodes=3,
                    language_reward_weight=0.6,
                    deterministic=True,
                    record_video=self.record_videos,
                    video_path=video_path,
                    render=self.show_live  # Live visualization
                )

                cmd_result["evaluation_metrics"] = {
                    "mean_similarity": eval_results.get("mean_similarity", 0.0),
                    "mean_success_rate": eval_results.get("mean_success_rate", 0.0),
                    "episode_success_rate": eval_results.get("episode_success_rate", 0.0),
                    "mean_total_reward": eval_results.get("mean_total_reward", 0.0),
                    "mean_motion_quality": eval_results.get("mean_motion_overall_quality", 0.0),
                    "mean_fall_count": eval_results.get("mean_fall_count", 0.0),
                    "mean_stable_steps": eval_results.get("mean_max_stable_steps", 0.0),
                }

                if video_path:
                    cmd_result["video_path"] = video_path

                # Aggregate
                all_similarities.append(cmd_result["evaluation_metrics"]["mean_similarity"])
                all_success_rates.append(cmd_result["evaluation_metrics"]["mean_success_rate"])
                all_rewards.append(cmd_result["evaluation_metrics"]["mean_total_reward"])

                print(f"    ✓ Similarity: {cmd_result['evaluation_metrics']['mean_similarity']:.3f}")
                print(f"    ✓ Success Rate: {cmd_result['evaluation_metrics']['mean_success_rate']:.3f}")

            except Exception as e:
                print(f"    ✗ Evaluation failed: {e}")
                cmd_result["evaluation_metrics"]["error"] = str(e)

            results["command_results"].append(cmd_result)

        # Aggregate metrics
        if all_similarities:
            results["aggregate_metrics"] = {
                "mean_similarity": float(np.mean(all_similarities)),
                "std_similarity": float(np.std(all_similarities)),
                "mean_success_rate": float(np.mean(all_success_rates)),
                "std_success_rate": float(np.std(all_success_rates)),
                "mean_reward": float(np.mean(all_rewards)),
                "std_reward": float(np.std(all_rewards)),
                "num_commands_tested": len(commands),
                "num_successful": sum(1 for s in all_success_rates if s > 0.5)
            }

        return results

    def run_full_experiment(self,
                            environments: List[str] = None,
                            training_steps: int = 10000) -> Dict:
        """
        Run experiment across all specified environments
        """
        if environments is None:
            environments = list(self.ENVIRONMENTS.keys())

        print("\n" + "#" * 70)
        print("  STARTING FULL MULTI-ENVIRONMENT EXPERIMENT")
        print("#" * 70)
        print(f"  Environments to test: {environments}")
        print(f"  Training steps per command: {training_steps}")
        print("#" * 70)

        # Initialize components
        self.initialize_components()

        experiment_start = time.time()

        # Run tests for each environment
        for env_key in environments:
            if env_key not in self.ENVIRONMENTS:
                print(f"\n⚠ Unknown environment: {env_key}")
                continue

            try:
                env_results = self.run_single_environment_test(
                    env_key=env_key,
                    training_steps=training_steps
                )
                self.experiment_results[env_key] = env_results
            except Exception as e:
                print(f"\n✗ Environment {env_key} failed: {e}")
                self.experiment_results[env_key] = {"error": str(e)}

        experiment_time = time.time() - experiment_start

        # Generate outputs
        print("\n" + "=" * 70)
        print("  GENERATING OUTPUTS")
        print("=" * 70)

        # Save JSON results
        self._save_json_results()

        # Generate plots
        self._generate_plots()

        # Generate text summary
        self._generate_text_summary(experiment_time)

        print("\n" + "#" * 70)
        print("  EXPERIMENT COMPLETED!")
        print("#" * 70)
        print(f"  Total time: {experiment_time / 60:.1f} minutes")
        print(f"  Results saved to: {self.output_dir}")
        print("#" * 70)

        return self.experiment_results

    def _save_json_results(self):
        """Save detailed results to JSON"""
        json_path = self.output_dir / f"results_{self.experiment_timestamp}.json"

        with open(json_path, 'w') as f:
            json.dump(self.experiment_results, f, indent=2, default=str)

        print(f"  ✓ JSON results saved: {json_path}")

    def _generate_plots(self):
        """Generate comprehensive visualization plots"""
        print("  Generating plots...")

        # Prepare data
        env_names = []
        similarities = []
        success_rates = []
        rewards = []

        for env_key, results in self.experiment_results.items():
            if "aggregate_metrics" in results and results["aggregate_metrics"]:
                env_names.append(self.ENVIRONMENTS[env_key]["name"])
                similarities.append(results["aggregate_metrics"].get("mean_similarity", 0))
                success_rates.append(results["aggregate_metrics"].get("mean_success_rate", 0))
                rewards.append(results["aggregate_metrics"].get("mean_reward", 0))

        if not env_names:
            print("    ⚠ No data to plot")
            return

        # Plot 1: Environment Comparison Bar Chart
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle("Multi-Environment Experiment Results", fontsize=14, fontweight='bold')

        colors = sns.color_palette("husl", len(env_names))

        # Similarity
        axes[0].bar(env_names, similarities, color=colors)
        axes[0].set_ylabel("Motion-Language Similarity")
        axes[0].set_title("Similarity Score by Environment")
        axes[0].set_ylim([0, 1])
        axes[0].tick_params(axis='x', rotation=45)
        for i, v in enumerate(similarities):
            axes[0].text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=9)

        # Success Rate
        axes[1].bar(env_names, success_rates, color=colors)
        axes[1].set_ylabel("Success Rate")
        axes[1].set_title("Success Rate by Environment")
        axes[1].set_ylim([0, 1])
        axes[1].tick_params(axis='x', rotation=45)
        for i, v in enumerate(success_rates):
            axes[1].text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=9)

        # Reward
        axes[2].bar(env_names, rewards, color=colors)
        axes[2].set_ylabel("Mean Reward")
        axes[2].set_title("Mean Reward by Environment")
        axes[2].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plot_path = self.plots_dir / f"environment_comparison_{self.experiment_timestamp}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"    ✓ Environment comparison plot: {plot_path}")

        # Plot 2: Command-level Performance Heatmap
        self._generate_command_heatmap()

        # Plot 3: Radar Chart for Environment Capabilities
        self._generate_radar_chart()

    def _generate_command_heatmap(self):
        """Generate heatmap of command performance across environments"""

        # Collect all unique commands
        all_commands = set()
        for results in self.experiment_results.values():
            if "command_results" in results:
                for cmd_result in results["command_results"]:
                    all_commands.add(cmd_result["command"])

        if not all_commands:
            return

        all_commands = sorted(list(all_commands))
        env_keys = [k for k in self.experiment_results.keys()
                    if "command_results" in self.experiment_results[k]]

        if not env_keys:
            return

        # Build matrix
        matrix = np.zeros((len(all_commands), len(env_keys)))

        for j, env_key in enumerate(env_keys):
            results = self.experiment_results[env_key]
            cmd_scores = {}
            for cmd_result in results.get("command_results", []):
                if "evaluation_metrics" in cmd_result:
                    score = cmd_result["evaluation_metrics"].get("mean_similarity", 0)
                    cmd_scores[cmd_result["command"]] = score

            for i, cmd in enumerate(all_commands):
                matrix[i, j] = cmd_scores.get(cmd, np.nan)

        # Plot heatmap
        fig, ax = plt.subplots(figsize=(10, 8))

        # Mask NaN values
        mask = np.isnan(matrix)

        sns.heatmap(matrix,
                    xticklabels=[self.ENVIRONMENTS[k]["name"] for k in env_keys],
                    yticklabels=all_commands,
                    annot=True,
                    fmt='.3f',
                    cmap='RdYlGn',
                    mask=mask,
                    vmin=0, vmax=1,
                    ax=ax)

        ax.set_title("Command Performance Across Environments\n(Motion-Language Similarity)",
                     fontsize=12, fontweight='bold')
        ax.set_xlabel("Environment")
        ax.set_ylabel("Command")

        plt.tight_layout()
        plot_path = self.plots_dir / f"command_heatmap_{self.experiment_timestamp}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"    ✓ Command heatmap: {plot_path}")

    def _generate_radar_chart(self):
        """Generate radar chart comparing environments"""

        metrics = ['Similarity', 'Success Rate', 'Stability', 'Quality']

        env_data = {}
        for env_key, results in self.experiment_results.items():
            if "aggregate_metrics" in results and results["aggregate_metrics"]:
                agg = results["aggregate_metrics"]
                env_data[self.ENVIRONMENTS[env_key]["name"]] = [
                    agg.get("mean_similarity", 0),
                    agg.get("mean_success_rate", 0),
                    0.5,  # Placeholder for stability
                    0.5,  # Placeholder for quality
                ]

        if not env_data:
            return

        # Create radar chart
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

        colors = sns.color_palette("husl", len(env_data))

        for idx, (env_name, values) in enumerate(env_data.items()):
            values += values[:1]  # Complete the circle
            ax.plot(angles, values, 'o-', linewidth=2, label=env_name, color=colors[idx])
            ax.fill(angles, values, alpha=0.25, color=colors[idx])

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics)
        ax.set_ylim(0, 1)
        ax.set_title("Environment Capability Comparison", fontsize=12, fontweight='bold')
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

        plt.tight_layout()
        plot_path = self.plots_dir / f"radar_chart_{self.experiment_timestamp}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"    ✓ Radar chart: {plot_path}")

    def _generate_text_summary(self, experiment_time: float):
        """Generate comprehensive text summary"""

        summary_path = self.output_dir / f"results_summary_{self.experiment_timestamp}.txt"

        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write("  MULTI-ENVIRONMENT EXPERIMENT RESULTS SUMMARY\n")
            f.write("=" * 70 + "\n\n")

            f.write(f"Experiment Timestamp: {self.experiment_timestamp}\n")
            f.write(f"Total Duration: {experiment_time / 60:.2f} minutes\n")
            f.write(f"Device: {self.device}\n")
            f.write(f"DeepSeek LLM: {'Enabled' if self.use_deepseek else 'Disabled'}\n")
            f.write(f"Video Recording: {'Enabled' if self.record_videos else 'Disabled'}\n\n")

            f.write("-" * 70 + "\n")
            f.write("  ENVIRONMENT RESULTS\n")
            f.write("-" * 70 + "\n\n")

            for env_key, results in self.experiment_results.items():
                env_config = self.ENVIRONMENTS.get(env_key, {})
                f.write(f"\n{'=' * 50}\n")
                f.write(f"Environment: {env_config.get('name', env_key)}\n")
                f.write(f"Description: {env_config.get('description', 'N/A')}\n")
                f.write(f"Difficulty: {env_config.get('difficulty', 'N/A')}\n")
                f.write(f"{'=' * 50}\n\n")

                if "error" in results:
                    f.write(f"ERROR: {results['error']}\n\n")
                    continue

                # Aggregate metrics
                if "aggregate_metrics" in results and results["aggregate_metrics"]:
                    agg = results["aggregate_metrics"]
                    f.write("AGGREGATE METRICS:\n")
                    f.write(
                        f"  Mean Similarity:    {agg.get('mean_similarity', 0):.4f} (+/- {agg.get('std_similarity', 0):.4f})\n")
                    f.write(
                        f"  Mean Success Rate:  {agg.get('mean_success_rate', 0):.4f} (+/- {agg.get('std_success_rate', 0):.4f})\n")
                    f.write(
                        f"  Mean Reward:        {agg.get('mean_reward', 0):.2f} (+/- {agg.get('std_reward', 0):.2f})\n")
                    f.write(f"  Commands Tested:    {agg.get('num_commands_tested', 0)}\n")
                    f.write(f"  Successful (>0.5):  {agg.get('num_successful', 0)}\n\n")

                # Individual command results
                if "command_results" in results:
                    f.write("COMMAND-LEVEL RESULTS:\n")
                    f.write("-" * 40 + "\n")

                    for cmd_result in results["command_results"]:
                        f.write(f"\n  Command: \"{cmd_result['command']}\"\n")

                        if cmd_result.get("chain_of_thought"):
                            f.write(f"    LLM Reasoning: {cmd_result['chain_of_thought'][:100]}...\n")

                        if "evaluation_metrics" in cmd_result:
                            metrics = cmd_result["evaluation_metrics"]
                            f.write(f"    Similarity:    {metrics.get('mean_similarity', 0):.4f}\n")
                            f.write(f"    Success Rate:  {metrics.get('mean_success_rate', 0):.4f}\n")
                            f.write(f"    Reward:        {metrics.get('mean_total_reward', 0):.2f}\n")
                            f.write(f"    Falls:         {metrics.get('mean_fall_count', 0):.1f}\n")

                        if cmd_result.get("video_path"):
                            f.write(f"    Video:         {cmd_result['video_path']}\n")

                # LLM interactions
                if "llm_interactions" in results and results["llm_interactions"]:
                    f.write("\nLLM INTERACTIONS:\n")
                    f.write("-" * 40 + "\n")
                    for interaction in results["llm_interactions"][:3]:  # First 3
                        f.write(f"\n  Input: \"{interaction['command']}\"\n")
                        f.write(f"  Reasoning: {interaction.get('reasoning', 'N/A')[:200]}\n")
                        f.write(f"  Actions: {interaction.get('actions', [])}\n")

            # Summary table
            f.write("\n" + "=" * 70 + "\n")
            f.write("  SUMMARY TABLE\n")
            f.write("=" * 70 + "\n\n")

            f.write(f"{'Environment':<20} {'Similarity':>12} {'Success':>12} {'Reward':>12}\n")
            f.write("-" * 60 + "\n")

            for env_key, results in self.experiment_results.items():
                if "aggregate_metrics" in results and results["aggregate_metrics"]:
                    agg = results["aggregate_metrics"]
                    env_name = self.ENVIRONMENTS.get(env_key, {}).get("name", env_key)
                    f.write(f"{env_name:<20} {agg.get('mean_similarity', 0):>12.4f} "
                            f"{agg.get('mean_success_rate', 0):>12.4f} "
                            f"{agg.get('mean_reward', 0):>12.2f}\n")

            f.write("\n" + "=" * 70 + "\n")
            f.write("  END OF REPORT\n")
            f.write("=" * 70 + "\n")

        print(f"  ✓ Text summary saved: {summary_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Multi-Environment Experiment Runner for Motion-Language Control",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--output-dir', type=str, default='./experiment_results',
                        help='Output directory for results')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device (cuda/cpu/auto)')
    parser.add_argument('--envs', nargs='+',
                        default=['humanoid', 'ant', 'halfcheetah', 'hopper', 'walker2d'],
                        help='Environments to test (humanoid, ant, halfcheetah, hopper, walker2d, swimmer)')
    parser.add_argument('--training-steps', type=int, default=20000,
                        help='Training steps per command')
    parser.add_argument('--no-deepseek', action='store_true',
                        help='Disable DeepSeek LLM')
    parser.add_argument('--no-video', action='store_true',
                        help='Disable video recording')
    parser.add_argument('--live', action='store_true',
                        help='Show live visualization during evaluation')
    parser.add_argument('--quick', action='store_true',
                        help='Quick test with minimal steps')

    args = parser.parse_args()

    # Quick mode adjustments
    if args.quick:
        args.training_steps = 2000
        args.envs = ['humanoid']

    print("\n🚀 Starting Multi-Environment Experiment")
    print(f"   Environments: {args.envs}")
    print(f"   Training steps: {args.training_steps}")

    # Create experiment runner
    experiment = MultiEnvironmentExperiment(
        output_dir=args.output_dir,
        device=args.device,
        use_deepseek=not args.no_deepseek,
        record_videos=not args.no_video,
        verbose=1
    )
    experiment.show_live = args.live  # Enable live visualization

    # Run experiment
    results = experiment.run_full_experiment(
        environments=args.envs,
        training_steps=args.training_steps
    )

    print("\n✅ Experiment completed!")
    print(f"   Check results in: {args.output_dir}")


if __name__ == "__main__":
    main()