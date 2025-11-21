#!/usr/bin/env python3
"""
Comprehensive Reward Tracking and Visualization System
Tracks all rewards during training and creates beautiful visualizations
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
from datetime import datetime
from collections import defaultdict
import pandas as pd
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import Logger

# Set style for publication-quality plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10


class RewardTracker(BaseCallback):
    """
    Advanced callback for tracking ALL rewards during training
    Tracks: episode rewards, language rewards, similarities, success rates, etc.
    """

    def __init__(
            self,
            instruction: str,
            save_path: str = "./logs/rewards/",
            save_freq: int = 1000,
            verbose: int = 1,
    ):
        super().__init__(verbose)
        self.instruction = instruction
        self.save_path = Path(save_path)
        self.save_path.mkdir(parents=True, exist_ok=True)
        self.save_freq = save_freq

        # Training metrics
        self.timesteps: List[int] = []
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []

        # Language-specific metrics
        self.language_rewards: List[float] = []
        self.env_rewards: List[float] = []
        self.similarities: List[float] = []
        self.success_rates: List[float] = []

        # Motion quality metrics
        self.motion_smoothness: List[float] = []
        self.motion_stability: List[float] = []
        self.motion_naturalness: List[float] = []
        self.motion_quality: List[float] = []

        # Stability metrics
        self.fall_counts: List[int] = []
        self.stable_steps: List[int] = []
        self.stability_bonuses: List[float] = []

        # Per-step tracking (for detailed analysis)
        self.step_language_rewards: List[float] = []
        self.step_similarities: List[float] = []

        # Episode tracking
        self.current_episode_data = {
            'total_reward': 0.0,
            'language_reward': 0.0,
            'env_reward': 0.0,
            'similarities': [],
            'success_rates': [],
            'length': 0,
        }

        self.episode_count = 0
        self.total_timesteps = 0

    def _on_step(self) -> bool:
        """Called at each environment step"""
        self.total_timesteps += 1

        # Get info from the first environment
        if len(self.locals['infos']) > 0:
            info = self.locals['infos'][0]

            # Track step-level metrics
            lang_reward = info.get('language_reward', 0.0)
            similarity = info.get('motion_language_similarity', 0.0)

            self.step_language_rewards.append(float(lang_reward))
            self.step_similarities.append(float(similarity))

            # Accumulate episode data
            self.current_episode_data['total_reward'] += self.locals['rewards'][0]
            self.current_episode_data['language_reward'] += lang_reward
            self.current_episode_data['env_reward'] += info.get('original_reward', 0.0)
            self.current_episode_data['similarities'].append(similarity)
            self.current_episode_data['success_rates'].append(info.get('success_rate', 0.0))
            self.current_episode_data['length'] += 1

            # Check if episode is done
            if self.locals['dones'][0]:
                self._on_episode_end(info)

        # Save periodically
        if self.total_timesteps % self.save_freq == 0:
            self.save_metrics()

        return True

    def _on_episode_end(self, info: Dict):
        """Called when an episode ends"""
        self.episode_count += 1

        # Store episode metrics
        self.timesteps.append(self.total_timesteps)
        self.episode_rewards.append(self.current_episode_data['total_reward'])
        self.episode_lengths.append(self.current_episode_data['length'])

        # Language metrics
        self.language_rewards.append(self.current_episode_data['language_reward'])
        self.env_rewards.append(self.current_episode_data['env_reward'])

        # Aggregated metrics
        similarities = self.current_episode_data['similarities']
        success_rates = self.current_episode_data['success_rates']

        self.similarities.append(np.mean(similarities) if similarities else 0.0)
        self.success_rates.append(np.mean(success_rates) if success_rates else 0.0)

        # Motion quality
        self.motion_smoothness.append(info.get('motion_smoothness', 0.0))
        self.motion_stability.append(info.get('motion_stability', 0.0))
        self.motion_naturalness.append(info.get('motion_naturalness', 0.0))
        self.motion_quality.append(info.get('motion_overall_quality', 0.0))

        # Stability
        self.fall_counts.append(info.get('fall_count', 0))
        self.stable_steps.append(info.get('consecutive_stable_steps', 0))
        self.stability_bonuses.append(info.get('stability_bonus', 0.0))

        # Print progress
        if self.verbose > 0 and self.episode_count % 10 == 0:
            print(f"\n[Episode {self.episode_count}] Timesteps: {self.total_timesteps}")
            print(f"  Total Reward: {self.episode_rewards[-1]:.2f}")
            print(f"  Language Reward: {self.language_rewards[-1]:.2f}")
            print(f"  Similarity: {self.similarities[-1]:.3f}")
            print(f"  Success Rate: {self.success_rates[-1]:.3f}")
            print(f"  Falls: {self.fall_counts[-1]}")

        # Reset episode data
        self.current_episode_data = {
            'total_reward': 0.0,
            'language_reward': 0.0,
            'env_reward': 0.0,
            'similarities': [],
            'success_rates': [],
            'length': 0,
        }

    def save_metrics(self):
        """Save all metrics to JSON"""
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        instruction_safe = self.instruction.replace(' ', '_')
        filename = self.save_path / f"metrics_{instruction_safe}_{timestamp}.json"

        # Convert numpy types to Python native types
        def convert(obj):
            if isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, list):
                return [convert(item) for item in obj]
            return obj

        metrics = {
            'instruction': self.instruction,
            'episode_count': int(self.episode_count),
            'total_timesteps': int(self.total_timesteps),
            'timesteps': convert(self.timesteps),
            'episode_rewards': convert(self.episode_rewards),
            'episode_lengths': convert(self.episode_lengths),
            'language_rewards': convert(self.language_rewards),
            'env_rewards': convert(self.env_rewards),
            'similarities': convert(self.similarities),
            'success_rates': convert(self.success_rates),
            'motion_smoothness': convert(self.motion_smoothness),
            'motion_stability': convert(self.motion_stability),
            'motion_naturalness': convert(self.motion_naturalness),
            'motion_quality': convert(self.motion_quality),
            'fall_counts': convert(self.fall_counts),
            'stable_steps': convert(self.stable_steps),
            'stability_bonuses': convert(self.stability_bonuses),
        }

        with open(filename, 'w') as f:
            json.dump(metrics, f, indent=2)

        if self.verbose > 1:
            print(f"Saved metrics to {filename}")

    def get_summary(self) -> Dict:
        """Get summary statistics"""
        if len(self.episode_rewards) == 0:
            return {}

        return {
            'total_episodes': self.episode_count,
            'total_timesteps': self.total_timesteps,
            'mean_episode_reward': float(np.mean(self.episode_rewards)),
            'std_episode_reward': float(np.std(self.episode_rewards)),
            'mean_language_reward': float(np.mean(self.language_rewards)),
            'mean_similarity': float(np.mean(self.similarities)),
            'mean_success_rate': float(np.mean(self.success_rates)),
            'mean_motion_quality': float(np.mean(self.motion_quality)),
            'mean_falls': float(np.mean(self.fall_counts)),
            'mean_stable_steps': float(np.mean(self.stable_steps)),
            'final_episode_reward': float(self.episode_rewards[-1]) if self.episode_rewards else 0.0,
            'final_similarity': float(self.similarities[-1]) if self.similarities else 0.0,
        }


class RewardVisualizer:
    """
    Creates beautiful visualizations of training metrics
    """

    def __init__(self, save_dir: str = "./plots/"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Color schemes
        self.colors = {
            'total': '#2E86AB',
            'language': '#A23B72',
            'env': '#F18F01',
            'similarity': '#C73E1D',
            'success': '#6A994E',
            'quality': '#BC4B51',
            'stability': '#5E548E',
        }

    def plot_reward_progression(
            self,
            metrics: Dict,
            title: Optional[str] = None,
            smoothing_window: int = 10,
    ):
        """
        Plot reward progression over training
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(title or f"Training Progress: {metrics['instruction']}",
                     fontsize=14, fontweight='bold')

        timesteps = np.array(metrics['timesteps'])

        # 1. Episode Rewards
        ax = axes[0, 0]
        episode_rewards = np.array(metrics['episode_rewards'])
        ax.plot(timesteps, episode_rewards, alpha=0.3, color=self.colors['total'],
                label='Raw')

        # Smoothed
        if len(episode_rewards) >= smoothing_window:
            smoothed = self._smooth(episode_rewards, smoothing_window)
            ax.plot(timesteps, smoothed, color=self.colors['total'],
                    linewidth=2, label=f'Smoothed ({smoothing_window})')

        ax.set_xlabel('Timesteps')
        ax.set_ylabel('Episode Reward')
        ax.set_title('Total Episode Reward')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 2. Language vs Environment Rewards
        ax = axes[0, 1]
        lang_rewards = np.array(metrics['language_rewards'])
        env_rewards = np.array(metrics['env_rewards'])

        if len(lang_rewards) >= smoothing_window:
            lang_smoothed = self._smooth(lang_rewards, smoothing_window)
            env_smoothed = self._smooth(env_rewards, smoothing_window)
            ax.plot(timesteps, lang_smoothed, color=self.colors['language'],
                    linewidth=2, label='Language Reward')
            ax.plot(timesteps, env_smoothed, color=self.colors['env'],
                    linewidth=2, label='Environment Reward')
        else:
            ax.plot(timesteps, lang_rewards, color=self.colors['language'], label='Language')
            ax.plot(timesteps, env_rewards, color=self.colors['env'], label='Environment')

        ax.set_xlabel('Timesteps')
        ax.set_ylabel('Reward')
        ax.set_title('Language vs Environment Rewards')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 3. Motion-Language Similarity
        ax = axes[1, 0]
        similarities = np.array(metrics['similarities'])
        ax.plot(timesteps, similarities, alpha=0.3, color=self.colors['similarity'])

        if len(similarities) >= smoothing_window:
            sim_smoothed = self._smooth(similarities, smoothing_window)
            ax.plot(timesteps, sim_smoothed, color=self.colors['similarity'], linewidth=2)

        ax.axhline(y=0.6, color='red', linestyle='--', alpha=0.5, label='Success Threshold')
        ax.set_xlabel('Timesteps')
        ax.set_ylabel('Similarity Score')
        ax.set_title('Motion-Language Similarity')
        ax.set_ylim([0, 1])
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 4. Success Rate and Motion Quality
        ax = axes[1, 1]
        success_rates = np.array(metrics['success_rates'])
        motion_quality = np.array(metrics['motion_quality'])

        if len(success_rates) >= smoothing_window:
            success_smoothed = self._smooth(success_rates, smoothing_window)
            quality_smoothed = self._smooth(motion_quality, smoothing_window)
            ax.plot(timesteps, success_smoothed, color=self.colors['success'],
                    linewidth=2, label='Success Rate')
            ax.plot(timesteps, quality_smoothed, color=self.colors['quality'],
                    linewidth=2, label='Motion Quality')
        else:
            ax.plot(timesteps, success_rates, color=self.colors['success'], label='Success')
            ax.plot(timesteps, motion_quality, color=self.colors['quality'], label='Quality')

        ax.set_xlabel('Timesteps')
        ax.set_ylabel('Score')
        ax.set_title('Success Rate & Motion Quality')
        ax.set_ylim([0, 1])
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save
        filename = self.save_dir / f"reward_progression_{metrics['instruction'].replace(' ', '_')}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"✓ Saved reward progression plot: {filename}")

        return fig

    def plot_stability_metrics(self, metrics: Dict, smoothing_window: int = 10):
        """
        Plot stability-specific metrics
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(f"Stability Metrics: {metrics['instruction']}",
                     fontsize=14, fontweight='bold')

        timesteps = np.array(metrics['timesteps'])

        # 1. Fall Count
        ax = axes[0]
        fall_counts = np.array(metrics['fall_counts'])
        ax.plot(timesteps, fall_counts, alpha=0.3, color='red')

        if len(fall_counts) >= smoothing_window:
            falls_smoothed = self._smooth(fall_counts, smoothing_window)
            ax.plot(timesteps, falls_smoothed, color='red', linewidth=2)

        ax.set_xlabel('Timesteps')
        ax.set_ylabel('Fall Count per Episode')
        ax.set_title('Falls Over Time (Lower is Better)')
        ax.grid(True, alpha=0.3)

        # 2. Stable Steps
        ax = axes[1]
        stable_steps = np.array(metrics['stable_steps'])
        ax.plot(timesteps, stable_steps, alpha=0.3, color='green')

        if len(stable_steps) >= smoothing_window:
            stable_smoothed = self._smooth(stable_steps, smoothing_window)
            ax.plot(timesteps, stable_smoothed, color='green', linewidth=2)

        ax.set_xlabel('Timesteps')
        ax.set_ylabel('Max Consecutive Stable Steps')
        ax.set_title('Stability Duration (Higher is Better)')
        ax.grid(True, alpha=0.3)

        # 3. Stability Bonus
        ax = axes[2]
        stability_bonuses = np.array(metrics['stability_bonuses'])
        ax.plot(timesteps, stability_bonuses, alpha=0.3, color=self.colors['stability'])

        if len(stability_bonuses) >= smoothing_window:
            bonus_smoothed = self._smooth(stability_bonuses, smoothing_window)
            ax.plot(timesteps, bonus_smoothed, color=self.colors['stability'], linewidth=2)

        ax.set_xlabel('Timesteps')
        ax.set_ylabel('Stability Bonus Reward')
        ax.set_title('Stability Bonus Over Time')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        filename = self.save_dir / f"stability_metrics_{metrics['instruction'].replace(' ', '_')}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"✓ Saved stability metrics plot: {filename}")

        return fig

    def compare_instructions(
            self,
            metrics_list: List[Dict],
            metric_name: str = 'episode_rewards',
            smoothing_window: int = 10,
    ):
        """
        Compare multiple instructions on the same plot
        """
        fig, ax = plt.subplots(figsize=(12, 6))

        colors = plt.cm.tab10(np.linspace(0, 1, len(metrics_list)))

        for i, metrics in enumerate(metrics_list):
            instruction = metrics['instruction']
            timesteps = np.array(metrics['timesteps'])
            values = np.array(metrics[metric_name])

            if len(values) >= smoothing_window:
                smoothed = self._smooth(values, smoothing_window)
                ax.plot(timesteps, smoothed, color=colors[i], linewidth=2,
                        label=instruction, alpha=0.8)
            else:
                ax.plot(timesteps, values, color=colors[i], linewidth=2,
                        label=instruction, alpha=0.8)

        ax.set_xlabel('Timesteps', fontsize=12)
        ax.set_ylabel(metric_name.replace('_', ' ').title(), fontsize=12)
        ax.set_title(f'Comparison: {metric_name.replace("_", " ").title()}',
                     fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        filename = self.save_dir / f"comparison_{metric_name}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"✓ Saved comparison plot: {filename}")

        return fig

    def create_summary_table(self, metrics_list: List[Dict]) -> pd.DataFrame:
        """
        Create a summary table comparing all instructions
        """
        summaries = []

        for metrics in metrics_list:
            if len(metrics['episode_rewards']) == 0:
                continue

            summary = {
                'Instruction': metrics['instruction'],
                'Episodes': metrics['episode_count'],
                'Mean Reward': f"{np.mean(metrics['episode_rewards']):.2f}",
                'Std Reward': f"{np.std(metrics['episode_rewards']):.2f}",
                'Mean Similarity': f"{np.mean(metrics['similarities']):.3f}",
                'Mean Success Rate': f"{np.mean(metrics['success_rates']):.3f}",
                'Mean Falls': f"{np.mean(metrics['fall_counts']):.1f}",
                'Mean Stable Steps': f"{np.mean(metrics['stable_steps']):.0f}",
                'Motion Quality': f"{np.mean(metrics['motion_quality']):.3f}",
            }
            summaries.append(summary)

        df = pd.DataFrame(summaries)

        # Save to CSV
        filename = self.save_dir / "summary_table.csv"
        df.to_csv(filename, index=False)
        print(f"✓ Saved summary table: {filename}")

        return df

    def _smooth(self, values: np.ndarray, window: int) -> np.ndarray:
        """Apply moving average smoothing"""
        if len(values) < window:
            return values
        return np.convolve(values, np.ones(window) / window, mode='same')


def load_metrics(filepath: str) -> Dict:
    """Load metrics from JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)


def demo_visualization():
    """
    Demo: Create visualizations from existing data
    """
    print("=" * 60)
    print("Reward Visualization Demo")
    print("=" * 60)

    # Create sample metrics (replace with actual data)
    sample_metrics = {
        'instruction': 'walk forward',
        'episode_count': 100,
        'total_timesteps': 100000,
        'timesteps': list(range(0, 100000, 1000)),
        'episode_rewards': list(100 + 50 * np.random.randn(100) + np.linspace(0, 200, 100)),
        'episode_lengths': [1000] * 100,
        'language_rewards': list(50 + 20 * np.random.randn(100) + np.linspace(0, 100, 100)),
        'env_rewards': list(50 + 30 * np.random.randn(100) + np.linspace(0, 100, 100)),
        'similarities': list(0.3 + 0.1 * np.random.randn(100) + np.linspace(0, 0.4, 100)),
        'success_rates': list(0.2 + 0.1 * np.random.randn(100) + np.linspace(0, 0.6, 100)),
        'motion_smoothness': list(0.5 + 0.1 * np.random.randn(100)),
        'motion_stability': list(0.4 + 0.1 * np.random.randn(100) + np.linspace(0, 0.3, 100)),
        'motion_naturalness': list(0.5 + 0.1 * np.random.randn(100)),
        'motion_quality': list(0.45 + 0.1 * np.random.randn(100) + np.linspace(0, 0.3, 100)),
        'fall_counts': list(np.maximum(0, 10 - np.linspace(0, 8, 100) + np.random.randn(100))),
        'stable_steps': list(100 + 50 * np.random.randn(100) + np.linspace(0, 400, 100)),
        'stability_bonuses': list(0.1 + 0.05 * np.random.randn(100) + np.linspace(0, 0.3, 100)),
    }

    visualizer = RewardVisualizer()

    # Create plots
    print("\n1. Creating reward progression plot...")
    visualizer.plot_reward_progression(sample_metrics)

    print("\n2. Creating stability metrics plot...")
    visualizer.plot_stability_metrics(sample_metrics)

    print("\n✓ Demo completed! Check ./plots/ directory")


if __name__ == "__main__":
    demo_visualization()