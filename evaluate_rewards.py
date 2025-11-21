import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from agents.hierarchical_agent import EnhancedMotionLanguageAgent


def evaluate_model_rewards(model_path, instruction="walk forward", n_episodes=10):
    """Evaluate a model and return detailed reward breakdown"""
    
    agent = EnhancedMotionLanguageAgent(
        env_name="Humanoid-v4",
        use_stability_focus=True
    )
    
    results = agent.evaluate_instruction(
        instruction=instruction,
        model_path=model_path,
        num_episodes=n_episodes,
        deterministic=True
    )
    
    # Extract episode-by-episode data
    episodes_data = []
    for ep_data in results['episode_data']:
        episodes_data.append({
            'total_reward': ep_data['total_reward'],
            'language_reward': ep_data['language_reward'],
            'env_reward': ep_data['env_reward'],
            'episode_length': len(ep_data['vx_list']),
            'falls': ep_data['stability_metrics']['fall_count'],
            'max_stable_steps': ep_data['stability_metrics']['max_stable_steps'],
        })
    
    return episodes_data, results


def plot_reward_breakdown(episodes_data, save_path=None):
    """Plot detailed reward breakdown"""
    n_episodes = len(episodes_data)
    episodes = list(range(1, n_episodes + 1))
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Total rewards
    ax = axes[0, 0]
    total_rewards = [ep['total_reward'] for ep in episodes_data]
    ax.bar(episodes, total_rewards, color='steelblue', alpha=0.7)
    ax.axhline(np.mean(total_rewards), color='red', linestyle='--', 
               label=f'Mean: {np.mean(total_rewards):.2f}')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Total Reward')
    ax.set_title('Total Reward per Episode')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Reward components
    ax = axes[0, 1]
    lang_rewards = [ep['language_reward'] for ep in episodes_data]
    env_rewards = [ep['env_reward'] for ep in episodes_data]
    
    width = 0.35
    x = np.arange(len(episodes))
    ax.bar(x - width/2, lang_rewards, width, label='Language Reward', alpha=0.7)
    ax.bar(x + width/2, env_rewards, width, label='Env Reward', alpha=0.7)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    ax.set_title('Reward Components')
    ax.set_xticks(x)
    ax.set_xticklabels(episodes)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Episode lengths
    ax = axes[1, 0]
    lengths = [ep['episode_length'] for ep in episodes_data]
    ax.bar(episodes, lengths, color='orange', alpha=0.7)
    ax.axhline(np.mean(lengths), color='red', linestyle='--',
               label=f'Mean: {np.mean(lengths):.0f}')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Steps')
    ax.set_title('Episode Length')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Stability metrics
    ax = axes[1, 1]
    falls = [ep['falls'] for ep in episodes_data]
    max_stable = [ep['max_stable_steps'] for ep in episodes_data]
    
    x = np.arange(len(episodes))
    ax.bar(x - width/2, falls, width, label='Falls', color='red', alpha=0.7)
    ax.bar(x + width/2, max_stable, width, label='Max Stable Steps', color='green', alpha=0.7)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Count')
    ax.set_title('Stability Metrics')
    ax.set_xticks(x)
    ax.set_xticklabels(episodes)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    else:
        plt.show()
    
    return fig


def compare_models(model_paths, labels, instruction="walk forward", n_episodes=5):
    """Compare multiple models side by side"""
    
    all_results = []
    
    for model_path, label in zip(model_paths, labels):
        print(f"\nEvaluating {label}...")
        episodes_data, results = evaluate_model_rewards(model_path, instruction, n_episodes)
        
        summary = {
            'label': label,
            'mean_total_reward': np.mean([ep['total_reward'] for ep in episodes_data]),
            'mean_language_reward': np.mean([ep['language_reward'] for ep in episodes_data]),
            'mean_env_reward': np.mean([ep['env_reward'] for ep in episodes_data]),
            'mean_episode_length': np.mean([ep['episode_length'] for ep in episodes_data]),
            'mean_falls': np.mean([ep['falls'] for ep in episodes_data]),
            'success_rate': results['episode_success_rate'],
        }
        
        all_results.append(summary)
    
    # Create comparison plot
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    metrics = [
        ('mean_total_reward', 'Total Reward', axes[0, 0]),
        ('mean_language_reward', 'Language Reward', axes[0, 1]),
        ('mean_env_reward', 'Env Reward', axes[0, 2]),
        ('mean_episode_length', 'Episode Length', axes[1, 0]),
        ('mean_falls', 'Falls per Episode', axes[1, 1]),
        ('success_rate', 'Success Rate', axes[1, 2]),
    ]
    
    x_pos = np.arange(len(labels))
    
    for metric_key, metric_name, ax in metrics:
        values = [r[metric_key] for r in all_results]
        bars = ax.bar(x_pos, values, alpha=0.7)
        
        # Color bars
        for i, bar in enumerate(bars):
            if metric_key == 'mean_falls':
                bar.set_color('red' if values[i] > 1 else 'green')
            else:
                bar.set_color('green' if i == np.argmax(values) else 'steelblue')
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_ylabel(metric_name)
        ax.set_title(metric_name)
        ax.grid(True, alpha=0.3)
        
        # Add value labels
        for i, v in enumerate(values):
            ax.text(i, v, f'{v:.2f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    return fig, all_results


def print_reward_summary(episodes_data, model_name="Model"):
    """Print text summary of rewards"""
    print(f"\n{'='*60}")
    print(f"{model_name} - Reward Summary")
    print(f"{'='*60}")
    
    total_rewards = [ep['total_reward'] for ep in episodes_data]
    lang_rewards = [ep['language_reward'] for ep in episodes_data]
    env_rewards = [ep['env_reward'] for ep in episodes_data]
    lengths = [ep['episode_length'] for ep in episodes_data]
    falls = [ep['falls'] for ep in episodes_data]
    
    print(f"\nTotal Reward:")
    print(f"  Mean:   {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
    print(f"  Min:    {np.min(total_rewards):.2f}")
    print(f"  Max:    {np.max(total_rewards):.2f}")
    
    print(f"\nLanguage Reward:")
    print(f"  Mean:   {np.mean(lang_rewards):.2f} ± {np.std(lang_rewards):.2f}")
    
    print(f"\nEnvironment Reward:")
    print(f"  Mean:   {np.mean(env_rewards):.2f} ± {np.std(env_rewards):.2f}")
    
    print(f"\nEpisode Length:")
    print(f"  Mean:   {np.mean(lengths):.0f} ± {np.std(lengths):.0f}")
    print(f"  Max:    {np.max(lengths):.0f}")
    
    print(f"\nStability:")
    print(f"  Mean Falls: {np.mean(falls):.2f}")
    print(f"  Episodes with 0 falls: {sum(1 for f in falls if f == 0)}/{len(falls)}")
    
    print(f"\n{'='*60}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--instruction', type=str, default='walk forward',
                       help='Instruction to evaluate')
    parser.add_argument('--episodes', type=int, default=10,
                       help='Number of episodes to evaluate')
    parser.add_argument('--output', type=str, default=None,
                       help='Output plot path')
    parser.add_argument('--compare', nargs='+', default=None,
                       help='Multiple models to compare')
    parser.add_argument('--labels', nargs='+', default=None,
                       help='Labels for comparison')
    args = parser.parse_args()
    
    if args.compare:
        labels = args.labels if args.labels else [f"Model {i+1}" for i in range(len(args.compare))]
        fig, results = compare_models(args.compare, labels, args.instruction, args.episodes)
        
        if args.output:
            fig.savefig(args.output, dpi=150, bbox_inches='tight')
            print(f"\nSaved comparison to {args.output}")
        else:
            plt.show()
        
        # Print summary table
        print(f"\n{'='*80}")
        print("COMPARISON SUMMARY")
        print(f"{'='*80}")
        print(f"{'Model':<20} {'Total Reward':<15} {'Episode Len':<15} {'Success Rate':<15}")
        print(f"{'-'*80}")
        for r in results:
            print(f"{r['label']:<20} {r['mean_total_reward']:<15.2f} "
                  f"{r['mean_episode_length']:<15.0f} {r['success_rate']*100:<15.1f}%")
    
    else:
        episodes_data, results = evaluate_model_rewards(
            args.model, args.instruction, args.episodes
        )
        
        print_reward_summary(episodes_data, Path(args.model).parent.name)
        plot_reward_breakdown(episodes_data, args.output)
