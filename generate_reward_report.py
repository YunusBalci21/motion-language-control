import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from agents.hierarchical_agent import EnhancedMotionLanguageAgent
from plot_training_rewards import load_tensorboard_data, create_reward_summary


def generate_complete_report(checkpoint_dir, output_dir="./reports"):
    """Generate complete performance report with rewards"""
    
    checkpoint_path = Path(checkpoint_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    model_name = checkpoint_path.name
    
    print(f"Generating report for {model_name}...")
    print("="*60)
    
    # 1. Load training data
    log_dirs = list(checkpoint_path.glob('**/events.out.tfevents.*'))
    if log_dirs:
        train_data = load_tensorboard_data(log_dirs[0].parent)
    else:
        train_data = None
        print("Warning: No training logs found")
    
    # 2. Find model file
    model_files = list(checkpoint_path.glob('final_model_*.zip'))
    if not model_files:
        print("Error: No model file found")
        return
    
    model_path = model_files[0]
    
    # 3. Evaluate model
    print("\nEvaluating model...")
    agent = EnhancedMotionLanguageAgent(
        env_name="Humanoid-v4",
        use_stability_focus=True
    )
    
    results = agent.evaluate_instruction(
        instruction="walk forward",
        model_path=str(model_path),
        num_episodes=10,
        deterministic=True
    )
    
    # 4. Create comprehensive plot
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Training rewards
    if train_data and 'rewards' in train_data:
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(train_data['steps'], train_data['rewards'], linewidth=2, color='steelblue')
        ax1.set_xlabel('Training Steps')
        ax1.set_ylabel('Episode Reward')
        ax1.set_title(f'{model_name} - Training Rewards', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Add final value annotation
        final_reward = train_data['rewards'][-1]
        ax1.axhline(final_reward, color='red', linestyle='--', alpha=0.5)
        ax1.text(train_data['steps'][-1], final_reward, 
                f' Final: {final_reward:.1f}', va='center')
    
    # Episode-by-episode evaluation rewards
    ax2 = fig.add_subplot(gs[1, 0])
    episode_rewards = [ep['total_reward'] for ep in results['episode_data']]
    episodes = list(range(1, len(episode_rewards) + 1))
    ax2.bar(episodes, episode_rewards, color='green', alpha=0.7)
    ax2.axhline(np.mean(episode_rewards), color='red', linestyle='--',
               label=f'Mean: {np.mean(episode_rewards):.1f}')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Total Reward')
    ax2.set_title('Evaluation: Total Reward')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Reward components
    ax3 = fig.add_subplot(gs[1, 1])
    lang_rewards = [ep['language_reward'] for ep in results['episode_data']]
    env_rewards = [ep['env_reward'] for ep in results['episode_data']]
    
    width = 0.35
    x = np.arange(len(episodes))
    ax3.bar(x - width/2, lang_rewards, width, label='Language', alpha=0.7)
    ax3.bar(x + width/2, env_rewards, width, label='Environment', alpha=0.7)
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Reward')
    ax3.set_title('Evaluation: Reward Components')
    ax3.set_xticks(x)
    ax3.set_xticklabels(episodes)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Episode lengths
    ax4 = fig.add_subplot(gs[1, 2])
    lengths = [len(ep['vx_list']) for ep in results['episode_data']]
    ax4.bar(episodes, lengths, color='orange', alpha=0.7)
    ax4.axhline(np.mean(lengths), color='red', linestyle='--',
               label=f'Mean: {np.mean(lengths):.0f}')
    ax4.set_xlabel('Episode')
    ax4.set_ylabel('Steps')
    ax4.set_title('Evaluation: Episode Length')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Summary statistics table
    ax5 = fig.add_subplot(gs[2, :])
    ax5.axis('off')
    
    # Create summary data
    summary_data = [
        ['Metric', 'Value'],
        ['', ''],
        ['TRAINING', ''],
        ['Final Training Reward', f"{train_data['rewards'][-1]:.2f}" if train_data else 'N/A'],
        ['Mean Training Reward', f"{np.mean(train_data['rewards']):.2f}" if train_data else 'N/A'],
        ['', ''],
        ['EVALUATION', ''],
        ['Mean Total Reward', f"{results['mean_total_reward']:.2f}"],
        ['Mean Language Reward', f"{results['mean_language_reward']:.2f}"],
        ['Mean Env Reward', f"{results['mean_env_reward']:.2f}"],
        ['Mean Episode Length', f"{results['mean_max_stable_steps']:.0f} steps"],
        ['Success Rate', f"{results['episode_success_rate']*100:.1f}%"],
        ['Mean Falls per Episode', f"{results['mean_fall_count']:.2f}"],
    ]
    
    table = ax5.table(cellText=summary_data, cellLoc='left', loc='center',
                     colWidths=[0.4, 0.4])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style header
    for i in range(2):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Style section headers
    for row in [2, 6]:
        for col in range(2):
            table[(row, col)].set_facecolor('#E0E0E0')
            table[(row, col)].set_text_props(weight='bold')
    
    # Main title
    fig.suptitle(f'Performance Report: {model_name}', 
                fontsize=16, fontweight='bold', y=0.98)
    
    # Save
    output_file = output_path / f"{model_name}_performance_report.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n✓ Report saved to: {output_file}")
    
    # Also save text summary
    text_file = output_path / f"{model_name}_summary.txt"
    with open(text_file, 'w') as f:
        f.write(f"PERFORMANCE REPORT: {model_name}\n")
        f.write("="*60 + "\n\n")
        
        f.write("TRAINING METRICS:\n")
        f.write("-"*60 + "\n")
        if train_data:
            f.write(f"Final Training Reward:  {train_data['rewards'][-1]:.2f}\n")
            f.write(f"Mean Training Reward:   {np.mean(train_data['rewards']):.2f}\n")
            f.write(f"Total Training Steps:   {train_data['steps'][-1]}\n")
        f.write("\n")
        
        f.write("EVALUATION METRICS:\n")
        f.write("-"*60 + "\n")
        f.write(f"Mean Total Reward:      {results['mean_total_reward']:.2f}\n")
        f.write(f"Mean Language Reward:   {results['mean_language_reward']:.2f}\n")
        f.write(f"Mean Env Reward:        {results['mean_env_reward']:.2f}\n")
        f.write(f"Mean Episode Length:    {results['mean_max_stable_steps']:.0f} steps\n")
        f.write(f"Success Rate:           {results['episode_success_rate']*100:.1f}%\n")
        f.write(f"Mean Falls per Episode: {results['mean_fall_count']:.2f}\n")
        f.write("\n")
        
        f.write("EPISODE-BY-EPISODE REWARDS:\n")
        f.write("-"*60 + "\n")
        for i, ep in enumerate(results['episode_data'], 1):
            f.write(f"Episode {i}: {ep['total_reward']:.2f}\n")
    
    print(f"✓ Text summary saved to: {text_file}")
    
    return output_file, text_file


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Checkpoint directory')
    parser.add_argument('--output', type=str, default='./reports',
                       help='Output directory for reports')
    args = parser.parse_args()
    
    generate_complete_report(args.checkpoint, args.output)
