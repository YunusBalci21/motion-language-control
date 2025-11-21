import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tensorboard.backend.event_processing import event_accumulator
import json


def load_tensorboard_data(log_dir):
    """Load reward data from TensorBoard logs"""
    ea = event_accumulator.EventAccumulator(str(log_dir))
    ea.Reload()
    
    data = {}
    
    # Get available tags
    tags = ea.Tags()
    
    # Extract episode reward
    if 'rollout/ep_rew_mean' in tags['scalars']:
        rewards = ea.Scalars('rollout/ep_rew_mean')
        data['steps'] = [r.step for r in rewards]
        data['rewards'] = [r.value for r in rewards]
    
    # Extract episode length
    if 'rollout/ep_len_mean' in tags['scalars']:
        lengths = ea.Scalars('rollout/ep_len_mean')
        data['ep_lengths'] = [r.value for r in lengths]
    
    return data


def plot_training_rewards(checkpoint_dir, save_path=None):
    """Plot training rewards from checkpoint directory"""
    checkpoint_path = Path(checkpoint_dir)
    
    # Find tensorboard logs
    log_dirs = list(checkpoint_path.glob('**/events.out.tfevents.*'))
    if not log_dirs:
        print(f"No TensorBoard logs found in {checkpoint_dir}")
        return
    
    # Load data
    data = load_tensorboard_data(log_dirs[0].parent)
    
    if not data:
        print("No reward data found")
        return
    
    # Create plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot rewards
    ax1.plot(data['steps'], data['rewards'], linewidth=2)
    ax1.set_xlabel('Training Steps')
    ax1.set_ylabel('Episode Reward')
    ax1.set_title('Training Rewards Over Time')
    ax1.grid(True, alpha=0.3)
    
    # Plot episode length
    if 'ep_lengths' in data:
        ax2.plot(data['steps'], data['ep_lengths'], linewidth=2, color='orange')
        ax2.set_xlabel('Training Steps')
        ax2.set_ylabel('Episode Length (steps)')
        ax2.set_title('Episode Length Over Time')
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    else:
        plt.show()
    
    return fig


def compare_phases(phase_dirs, labels=None, save_path=None):
    """Compare rewards across multiple training phases"""
    if labels is None:
        labels = [f"Phase {i+1}" for i in range(len(phase_dirs))]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    colors = ['blue', 'green', 'red', 'purple']
    
    for i, (phase_dir, label) in enumerate(zip(phase_dirs, labels)):
        phase_path = Path(phase_dir)
        log_dirs = list(phase_path.glob('**/events.out.tfevents.*'))
        
        if not log_dirs:
            continue
        
        data = load_tensorboard_data(log_dirs[0].parent)
        
        if not data:
            continue
        
        color = colors[i % len(colors)]
        
        # Plot rewards
        ax1.plot(data['steps'], data['rewards'], 
                label=label, linewidth=2, color=color, alpha=0.8)
        
        # Plot episode lengths
        if 'ep_lengths' in data:
            ax2.plot(data['steps'], data['ep_lengths'], 
                    label=label, linewidth=2, color=color, alpha=0.8)
    
    ax1.set_xlabel('Training Steps')
    ax1.set_ylabel('Episode Reward')
    ax1.set_title('Reward Comparison Across Phases')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.set_xlabel('Training Steps')
    ax2.set_ylabel('Episode Length')
    ax2.set_title('Episode Length Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved comparison to {save_path}")
    else:
        plt.show()
    
    return fig


def create_reward_summary(checkpoint_dirs, labels=None):
    """Create summary statistics for each phase"""
    if labels is None:
        labels = [Path(d).name for d in checkpoint_dirs]
    
    summary = []
    
    for phase_dir, label in zip(checkpoint_dirs, labels):
        phase_path = Path(phase_dir)
        log_dirs = list(phase_path.glob('**/events.out.tfevents.*'))
        
        if not log_dirs:
            continue
        
        data = load_tensorboard_data(log_dirs[0].parent)
        
        if not data or 'rewards' not in data:
            continue
        
        rewards = np.array(data['rewards'])
        lengths = np.array(data['ep_lengths']) if 'ep_lengths' in data else None
        
        stats = {
            'Phase': label,
            'Final Reward': rewards[-1] if len(rewards) > 0 else 0,
            'Mean Reward': np.mean(rewards),
            'Max Reward': np.max(rewards),
            'Final Episode Length': lengths[-1] if lengths is not None and len(lengths) > 0 else 0,
            'Max Episode Length': np.max(lengths) if lengths is not None else 0,
        }
        
        summary.append(stats)
    
    return summary


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/phase1_stand',
                       help='Checkpoint directory to plot')
    parser.add_argument('--compare', nargs='+', default=None,
                       help='Multiple checkpoint dirs to compare')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file path')
    parser.add_argument('--summary', action='store_true',
                       help='Print summary statistics')
    args = parser.parse_args()
    
    if args.compare:
        compare_phases(args.compare, save_path=args.output)
        
        if args.summary:
            summary = create_reward_summary(args.compare)
            print("\n=== REWARD SUMMARY ===")
            for stat in summary:
                print(f"\n{stat['Phase']}:")
                print(f"  Final Reward: {stat['Final Reward']:.2f}")
                print(f"  Mean Reward: {stat['Mean Reward']:.2f}")
                print(f"  Max Reward: {stat['Max Reward']:.2f}")
                print(f"  Final Episode Length: {stat['Final Episode Length']:.0f}")
                print(f"  Max Episode Length: {stat['Max Episode Length']:.0f}")
    else:
        plot_training_rewards(args.checkpoint, save_path=args.output)
