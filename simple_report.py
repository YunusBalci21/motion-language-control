import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Add src to path
project_root = Path.cwd()
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

from agents.hierarchical_agent import EnhancedMotionLanguageAgent


def simple_report(phase_dirs):
    """Generate simple reports for all phases"""
    
    reports_dir = Path("./reports")
    reports_dir.mkdir(exist_ok=True)
    
    all_results = []
    
    for phase_dir in phase_dirs:
        print(f"\nEvaluating {phase_dir.name}...")
        print("="*60)
        
        # Find model
        model_files = list(phase_dir.glob('final_model_*.zip'))
        if not model_files:
            print(f"No model found in {phase_dir}")
            continue
        
        model_path = model_files[0]
        
        # Determine instruction
        if 'stand' in model_path.name:
            instruction = "stand still"
        elif 'slowly' in model_path.name:
            instruction = "walk forward slowly"
        elif 'steadily' in model_path.name:
            instruction = "walk forward steadily"
        else:
            instruction = "walk forward"
        
        # Evaluate
        agent = EnhancedMotionLanguageAgent(
            env_name="Humanoid-v4",
            use_stability_focus=True
        )
        
        results = agent.evaluate_instruction(
            instruction=instruction,
            model_path=str(model_path),
            num_episodes=10,
            deterministic=True
        )
        
        all_results.append({
            'name': phase_dir.name,
            'results': results,
            'instruction': instruction
        })
        
        # Save text summary
        text_file = reports_dir / f"{phase_dir.name}_summary.txt"
        with open(text_file, 'w') as f:
            f.write(f"PHASE: {phase_dir.name}\n")
            f.write(f"Instruction: {instruction}\n")
            f.write("="*60 + "\n\n")
            
            f.write(f"Mean Total Reward:      {results['mean_total_reward']:.2f}\n")
            f.write(f"Mean Language Reward:   {results['mean_language_reward']:.2f}\n")
            f.write(f"Mean Env Reward:        {results['mean_env_reward']:.2f}\n")
            f.write(f"Mean Episode Length:    {results['mean_max_stable_steps']:.0f} steps\n")
            f.write(f"Success Rate:           {results['episode_success_rate']*100:.1f}%\n")
            f.write(f"Mean Falls per Episode: {results['mean_fall_count']:.2f}\n")
            f.write("\n")
            
            f.write("Episode Rewards:\n")
            for i, ep in enumerate(results['episode_data'], 1):
                f.write(f"  Episode {i}: {ep['total_reward']:.2f}\n")
        
        print(f"✓ Saved: {text_file}")
    
    # Create comparison plot
    if len(all_results) > 1:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        labels = [r['name'] for r in all_results]
        
        # Total rewards
        ax = axes[0, 0]
        values = [r['results']['mean_total_reward'] for r in all_results]
        ax.bar(range(len(labels)), values, alpha=0.7)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_ylabel('Mean Total Reward')
        ax.set_title('Mean Total Reward by Phase')
        ax.grid(True, alpha=0.3)
        
        # Episode lengths
        ax = axes[0, 1]
        values = [r['results']['mean_max_stable_steps'] for r in all_results]
        ax.bar(range(len(labels)), values, alpha=0.7, color='orange')
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_ylabel('Mean Episode Length (steps)')
        ax.set_title('Episode Length by Phase')
        ax.grid(True, alpha=0.3)
        
        # Success rate
        ax = axes[1, 0]
        values = [r['results']['episode_success_rate']*100 for r in all_results]
        ax.bar(range(len(labels)), values, alpha=0.7, color='green')
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_ylabel('Success Rate (%)')
        ax.set_title('Success Rate by Phase')
        ax.grid(True, alpha=0.3)
        
        # Falls
        ax = axes[1, 1]
        values = [r['results']['mean_fall_count'] for r in all_results]
        ax.bar(range(len(labels)), values, alpha=0.7, color='red')
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_ylabel('Mean Falls per Episode')
        ax.set_title('Falls by Phase')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        comp_file = reports_dir / "phases_comparison.png"
        plt.savefig(comp_file, dpi=150, bbox_inches='tight')
        print(f"\n✓ Saved comparison: {comp_file}")
    
    # Print summary table
    print("\n" + "="*80)
    print("SUMMARY TABLE")
    print("="*80)
    print(f"{'Phase':<20} {'Total Reward':<15} {'Episode Length':<18} {'Success Rate':<15}")
    print("-"*80)
    for r in all_results:
        res = r['results']
        print(f"{r['name']:<20} {res['mean_total_reward']:<15.2f} "
              f"{res['mean_max_stable_steps']:<18.0f} "
              f"{res['episode_success_rate']*100:<15.1f}%")
    
    print("\n✓ All reports saved to ./reports/")


if __name__ == "__main__":
    checkpoints_dir = Path("./checkpoints")
    
    # Find only phase directories
    phase_dirs = sorted([d for d in checkpoints_dir.iterdir() 
                        if d.is_dir() and d.name.startswith('phase')])
    
    if not phase_dirs:
        print("No phase directories found")
    else:
        print(f"Found {len(phase_dirs)} phases")
        simple_report(phase_dirs)
