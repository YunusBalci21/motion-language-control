"""
Multi-Environment Evaluation and Comparison
Evaluates trained models across different environments for paper
"""

import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from agents.hierarchical_agent import EnhancedMotionLanguageAgent


def evaluate_environment(env_name, model_path, instruction, n_episodes=10):
    """Evaluate a single environment"""
    
    print(f"\nEvaluating {env_name}...")
    print("-"*60)
    
    agent = EnhancedMotionLanguageAgent(
        env_name=env_name,
        use_stability_focus=True
    )
    
    results = agent.evaluate_instruction(
        instruction=instruction,
        model_path=model_path,
        num_episodes=n_episodes,
        deterministic=True
    )
    
    print(f"✓ {env_name} evaluation complete")
    print(f"  Mean Reward: {results['mean_total_reward']:.2f}")
    print(f"  Success Rate: {results['episode_success_rate']*100:.1f}%")
    print(f"  Episode Length: {results['mean_max_stable_steps']:.0f}")
    
    return results


def evaluate_all_environments():
    """Evaluate all trained environments"""
    
    environments = [
        {
            'name': 'Humanoid-v4',
            'display': 'Humanoid',
            'model': './checkpoints/phase3_normal_walk/final_model_walk_forward.zip',
            'instruction': 'walk forward'
        },
        {
            'name': 'Hopper-v4',
            'display': 'Hopper',
            'model': './checkpoints/hopper_forward/final_model_hop_forward.zip',
            'instruction': 'hop forward'
        },
        {
            'name': 'Ant-v4',
            'display': 'Ant',
            'model': './checkpoints/ant_forward/final_model_walk_forward.zip',
            'instruction': 'walk forward'
        },
        {
            'name': 'Walker2d-v4',
            'display': 'Walker2d',
            'model': './checkpoints/walker2d_forward/final_model_walk_forward.zip',
            'instruction': 'walk forward'
        },
        {
            'name': 'HalfCheetah-v4',
            'display': 'HalfCheetah',
            'model': './checkpoints/halfcheetah_forward/final_model_run_forward.zip',
            'instruction': 'run forward'
        }
    ]
    
    results_all = []
    
    for env_config in environments:
        if not Path(env_config['model']).exists():
            print(f"⚠ Model not found: {env_config['model']}")
            continue
        
        try:
            results = evaluate_environment(
                env_config['name'],
                env_config['model'],
                env_config['instruction'],
                n_episodes=10
            )
            
            results_all.append({
                'env': env_config['display'],
                'results': results
            })
        
        except Exception as e:
            print(f"⚠ Evaluation failed for {env_config['name']}: {e}")
    
    # Generate comparison plots
    if len(results_all) > 0:
        generate_comparison_plots(results_all)
        generate_comparison_table(results_all)
    
    return results_all


def generate_comparison_plots(results_all):
    """Generate comparison plots across environments"""
    
    envs = [r['env'] for r in results_all]
    rewards = [r['results']['mean_total_reward'] for r in results_all]
    lengths = [r['results']['mean_max_stable_steps'] for r in results_all]
    success = [r['results']['episode_success_rate']*100 for r in results_all]
    falls = [r['results']['mean_fall_count'] for r in results_all]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Total Rewards
    ax = axes[0, 0]
    bars = ax.bar(range(len(envs)), rewards, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax.set_xticks(range(len(envs)))
    ax.set_xticklabels(envs, rotation=45, ha='right')
    ax.set_ylabel('Mean Total Reward', fontsize=12, fontweight='bold')
    ax.set_title('Total Reward by Environment', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, rewards):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{val:.0f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Episode Length
    ax = axes[0, 1]
    bars = ax.bar(range(len(envs)), lengths, alpha=0.8, color='orange', edgecolor='black', linewidth=1.5)
    ax.set_xticks(range(len(envs)))
    ax.set_xticklabels(envs, rotation=45, ha='right')
    ax.set_ylabel('Mean Episode Length (steps)', fontsize=12, fontweight='bold')
    ax.set_title('Episode Length by Environment', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, lengths):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{val:.0f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Success Rate
    ax = axes[1, 0]
    bars = ax.bar(range(len(envs)), success, alpha=0.8, color='green', edgecolor='black', linewidth=1.5)
    ax.set_xticks(range(len(envs)))
    ax.set_xticklabels(envs, rotation=45, ha='right')
    ax.set_ylabel('Success Rate (%)', fontsize=12, fontweight='bold')
    ax.set_title('Success Rate by Environment', fontsize=13, fontweight='bold')
    ax.set_ylim(0, 110)
    ax.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, success):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{val:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Falls per Episode
    ax = axes[1, 1]
    bars = ax.bar(range(len(envs)), falls, alpha=0.8, color='red', edgecolor='black', linewidth=1.5)
    ax.set_xticks(range(len(envs)))
    ax.set_xticklabels(envs, rotation=45, ha='right')
    ax.set_ylabel('Falls per Episode', fontsize=12, fontweight='bold')
    ax.set_title('Stability by Environment', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, falls):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{val:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.suptitle('Motion-Language Control: Multi-Environment Comparison',
                fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    output_path = Path('./reports')
    output_path.mkdir(exist_ok=True)
    plt.savefig(output_path / 'multi_env_comparison.png', dpi=200, bbox_inches='tight')
    
    print(f"\n✓ Saved comparison plot: {output_path / 'multi_env_comparison.png'}")


def generate_comparison_table(results_all):
    """Generate comparison table for paper"""
    
    output_path = Path('./reports')
    output_path.mkdir(exist_ok=True)
    
    # Markdown table
    md_file = output_path / 'multi_env_results.md'
    with open(md_file, 'w', encoding='utf-8') as f:
        f.write("# Multi-Environment Results\n\n")
        f.write("## Performance Comparison\n\n")
        f.write("| Environment | Mean Reward | Episode Length | Success Rate | Falls/Episode |\n")
        f.write("|-------------|-------------|----------------|--------------|---------------|\n")
        
        for r in results_all:
            env = r['env']
            res = r['results']
            f.write(f"| {env:<11} | {res['mean_total_reward']:>11.2f} | "
                   f"{res['mean_max_stable_steps']:>14.0f} | "
                   f"{res['episode_success_rate']*100:>11.1f}% | "
                   f"{res['mean_fall_count']:>13.2f} |\n")
        
        f.write("\n## Detailed Results\n\n")
        
        for r in results_all:
            env = r['env']
            res = r['results']
            f.write(f"### {env}\n\n")
            f.write(f"- **Mean Total Reward**: {res['mean_total_reward']:.2f}\n")
            f.write(f"- **Mean Language Reward**: {res['mean_language_reward']:.2f}\n")
            f.write(f"- **Mean Env Reward**: {res['mean_env_reward']:.2f}\n")
            f.write(f"- **Episode Length**: {res['mean_max_stable_steps']:.0f} steps\n")
            f.write(f"- **Success Rate**: {res['episode_success_rate']*100:.1f}%\n")
            f.write(f"- **Falls per Episode**: {res['mean_fall_count']:.2f}\n\n")
    
    print(f"✓ Saved results table: {md_file}")
    
    # LaTeX table
    tex_file = output_path / 'multi_env_results.tex'
    with open(tex_file, 'w', encoding='utf-8') as f:
        f.write("\\begin{table}[h]\n")
        f.write("\\centering\n")
        f.write("\\caption{Performance Across Different MuJoCo Environments}\n")
        f.write("\\label{tab:multi-env}\n")
        f.write("\\begin{tabular}{lcccc}\n")
        f.write("\\toprule\n")
        f.write("Environment & Reward & Length & Success & Falls \\\\\n")
        f.write("\\midrule\n")
        
        for r in results_all:
            env = r['env']
            res = r['results']
            f.write(f"{env} & {res['mean_total_reward']:.0f} & "
                   f"{res['mean_max_stable_steps']:.0f} & "
                   f"{res['episode_success_rate']*100:.0f}\\% & "
                   f"{res['mean_fall_count']:.1f} \\\\\n")
        
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
    
    print(f"✓ Saved LaTeX table: {tex_file}")
    
    # Console summary
    print("\n" + "="*80)
    print("MULTI-ENVIRONMENT SUMMARY")
    print("="*80)
    print(f"{'Environment':<15} {'Reward':<12} {'Length':<12} {'Success':<12} {'Falls':<10}")
    print("-"*80)
    
    for r in results_all:
        env = r['env']
        res = r['results']
        print(f"{env:<15} {res['mean_total_reward']:<12.0f} "
              f"{res['mean_max_stable_steps']:<12.0f} "
              f"{res['episode_success_rate']*100:<11.1f}% "
              f"{res['mean_fall_count']:<10.2f}")
    
    print("="*80)


if __name__ == "__main__":
    print("="*60)
    print("MULTI-ENVIRONMENT EVALUATION")
    print("="*60)
    
    results = evaluate_all_environments()
    
    print("\n✓ All evaluations complete")
    print("✓ Results saved to ./reports/")
    print("\nFiles created:")
    print("  - multi_env_comparison.png")
    print("  - multi_env_results.md")
    print("  - multi_env_results.tex")
