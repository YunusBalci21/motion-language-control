#!/usr/bin/env python3
"""
Visualize Diagnostic Results
Analyzes and creates plots from your diagnostic JSON files
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 10


def load_diagnostic(filepath):
    """Load diagnostic JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)


def visualize_diagnostics(diagnostic_files):
    """
    Create comprehensive visualizations from diagnostic files
    """
    print("=" * 60)
    print("Diagnostic Results Visualization")
    print("=" * 60)

    # Load all diagnostics
    diagnostics = []
    for file in diagnostic_files:
        try:
            data = load_diagnostic(file)
            diagnostics.append(data)
            print(f"✓ Loaded: {file}")
        except FileNotFoundError:
            print(f"⚠ File not found: {file}")
            continue

    if not diagnostics:
        print("❌ No diagnostic files found!")
        return

    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # Title
    fig.suptitle('Diagnostic Results Analysis', fontsize=16, fontweight='bold')

    # === Plot 1: Similarity Heatmap (First Diagnostic) ===
    ax1 = fig.add_subplot(gs[0, :2])
    data = diagnostics[0]
    similarity_matrix = data['similarity_matrix']

    instructions = list(similarity_matrix.keys())
    motion_types = ['Stable Walking', 'Chaotic Motion', 'Minimal Motion']

    similarity_values = []
    for instruction in instructions:
        values = [similarity_matrix[instruction][motion_type]
                  for motion_type in motion_types]
        similarity_values.append(values)

    im = ax1.imshow(similarity_values, cmap='YlOrRd', aspect='auto', vmin=0.3, vmax=0.7)
    ax1.set_xticks(range(len(motion_types)))
    ax1.set_yticks(range(len(instructions)))
    ax1.set_xticklabels(motion_types, rotation=45, ha='right')
    ax1.set_yticklabels(instructions)
    ax1.set_title('Similarity Scores by Instruction and Motion Type', fontweight='bold')

    # Add text annotations
    for i in range(len(instructions)):
        for j in range(len(motion_types)):
            text = ax1.text(j, i, f'{similarity_values[i][j]:.3f}',
                            ha="center", va="center", color="black", fontsize=9)

    plt.colorbar(im, ax=ax1, label='Similarity Score')

    # === Plot 2: Similarity Statistics ===
    ax2 = fig.add_subplot(gs[0, 2])

    all_similarities = []
    for data in diagnostics:
        sim_matrix = data['similarity_matrix']
        for instruction in sim_matrix:
            for motion_type in sim_matrix[instruction]:
                all_similarities.append(sim_matrix[instruction][motion_type])

    ax2.hist(all_similarities, bins=20, color='#2E86AB', alpha=0.7, edgecolor='black')
    ax2.axvline(np.mean(all_similarities), color='red', linestyle='--',
                linewidth=2, label=f'Mean: {np.mean(all_similarities):.3f}')
    ax2.axvline(0.6, color='green', linestyle='--', linewidth=2,
                label='Success Threshold: 0.6', alpha=0.7)
    ax2.set_xlabel('Similarity Score')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Distribution of Similarity Scores', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # === Plot 3: Environment Stats Comparison ===
    ax3 = fig.add_subplot(gs[1, 0])

    diag_labels = [f'Run {i + 1}' for i in range(len(diagnostics))]
    mean_rewards = [d['environment_stats']['mean_reward'] for d in diagnostics]
    mean_similarities = [d['environment_stats']['mean_similarity'] for d in diagnostics]
    mean_lang_rewards = [d['environment_stats']['mean_language_reward'] for d in diagnostics]

    x = np.arange(len(diagnostics))
    width = 0.25

    ax3.bar(x - width, mean_rewards, width, label='Mean Reward', color='#2E86AB')
    ax3.bar(x, mean_similarities, width, label='Mean Similarity', color='#A23B72')
    ax3.bar(x + width, mean_lang_rewards, width, label='Lang Reward', color='#F18F01')

    ax3.set_xlabel('Diagnostic Run')
    ax3.set_ylabel('Score')
    ax3.set_title('Environment Stats Comparison', fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(diag_labels)
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')

    # === Plot 4: Instruction Performance ===
    ax4 = fig.add_subplot(gs[1, 1])

    # Average similarity for each instruction across all diagnostics
    instruction_scores = {}
    for data in diagnostics:
        for instruction, motion_dict in data['similarity_matrix'].items():
            if instruction not in instruction_scores:
                instruction_scores[instruction] = []
            # Average across motion types
            avg_score = np.mean(list(motion_dict.values()))
            instruction_scores[instruction].append(avg_score)

    instructions_sorted = sorted(instruction_scores.keys(),
                                 key=lambda x: np.mean(instruction_scores[x]),
                                 reverse=True)
    avg_scores = [np.mean(instruction_scores[inst]) for inst in instructions_sorted]
    std_scores = [np.std(instruction_scores[inst]) for inst in instructions_sorted]

    y_pos = np.arange(len(instructions_sorted))
    ax4.barh(y_pos, avg_scores, xerr=std_scores, color='#6A994E', alpha=0.7)
    ax4.set_yticks(y_pos)
    ax4.set_yticklabels(instructions_sorted)
    ax4.set_xlabel('Average Similarity Score')
    ax4.set_title('Instruction Performance Ranking', fontweight='bold')
    ax4.axvline(x=0.6, color='red', linestyle='--', alpha=0.5, label='Success Threshold')
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='x')

    # === Plot 5: Motion Type Preference ===
    ax5 = fig.add_subplot(gs[1, 2])

    motion_type_scores = {'Stable Walking': [], 'Chaotic Motion': [], 'Minimal Motion': []}

    for data in diagnostics:
        for instruction, motion_dict in data['similarity_matrix'].items():
            for motion_type, score in motion_dict.items():
                motion_type_scores[motion_type].append(score)

    motion_types_list = list(motion_type_scores.keys())
    avg_scores = [np.mean(motion_type_scores[mt]) for mt in motion_types_list]

    colors = ['#2E86AB', '#A23B72', '#F18F01']
    ax5.bar(motion_types_list, avg_scores, color=colors, alpha=0.7)
    ax5.set_ylabel('Average Similarity Score')
    ax5.set_title('Average Score by Motion Type', fontweight='bold')
    ax5.tick_params(axis='x', rotation=45)
    ax5.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for i, (mt, score) in enumerate(zip(motion_types_list, avg_scores)):
        ax5.text(i, score + 0.01, f'{score:.3f}', ha='center', va='bottom', fontweight='bold')

    # === Plot 6: Detailed Similarity Matrix (Second Diagnostic if available) ===
    ax6 = fig.add_subplot(gs[2, :])

    if len(diagnostics) > 1:
        data = diagnostics[1]
        similarity_matrix = data['similarity_matrix']

        # Create a different view - all instructions vs all motion types
        all_data = []
        row_labels = []

        for instruction in similarity_matrix:
            row = [similarity_matrix[instruction][mt] for mt in motion_types]
            all_data.append(row)
            row_labels.append(instruction)

        im = ax6.imshow(all_data, cmap='RdYlGn', aspect='auto', vmin=0.3, vmax=0.7)
        ax6.set_xticks(range(len(motion_types)))
        ax6.set_yticks(range(len(row_labels)))
        ax6.set_xticklabels(motion_types)
        ax6.set_yticklabels(row_labels)
        ax6.set_title('Detailed Similarity Matrix (Run 2)', fontweight='bold')

        # Add annotations
        for i in range(len(row_labels)):
            for j in range(len(motion_types)):
                text = ax6.text(j, i, f'{all_data[i][j]:.3f}',
                                ha="center", va="center", color="black", fontsize=9)

        plt.colorbar(im, ax=ax6, label='Similarity Score')
    else:
        ax6.text(0.5, 0.5, 'No second diagnostic file available',
                 ha='center', va='center', transform=ax6.transAxes, fontsize=12)
        ax6.axis('off')

    plt.savefig('diagnostic_visualization.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Visualization saved to: diagnostic_visualization.png")

    # Print summary statistics
    print("\n" + "=" * 60)
    print("Summary Statistics")
    print("=" * 60)

    for i, data in enumerate(diagnostics):
        print(f"\nDiagnostic Run {i + 1}:")
        stats = data['similarity_stats']
        env_stats = data['environment_stats']

        print(f"  Similarity Scores:")
        print(f"    Mean: {stats['mean']:.3f}")
        print(f"    Std:  {stats['std']:.3f}")
        print(f"    Min:  {stats['min']:.3f}")
        print(f"    Max:  {stats['max']:.3f}")

        print(f"  Environment Stats:")
        print(f"    Mean Reward: {env_stats['mean_reward']:.3f}")
        print(f"    Mean Similarity: {env_stats['mean_similarity']:.3f}")
        print(f"    Mean Language Reward: {env_stats['mean_language_reward']:.3f}")

        print(f"  Diagnostics:")
        for diag in data['diagnostics']:
            print(f"    {diag}")

        print(f"  Recommendations:")
        for rec in data['recommendations'][:3]:  # Show first 3
            print(f"    {rec}")

    plt.show()


def create_comparison_report(diagnostic_files):
    """Create a text report comparing diagnostic results"""

    print("\n" + "=" * 60)
    print("COMPARISON REPORT")
    print("=" * 60)

    diagnostics = []
    for file in diagnostic_files:
        try:
            data = load_diagnostic(file)
            diagnostics.append(data)
        except FileNotFoundError:
            continue

    if len(diagnostics) < 2:
        print("Need at least 2 diagnostic files for comparison")
        return

    print(f"\nComparing {len(diagnostics)} diagnostic runs...\n")

    # Compare similarity improvements
    print("📊 Similarity Score Changes:")
    for instruction in diagnostics[0]['similarity_matrix']:
        scores = []
        for data in diagnostics:
            if instruction in data['similarity_matrix']:
                avg = np.mean(list(data['similarity_matrix'][instruction].values()))
                scores.append(avg)

        if len(scores) >= 2:
            improvement = scores[-1] - scores[0]
            arrow = "📈" if improvement > 0 else "📉"
            print(f"  {arrow} '{instruction}':")
            for i, score in enumerate(scores):
                print(f"      Run {i + 1}: {score:.3f}")
            print(f"      Change: {improvement:+.3f} ({improvement / scores[0] * 100:+.1f}%)")

    # Compare environment stats
    print("\n🎯 Environment Performance:")
    metrics = ['mean_reward', 'mean_similarity', 'mean_language_reward']
    for metric in metrics:
        values = [d['environment_stats'][metric] for d in diagnostics]
        improvement = values[-1] - values[0]
        arrow = "📈" if improvement > 0 else "📉"
        print(f"  {arrow} {metric.replace('_', ' ').title()}:")
        for i, val in enumerate(values):
            print(f"      Run {i + 1}: {val:.3f}")
        print(f"      Change: {improvement:+.3f}")

    # Improvement summary
    print("\n✨ Overall Assessment:")

    sim_improvement = (diagnostics[-1]['similarity_stats']['mean'] -
                       diagnostics[0]['similarity_stats']['mean'])

    if sim_improvement > 0.05:
        print("  ✓ Significant improvement in similarity scores!")
    elif sim_improvement > 0:
        print("  → Slight improvement in similarity scores")
    else:
        print("  ⚠ No improvement in similarity scores")

    reward_improvement = (diagnostics[-1]['environment_stats']['mean_reward'] -
                          diagnostics[0]['environment_stats']['mean_reward'])

    if reward_improvement > 0.5:
        print("  ✓ Good improvement in environment rewards!")
    elif reward_improvement > 0:
        print("  → Slight improvement in environment rewards")
    else:
        print("  ⚠ No improvement in environment rewards")


if __name__ == "__main__":
    # Look for diagnostic files
    diagnostic_files = [
        "diagnostic_results_1757522425.json",
        "diagnostic_results_1757522705.json",
    ]

    print("Looking for diagnostic files...")
    found_files = [f for f in diagnostic_files if Path(f).exists()]

    if not found_files:
        print("❌ No diagnostic files found in current directory!")
        print("Expected files:")
        for f in diagnostic_files:
            print(f"  - {f}")
    else:
        print(f"Found {len(found_files)} diagnostic file(s)")

        # Create visualizations
        visualize_diagnostics(found_files)

        # Create comparison report
        if len(found_files) >= 2:
            create_comparison_report(found_files)