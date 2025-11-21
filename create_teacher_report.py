import matplotlib.pyplot as plt
import numpy as np

# Data from results
phases = ['Phase 2\nSlow Walk', 'Phase 3\nNormal Walk', 'Phase 4\nEndurance']
rewards = [3629.79, 3488.68, 3417.52]
lang_rewards = [136.41, 147.18, 136.35]
env_rewards = [1332.18, 1327.99, 1337.48]
episode_lengths = [218, 205, 200]
success_rates = [100.0, 100.0, 100.0]
falls = [13.40, 1.50, 2.70]

fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)

# Main title
fig.suptitle('Humanoid Walking Training Results\nMotion-Language Control with Stability Enhancements',
             fontsize=18, fontweight='bold')

# 1. Total Rewards (BIG)
ax1 = fig.add_subplot(gs[0, :])
bars = ax1.bar(phases, rewards, color=['#2196F3', '#4CAF50', '#FF9800'], alpha=0.8, edgecolor='black', linewidth=2)
ax1.set_ylabel('Mean Total Reward', fontsize=14, fontweight='bold')
ax1.set_title('Training Performance by Phase', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3, axis='y')
for i, (bar, val) in enumerate(zip(bars, rewards)):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width() / 2., height,
             f'{val:.0f}',
             ha='center', va='bottom', fontsize=12, fontweight='bold')
ax1.set_ylim(0, max(rewards) * 1.15)

# 2. Reward Components
ax2 = fig.add_subplot(gs[1, 0])
x = np.arange(len(phases))
width = 0.35
bars1 = ax2.bar(x - width / 2, lang_rewards, width, label='Language Reward', alpha=0.8, color='#9C27B0')
bars2 = ax2.bar(x + width / 2, env_rewards, width, label='Environment Reward', alpha=0.8, color='#00BCD4')
ax2.set_ylabel('Reward', fontsize=11, fontweight='bold')
ax2.set_title('Reward Components', fontsize=12, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(phases)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3, axis='y')

# 3. Episode Length
ax3 = fig.add_subplot(gs[1, 1])
bars = ax3.bar(phases, episode_lengths, color=['#FF5722', '#FFC107', '#8BC34A'], alpha=0.8, edgecolor='black',
               linewidth=1.5)
ax3.set_ylabel('Steps', fontsize=11, fontweight='bold')
ax3.set_title('Episode Length (Survival)', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3, axis='y')
for bar, val in zip(bars, episode_lengths):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width() / 2., height,
             f'{val:.0f}',
             ha='center', va='bottom', fontsize=11, fontweight='bold')

# 4. Success Rate
ax4 = fig.add_subplot(gs[1, 2])
bars = ax4.bar(phases, success_rates, color='#4CAF50', alpha=0.8, edgecolor='black', linewidth=1.5)
ax4.set_ylabel('Success Rate (%)', fontsize=11, fontweight='bold')
ax4.set_title('Task Success Rate', fontsize=12, fontweight='bold')
ax4.set_ylim(0, 110)
ax4.grid(True, alpha=0.3, axis='y')
for bar in bars:
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width() / 2., height,
             '100%',
             ha='center', va='bottom', fontsize=11, fontweight='bold')

# 5. Stability (Falls)
ax5 = fig.add_subplot(gs[2, 0])
bars = ax5.bar(phases, falls, color=['#F44336', '#4CAF50', '#FF9800'], alpha=0.8, edgecolor='black', linewidth=1.5)
ax5.set_ylabel('Falls per Episode', fontsize=11, fontweight='bold')
ax5.set_title('Stability Analysis', fontsize=12, fontweight='bold')
ax5.grid(True, alpha=0.3, axis='y')
for bar, val in zip(bars, falls):
    height = bar.get_height()
    color = '#4CAF50' if val < 3 else '#F44336'
    ax5.text(bar.get_x() + bar.get_width() / 2., height,
             f'{val:.1f}',
             ha='center', va='bottom', fontsize=11, fontweight='bold', color=color)

# 6. Summary Table
ax6 = fig.add_subplot(gs[2, 1:])
ax6.axis('off')

table_data = [
    ['Metric', 'Phase 2', 'Phase 3', 'Phase 4'],
    ['Total Reward', '3,630', '3,489', '3,418'],
    ['Episode Length', '218 steps', '205 steps', '200 steps'],
    ['Success Rate', '100%', '100%', '100%'],
    ['Falls per Episode', '13.4', '1.5', '2.7'],
    ['Stability Score', 'Poor', '⭐ Best', 'Good'],
]

table = ax6.table(cellText=table_data, cellLoc='center', loc='center',
                  colWidths=[0.25, 0.25, 0.25, 0.25])
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 2.5)

# Style header
for i in range(4):
    cell = table[(0, i)]
    cell.set_facecolor('#2196F3')
    cell.set_text_props(weight='bold', color='white', fontsize=12)

# Style metric names
for i in range(1, 6):
    cell = table[(i, 0)]
    cell.set_facecolor('#E3F2FD')
    cell.set_text_props(weight='bold')

# Highlight best values
table[(4, 2)].set_facecolor('#C8E6C9')  # Phase 3 falls
table[(5, 2)].set_facecolor('#C8E6C9')  # Phase 3 stability

plt.savefig('teacher_presentation.png', dpi=200, bbox_inches='tight', facecolor='white')
print("✓ Created teacher_presentation.png")

# Create summary text
with open('teacher_summary.txt', 'w', encoding='utf-8') as f:
    f.write("MOTION-LANGUAGE CONTROL - TRAINING RESULTS\n")
    f.write("=" * 60 + "\n\n")

    f.write("OVERVIEW:\n")
    f.write("-" * 60 + "\n")
    f.write("Trained humanoid robot to follow natural language instructions\n")
    f.write("using direct motion-language learning with stability enhancements.\n\n")

    f.write("TRAINING PHASES:\n")
    f.write("-" * 60 + "\n")
    f.write("Phase 2: Slow Walking (0.3 m/s)\n")
    f.write("Phase 3: Normal Walking (0.5 m/s)\n")
    f.write("Phase 4: Endurance Training\n\n")

    f.write("KEY RESULTS:\n")
    f.write("-" * 60 + "\n")
    f.write("✓ 100% task success rate across all walking phases\n")
    f.write("✓ Mean episode reward: 3,400-3,600\n")
    f.write("✓ Episode survival: ~200 steps per episode\n")
    f.write("✓ Best stability: Phase 3 (1.5 falls/episode)\n\n")

    f.write("DETAILED METRICS:\n")
    f.write("-" * 60 + "\n")
    f.write(f"{'Phase':<25} {'Total Reward':<15} {'Falls':<10} {'Length':<10}\n")
    f.write("-" * 60 + "\n")
    f.write(f"{'Phase 2: Slow Walk':<25} {3629.79:<15.0f} {13.4:<10.1f} {218:<10.0f}\n")
    f.write(f"{'Phase 3: Normal Walk':<25} {3488.68:<15.0f} {1.5:<10.1f} {205:<10.0f}\n")
    f.write(f"{'Phase 4: Endurance':<25} {3417.52:<15.0f} {2.7:<10.1f} {200:<10.0f}\n\n")

    f.write("CONCLUSION:\n")
    f.write("-" * 60 + "\n")
    f.write("Successfully trained humanoid to walk forward following natural\n")
    f.write("language instructions. Phase 3 achieved best balance between\n")
    f.write("performance and stability.\n")

print("✓ Created teacher_summary.txt")
print("\n" + "=" * 60)
print("FILES READY FOR TEACHER:")
print("  1. teacher_presentation.png - Visual results")
print("  2. teacher_summary.txt - Text summary")
print("=" * 60)