import sys
from pathlib import Path
import numpy as np

# Find project root
current = Path.cwd()
if current.name == 'src':
    project_root = current.parent
else:
    project_root = current

src_path = project_root / "src"
sys.path.insert(0, str(src_path))

from agents.hierarchical_agent import EnhancedMotionLanguageAgent

# Find phase 1 checkpoint
search_paths = [
    project_root / "checkpoints/phase1_stand/final_model_stand_still.zip",
    project_root / "src/checkpoints/phase1_stand/final_model_stand_still.zip",
    Path("checkpoints/phase1_stand/final_model_stand_still.zip"),
    Path("../checkpoints/phase1_stand/final_model_stand_still.zip"),
]

model_path = None
for path in search_paths:
    if path.exists():
        model_path = str(path)
        print(f"Found model: {model_path}")
        break

if not model_path:
    print("Error: Phase 1 model not found. Searched:")
    for p in search_paths:
        print(f"  - {p}")
    sys.exit(1)

print("\nEvaluating Phase 1 model...")
print("="*60)

agent = EnhancedMotionLanguageAgent(
    env_name="Humanoid-v4",
    use_stability_focus=True
)

results = agent.evaluate_instruction(
    instruction="stand still",
    model_path=model_path,
    num_episodes=5,
    deterministic=True
)

print("\n" + "="*60)
print("PHASE 1 EVALUATION RESULTS")
print("="*60)

if results:
    print(f"\nRewards:")
    print(f"  Mean Total:    {results['mean_total_reward']:.2f}")
    print(f"  Mean Language: {results['mean_language_reward']:.2f}")
    print(f"  Mean Env:      {results['mean_env_reward']:.2f}")
    
    print(f"\nPerformance:")
    print(f"  Episode Length:  {results['mean_max_stable_steps']:.0f} steps")
    print(f"  Success Rate:    {results['episode_success_rate']*100:.1f}%")
    print(f"  Falls per Ep:    {results['mean_fall_count']:.2f}")
    
    print(f"\nEpisode-by-episode rewards:")
    for i, ep in enumerate(results['episode_data'][:5], 1):
        print(f"  Episode {i}: {ep['total_reward']:.2f} reward, "
              f"{len(ep['vx_list'])} steps, "
              f"{ep['stability_metrics']['fall_count']} falls")
    
    print("\n" + "="*60)
    
    if results['mean_max_stable_steps'] > 300:
        print("✓ Phase 1 successful! Agent can stand.")
        print("Ready for Phase 2 (slow walking)")
    else:
        print("⚠ Phase 1 needs more training. Episodes too short.")
        print("Consider training Phase 1 longer.")
else:
    print("Evaluation failed")
