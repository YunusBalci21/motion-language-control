import sys
from pathlib import Path
from stable_baselines3 import PPO

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
phase1_paths = [
    project_root / "checkpoints/phase1_stand/final_model_stand_still.zip",
    project_root / "src/checkpoints/phase1_stand/final_model_stand_still.zip",
    Path("checkpoints/phase1_stand/final_model_stand_still.zip"),
    Path("../checkpoints/phase1_stand/final_model_stand_still.zip"),
]

phase1_model = None
for path in phase1_paths:
    if path.exists():
        phase1_model = str(path)
        print(f"Found Phase 1 model: {phase1_model}")
        break

if not phase1_model:
    print("Error: Phase 1 model not found. Searched:")
    for p in phase1_paths:
        print(f"  - {p}")
    print("\nStarting Phase 2 from scratch...")

# Train Phase 2
print("\n" + "="*60)
print("PHASE 2: Slow Walking")
print("="*60)

agent = EnhancedMotionLanguageAgent(
    env_name="Humanoid-v4",
    use_stability_focus=True
)

if phase1_model:
    print(f"Loading Phase 1 model...")
    dummy_env = agent.create_training_environment(n_envs=1)
    agent.rl_agent = PPO.load(phase1_model, env=dummy_env, device=agent.device)
    agent.rl_agent.learning_rate = 5e-5
    print("✓ Loaded Phase 1 model")

print("Training slow walking...")
agent.train_on_instruction_stable(
    instruction="walk forward slowly",
    total_timesteps=200000,
    n_envs=4,
    save_path="./checkpoints/phase2_slow_walk",
    record_training_videos=False
)

print("\n✓ Phase 2 Complete")
print("Model saved to ./checkpoints/phase2_slow_walk/")
