from src.agents.hierarchical_agent import EnhancedMotionLanguageAgent
import os

os.makedirs('./humanoid_checkpoints', exist_ok=True)

print("🤖 Training Humanoid to walk forward...")
print("Note: Humanoid is complex, this will take longer than Ant\n")

agent = EnhancedMotionLanguageAgent('Humanoid-v4', device='cuda')

model = agent.train_on_instruction(
    'walk forward',
    total_timesteps=300000,  # More timesteps for humanoid
    n_envs=4,
    language_reward_weight=0.9,
    use_vecnormalize=True,
    save_path='./humanoid_checkpoints/'
)

print("\n✅ Training complete!")