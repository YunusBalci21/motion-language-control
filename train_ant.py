from src.agents.hierarchical_agent import EnhancedMotionLanguageAgent
import os

os.makedirs('./ant_final', exist_ok=True)

print("🐜 Training Ant with high exploration...")

agent = EnhancedMotionLanguageAgent('Ant-v4', device='cuda')

# ANT-SPECIFIC hyperparameters (more exploration!)
agent.training_config['learning_rate'] = 3e-4  # Lower LR
agent.training_config['ent_coef'] = 0.05  # HIGH entropy (5x humanoid)
agent.training_config['n_steps'] = 4096  # More steps per update
agent.training_config['batch_size'] = 128  # Larger batches
agent.training_config['log_std_init'] = 1.0  # More action noise

model = agent.train_on_instruction(
    'walk forward',
    total_timesteps=500000,  # Train longer
    n_envs=4,
    language_reward_weight=0.95,  # High language weight
    use_vecnormalize=True,
    save_path='./ant_final/'
)

print("\n✅ Ant training complete!")