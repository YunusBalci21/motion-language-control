# train_ant_sac.py
from src.agents.hierarchical_agent import EnhancedMotionLanguageAgent
from stable_baselines3 import SAC
import os

print("🐜 Training Ant with SAC (automatic entropy tuning)...")

# Create agent
agent = EnhancedMotionLanguageAgent('Ant-v4', device='cuda')

# Create single wrapped environment (SAC doesn't use vectorized envs)
env = agent.make_single_env(
    instruction='walk forward',
    language_reward_weight=0.95,
    record_video=False,
    video_path=None,
    render_mode=None
)

print("Environment created")

# Create SAC model
print("Creating SAC agent...")
model = SAC(
    "MlpPolicy",
    env,
    learning_rate=3e-4,
    buffer_size=int(1e6),
    batch_size=256,
    ent_coef='auto',  # ✨ AUTOMATIC ENTROPY TUNING ✨
    gamma=0.99,
    tau=0.005,
    train_freq=1,
    gradient_steps=1,
    policy_kwargs=dict(
        net_arch=[256, 256],
        log_std_init=-3,
    ),
    verbose=1,
    device='cuda',
    tensorboard_log="./logs/"
)

# Train
print("\n🚀 Training for 500k steps with SAC...")
print("SAC maintains exploration - no more policy collapse!\n")

os.makedirs('./models', exist_ok=True)

model.learn(
    total_timesteps=500000,
    log_interval=10,
    progress_bar=True
)

# Save
model.save("models/ant_sac_walk")
print("\n✅ Done! Model saved to models/ant_sac_walk.zip")