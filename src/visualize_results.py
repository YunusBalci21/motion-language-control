import sys
import os
from pathlib import Path
from stable_baselines3 import PPO

# 1. Robust Import Setup
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))

try:
    from agents.hierarchical_agent import EnhancedMotionLanguageAgent
except ImportError:
    try:
        from hierarchical_agent import EnhancedMotionLanguageAgent
    except ImportError:
        print("CRITICAL ERROR: Could not find 'hierarchical_agent.py'.")
        sys.exit(1)


def train_survival():
    print("Phase 2: Survival Training (Fine-tuning for Balance)")
    print("====================================================")

    # 1. Initialize Agent
    agent = EnhancedMotionLanguageAgent(
        env_name="Humanoid-v4",
        use_stability_focus=True
    )

    # 2. Load the "Sprinter" Model
    # We load the model we just trained to keep the walking knowledge
    pretrained_path = "./checkpoints/walk_forward_v1/final_model_walk_forward.zip"

    if not os.path.exists(pretrained_path):
        print(f"Error: Pretrained model not found at {pretrained_path}")
        print("Please ensure Phase 1 training (train_walk.py) completed successfully.")
        return

    print(f"Loading pretrained model: {pretrained_path}")

    # We use a custom method to load and attach the model to the agent
    # This ensures we use the existing policy weights
    dummy_env = agent.create_training_environment(n_envs=1)
    agent.rl_agent = PPO.load(pretrained_path, env=dummy_env, device=agent.device)

    # 3. Adjust Hyperparameters for Stability
    # We lower the learning rate to "polish" the behavior rather than change it drastically
    agent.rl_agent.learning_rate = 1e-5

    # 4. Run Fine-tuning
    # We use the same instruction but the stability wrapper will now enforce
    # stricter penalties because the agent is moving faster.
    print("Starting survival fine-tuning...")
    agent.train_on_instruction_stable(
        instruction="walk forward stably",  # Adding "stably" might trigger extra reward logic if configured
        total_timesteps=200000,
        n_envs=4,
        save_path="./checkpoints/walk_forward_survival",
        record_training_videos=False
    )

    print("\nSurvival training complete!")
    print("Model saved to ./checkpoints/walk_forward_survival/")


if __name__ == "__main__":
    train_survival()