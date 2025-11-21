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


def train_endurance():
    print("Phase 3: Endurance Training (The 'Slow Down' Strategy)")
    print("======================================================")
    print("Goal: Force agent to slow down to 0.4 m/s to learn long-term balance.")

    # 1. Initialize Agent
    agent = EnhancedMotionLanguageAgent(
        env_name="Humanoid-v4",
        use_stability_focus=True
    )

    # 2. Load the "Survival" Model (Phase 2 result)
    # This model knows how to be stable-ish, but runs too fast.
    pretrained_path = "./checkpoints/walk_forward_survival/final_model_walk_forward_stably.zip"

    if not os.path.exists(pretrained_path):
        print(f"Error: Survival model not found at {pretrained_path}")
        print("Please check your checkpoints folder.")
        return

    print(f"Loading survival model: {pretrained_path}")

    dummy_env = agent.create_training_environment(n_envs=1)
    agent.rl_agent = PPO.load(pretrained_path, env=dummy_env, device=agent.device)

    # 3. Hyperparameters for Endurance
    # Very low learning rate to gently adjust the gait
    agent.rl_agent.learning_rate = 5e-6
    # Lower entropy coefficient to stabilize the policy (less random twitching)
    agent.rl_agent.ent_coef = 0.0

    # 4. Run Fine-tuning on SLOW instruction
    # The "slowly" keyword triggers a target velocity of 0.4 m/s in the wrapper.
    # This acts as a speed limit, forcing the agent to stop sprinting.
    print("Starting endurance fine-tuning...")
    agent.train_on_instruction_stable(
        instruction="walk forward slowly",
        total_timesteps=300000,  # Longer run to master balance
        n_envs=4,
        save_path="./checkpoints/walk_forward_endurance",
        record_training_videos=False
    )

    print("\nEndurance training complete!")
    print("Model saved to ./checkpoints/walk_forward_endurance/")
    print("Now try visualizing this model - it should walk for much longer!")


if __name__ == "__main__":
    train_endurance()