import sys
import os
from pathlib import Path

# 1. Add the 'src' folder to the Python path so we can find 'agents'
# This ensures it works whether you run it from root or src folder
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))

# 2. Robust Import: Look in 'agents' folder first
try:
    from agents.hierarchical_agent import EnhancedMotionLanguageAgent
except ImportError:
    try:
        # Fallback: Maybe it's in the same folder?
        from hierarchical_agent import EnhancedMotionLanguageAgent
    except ImportError:
        print("CRITICAL ERROR: Could not find 'hierarchical_agent.py'.")
        print("Make sure it exists in 'src/agents/' or 'src/'.")
        sys.exit(1)


def train():
    # Initialize the agent with stability focus enabled
    agent = EnhancedMotionLanguageAgent(
        env_name="Humanoid-v4",
        use_stability_focus=True
    )

    print("Starting long training run for 'Walk Forward'...")

    # Run the training loop
    agent.train_on_instruction_stable(
        instruction="walk forward",
        total_timesteps=300000,  # Real training duration (~1-2 hours on GPU)
        n_envs=4,  # Parallel environments
        save_path="./checkpoints/walk_forward_v1",
        record_training_videos=False  # Keep False for speed
    )


if __name__ == "__main__":
    train()