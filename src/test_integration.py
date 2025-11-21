import sys
from pathlib import Path
import numpy as np

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root / "src"))

# FIXED: Import from agents package
try:
    from agents.hierarchical_agent import EnhancedMotionLanguageAgent
except ImportError:
    # Fallback just in case file structure changes
    from hierarchical_agent import EnhancedMotionLanguageAgent


def test_full_stack():
    print("Testing Full Agent Stack with Stability Gate...")

    # Initialize Agent
    agent = EnhancedMotionLanguageAgent(
        env_name="Humanoid-v4",
        use_stability_focus=True
    )

    # Train for a very short burst to check logs
    print("Starting training loop...")

    # FIXED: "render_training" -> "record_training_videos"
    agent.train_on_instruction_stable(
        instruction="walk forward",
        total_timesteps=2000,  # Short run
        n_envs=1,
        record_training_videos=False
    )
    print("✓ Training loop ran without errors.")
    print("Check the logs above. If 'Language Reward' is low (near 0) when falling, it works.")


if __name__ == "__main__":
    test_full_stack()