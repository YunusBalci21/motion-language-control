# record_videos.py
import sys
from pathlib import Path

# The classes are defined in train_motion_language_agent.py
# We need to import from there
sys.path.insert(0, str(Path(__file__).parent))

# Import the classes directly from the training script
from train_motion_language_agent import EnhancedMotionLanguageAgent

print("Recording demonstration videos...")

agent = EnhancedMotionLanguageAgent("Humanoid-v4", use_stability_focus=True)

agent.evaluate_instruction(
    instruction="walk forward",
    model_path="enhanced_experiments/direct_motion_language_20251014_014343/checkpoints/step_01_walk_forward/final_model_walk_forward.zip",
    num_episodes=5,
    record_video=True,
    video_path="demo_videos_old1",
    deterministic=True,
    render=False
)

print("Videos saved to: ./demo_videos_old1/")