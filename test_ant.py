from src.agents.hierarchical_agent import EnhancedMotionLanguageAgent
import os

os.makedirs('./ant_videos', exist_ok=True)

agent = EnhancedMotionLanguageAgent('Ant-v4', device='cuda')
model = './ant_checkpoints/final_model_walk_forward.zip'

print('Recording Ant walking forward...\n')

results = agent.evaluate_instruction(
    'walk forward',
    model_path=model,
    num_episodes=3,
    record_video=True,
    video_path='./ant_videos',
    render=True,
    deterministic=True
)

print('\nDone! Check ./ant_videos/ for videos')

import glob
videos = glob.glob('./ant_videos/*.mp4')
if videos:
    print(f'Found {len(videos)} videos:')
    for v in videos:
        print(f'  - {v}')
else:
    print('No videos found')
