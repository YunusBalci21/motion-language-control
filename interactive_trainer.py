#!/usr/bin/env python3
import sys
import time
import json
import torch
import numpy as np
from pathlib import Path

# Add src to system path so we can import internal modules
sys.path.append("src")

# Import our new local fixes and tools
from motion_bridge_fix import patch_motion_tokenizer
from dynamic_physics_wrapper import DynamicPhysicsWrapper
from llm_training_coach import TrainingCoach

# Import original agent components from src
from models.motion_tokenizer import MotionTokenizer
from agents.hierarchical_agent import EnhancedMotionLanguageAgent


class InteractiveTrainer:
    def __init__(self, env_name="HalfCheetah-v4"):
        print(f"🤖 Initializing Interactive Coach for {env_name}...")

        # 1. Apply Patch for Cheetah/Ant
        patch_motion_tokenizer(MotionTokenizer)

        # 2. Initialize Components
        self.env_name = env_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.agent = EnhancedMotionLanguageAgent(env_name=env_name, device=self.device)
        self.coach = TrainingCoach()

        # 3. Default Physics Parameters
        self.current_params = {
            "forward_reward_weight": 2.0,
            "stability_weight": 1.0,
            "energy_penalty": 0.001,
            "target_speed": 1.0,
            "action_magnitude": 1.0
        }

        # 4. Monkey Patch the Agent's environment creator to use our DynamicWrapper
        self._patch_agent_env_creator()

    def _patch_agent_env_creator(self):
        """Injects the DynamicPhysicsWrapper into the agent"""

        def make_dynamic_env(instruction, **kwargs):
            import gymnasium as gym
            from stable_baselines3.common.monitor import Monitor

            # Create base Gym env
            env = gym.make(self.env_name, render_mode="rgb_array")
            env = Monitor(env)

            # Apply OUR Dynamic Wrapper
            env = DynamicPhysicsWrapper(
                env,
                self.agent.motion_tokenizer,
                instruction=instruction,
                record_video=kwargs.get('record_video', False),
                video_path=kwargs.get('video_path', None)
            )

            # Initialize with current dynamic params
            env.update_parameters(self.current_params)

            return env

        # Overwrite the method on the instance
        self.agent.make_single_env = make_dynamic_env

    def run_training_session(self, instruction, steps=5000):
        """Runs a short training burst"""
        print(f"\n🏋️ Training for {steps} steps on '{instruction}'...")
        print(f"   Params: {json.dumps(self.current_params, indent=2)}")

        save_path = f"./interactive_sessions/{int(time.time())}"

        # Train
        model_path = self.agent.train_on_instruction_stable(
            instruction=instruction,
            total_timesteps=steps,
            save_path=save_path,
            n_envs=1,
            verbose=1
        )

        # Evaluate & Record
        print("\n🎬 Generating Preview Video...")
        results = self.agent.evaluate_instruction(
            instruction=instruction,
            model_path=model_path,
            num_episodes=1,
            record_video=True,
            video_path=f"{save_path}/videos"
        )

        # Extract Key Metrics
        metrics = {
            "vx": float(np.mean(results['episode_data'][0]['vx_list'])),
            "success_rate": results['mean_success_rate'],
            "falls": results['mean_fall_count']
        }

        return metrics, f"{save_path}/videos"

    def chat_loop(self):
        print("\n" + "=" * 60)
        print(f"🎓 Physics Coaching Interface ({self.env_name})")
        print("=" * 60)

        while True:
            # Outer loop allows changing the instruction
            instruction = input("\nTarget Instruction (e.g. 'walk forward', or 'q' to quit): ")
            if instruction.lower() in ['q', 'quit', 'exit']:
                break

            # Initial Training for the new instruction
            metrics, video_path = self.run_training_session(instruction, steps=3000)

            while True:
                # Inner loop handles coaching for the current instruction
                print("\n" + "-" * 40)
                print(f"📊 Result Metrics for '{instruction}':")
                print(f"   • Speed: {metrics['vx']:.3f} m/s")
                print(f"   • Success: {metrics['success_rate'] * 100:.1f}%")
                print(f"   • Video saved at: {video_path}")
                print("-" * 40)

                feedback = input("\nCoach Feedback (or 'new' to change task, 'q' to quit): ")

                if feedback.lower() == 'new':
                    break  # Break inner loop, go back to instruction input
                if feedback.lower() in ['q', 'quit', 'exit']:
                    sys.exit(0)

                # LLM Analysis
                print("\n🤔 Consulting LLM Coach...")
                new_params = self.coach.analyze_feedback(
                    instruction, feedback, metrics, self.current_params
                )

                print("\n💡 Proposed Adjustment:")
                # Show diff
                for k, v in new_params.items():
                    if k in self.current_params and self.current_params[k] != v:
                        print(f"   {k}: {self.current_params[k]} -> {v}")

                confirm = input("Apply changes and retrain? (y/n): ")
                if confirm.lower() == 'y':
                    self.current_params = new_params
                    metrics, video_path = self.run_training_session(instruction, steps=5000)
                else:
                    print("Skipping training.")


if __name__ == "__main__":
    # You can change this to "HalfCheetah-v4" or "Ant-v4"
    trainer = InteractiveTrainer(env_name="HalfCheetah-v4")
    trainer.chat_loop()